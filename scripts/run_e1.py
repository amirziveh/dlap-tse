#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_e1.py — DLAP-TSE Phase 2: E1 benchmark battery
====================================================
Evaluates the benchmark models on TSE with the CPZ (2024) protocol:
  FF5, q-factor, PCA (5 factors), LASSO (char SDF), Market (CAPM reference).

Protocol:
  - Common period 2008-07..2026-06 (where FF5 & q factors exist)
  - Rolling: train 60 months -> test 12 months, step 12 (12 OOS windows)
  - Factor SDFs: w = max-Sharpe weights (shrunk Sigma^-1 mu) from train; OOS
    SDF portfolio return r_p = w'F
  - LASSO: linear char SDF, w = LASSO coefs (lambda by 3-fold CV in train);
    r_p = sum_i M_i R^e_i / sum_i M_i,  M_i = 1 - w'x_i
  - Metrics (pooled OOS): annualized Sharpe, XS R^2 (n-weighted), RMS & max
    pricing errors (alpha = E[M R^e]), plus HJ bound of test assets.

Outputs:
  results/e1_benchmarks.csv — one row per model
  results/e1_pooled_series.csv — pooled OOS SDF portfolio returns per model
"""
import csv
import os
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_core import (load_npz, load_rf, load_factors_ff5, load_factors_q,
                       shrunk_cov, max_sharpe_weights, sharpe_ann, xs_r2,
                       ols_betas, rolling_windows, UNKNOWN)

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

N_PCA = 5
MIN_TRAIN_OBS = 12
MIN_TEST_OBS = 6


def eval_factor_model(F_all, R_exc, months, windows, name):
    """F_all: (T x K) factor excess returns aligned to `months`.
    Returns dict with pooled metrics + per-window sharpe list."""
    T = len(months)
    pooled_rp = []
    r2s, r2_n = [], []
    alphas = []
    ss_res = ss_tot = 0.0
    per_win_sharpe = []
    n_windows = 0
    for tr_idx, te_idx in windows:
        F_tr, F_te = F_all[tr_idx], F_all[te_idx]
        R_tr, R_te = R_exc[tr_idx], R_exc[te_idx]
        w = max_sharpe_weights(F_tr)
        rp = F_te @ w
        rp = rp[np.isfinite(rp)]
        if len(rp) < 6:
            continue
        n_windows += 1
        pooled_rp.append(rp)
        per_win_sharpe.append(sharpe_ann(rp))

        # XS R^2 over stocks (beta * lambda); skip thin stocks (near-zero
        # train variance -> unstable betas that overflow)
        lam = F_tr.mean(axis=0)
        preds, reals = [], []
        for i in range(R_tr.shape[1]):
            ri_tr, ri_te = R_tr[:, i], R_te[:, i]
            if np.nanvar(ri_tr) < 1e-6:
                continue
            beta = ols_betas(ri_tr, F_tr)
            if beta is None:
                continue
            m_tr = np.isfinite(ri_tr)
            m_te = np.isfinite(ri_te)
            if m_te.sum() < MIN_TEST_OBS:
                continue
            preds.append(beta[1:] @ lam)
            reals.append(ri_te[m_te].mean())
        r2 = xs_r2(preds, reals)
        if not math.isnan(r2):
            r2s.append(r2)
            r2_n.append(len(reals))

        # pricing errors: M_t = 1 - w'F_t  (F excess, zero-mean)
        F_te_c = F_te - F_tr.mean(axis=0)
        M = 1.0 - F_te_c @ w
        for i in range(R_te.shape[1]):
            ri = R_te[:, i]
            m = np.isfinite(ri) & np.isfinite(M)
            if m.sum() < MIN_TEST_OBS:
                continue
            alphas.append(float((M[m] * ri[m]).mean()))

        # EV (explained return variation, same construction as the DL-SDF):
        # test-sample OLS of stock returns on the SDF portfolio return
        for i in range(R_te.shape[1]):
            ri = R_te[:, i]
            m = np.isfinite(ri) & np.isfinite(rp)
            if m.sum() < 8 or np.var(rp[m]) < 1e-12:
                continue
            b = np.polyfit(rp[m], ri[m], 1)
            e = ri[m] - (b[0] * rp[m] + b[1])
            ss_res += float((e ** 2).sum())
            ss_tot += float(((ri[m] - ri[m].mean()) ** 2).sum())
    if not pooled_rp:
        return {"name": name, "error": "no windows"}
    rp_all = np.concatenate(pooled_rp)
    return {
        "name": name,
        "n_windows": n_windows,
        "n_oos_months": len(rp_all),
        "sharpe_pooled": sharpe_ann(rp_all),
        "sharpe_mean_win": float(np.mean(per_win_sharpe)),
        "r2": float(np.average(r2s, weights=r2_n)) if r2s else math.nan,
        "ev": (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "rms_alpha_pct": float(np.sqrt(np.mean(np.square(alphas)))) * 100 if alphas else math.nan,
        "max_alpha_pct": float(np.max(np.abs(alphas))) * 100 if alphas else math.nan,
        "pooled_rp": rp_all,
    }


def lasso_cv(X, y, n_folds=3, n_lambdas=15):
    """Coordinate-descent LASSO with k-fold CV; returns (w, best_lambda).
    Defensive: clips X to [-10, 10], breaks on non-finite, warm-starts the
    lambda path."""
    X = np.clip(X, -10.0, 10.0)

    def soft_thresh(z, lam):
        return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

    def fit_lasso(X, y, lam, w0=None, iters=400, tol=1e-7):
        """Cyclic coordinate descent (Gauss-Seidel) with incremental residual.
        Converges for any lambda > 0 even with correlated, unstandardized X."""
        n, p = X.shape
        col_energy = np.maximum(np.einsum("ij,ij->j", X, X) / n, 1e-8)
        w = np.zeros(p) if w0 is None else w0.copy()
        r = y - X @ w  # residual
        for _ in range(iters):
            max_change = 0.0
            for j in range(p):
                rho = X[:, j] @ r / n + w[j] * col_energy[j]
                if not np.isfinite(rho):
                    return w
                w_new_j = soft_thresh(rho, lam) / col_energy[j]
                d = w_new_j - w[j]
                if d != 0.0:
                    r -= X[:, j] * d
                    w[j] = w_new_j
                    if abs(d) > max_change:
                        max_change = abs(d)
            if max_change < tol:
                break
        return w

    n = X.shape[0]
    lam_max = float(np.max(np.abs(X.T @ y) / n))
    lams = np.geomspace(lam_max * 0.002, lam_max, n_lambdas)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    best_lam, best_mse = lams[0], np.inf
    # warm starts: fit largest lambda first, reuse as init
    last_w = None
    for lam in lams:
        errs = []
        for f in range(n_folds):
            va = folds[f]
            tr = np.concatenate([folds[g] for g in range(n_folds) if g != f])
            w = fit_lasso(X[tr], y[tr], lam, w0=last_w)
            pred = X[va] @ w
            errs.append(float(np.mean((y[va] - pred) ** 2)))
        mse = float(np.mean(errs))
        if mse < best_mse:
            best_mse, best_lam = mse, lam
        last_w = w
    w = fit_lasso(X, y, best_lam)
    return w, best_lam


def eval_lasso(R_exc, X, months, windows, name="LASSO"):
    pooled_rp, r2s, r2_n, alphas = [], [], [], []
    ss_res = ss_tot = 0.0
    per_win_sharpe = []
    n_windows = 0
    lambdas = []
    for tr_idx, te_idx in windows:
        R_tr, R_te = R_exc[tr_idx], R_exc[te_idx]
        X_tr, X_te = X[tr_idx], X[te_idx]
        # pooled train sample (drop missing)
        m_tr = np.isfinite(R_tr) & np.isfinite(X_tr).all(axis=2)
        if m_tr.sum() < 1000:
            continue
        y = R_tr[m_tr]
        Xs = X_tr[m_tr]
        w, lam = lasso_cv(Xs, y)
        lambdas.append(lam)
        # OOS SDF portfolio return per test month
        rp = []
        for t in range(R_te.shape[0]):
            M = 1.0 - X_te[t] @ w
            R = R_te[t]
            m = np.isfinite(M) & np.isfinite(R)
            if m.sum() < 5:
                continue
            num = float((M[m] * R[m]).sum())
            den = float(M[m].sum())
            rp.append(num / den if den != 0 else math.nan)
        rp = np.array([v for v in rp if v == v])
        if len(rp) < 6:
            continue
        n_windows += 1
        pooled_rp.append(rp)
        per_win_sharpe.append(sharpe_ann(rp))
        # XS R^2: pred = w'xbar_i
        preds, reals = [], []
        for i in range(R_te.shape[1]):
            xi = X_te[:, i]
            ri = R_te[:, i]
            m = np.isfinite(xi).all(axis=1) & np.isfinite(ri)
            if m.sum() < MIN_TEST_OBS:
                continue
            preds.append(w @ xi[m].mean(axis=0))
            reals.append(ri[m].mean())
        r2 = xs_r2(preds, reals)
        if not math.isnan(r2):
            r2s.append(r2)
            r2_n.append(len(reals))
        # pricing errors
        for i in range(R_te.shape[1]):
            M = 1.0 - X_te[:, i] @ w
            ri = R_te[:, i]
            m = np.isfinite(M) & np.isfinite(ri)
            if m.sum() < MIN_TEST_OBS:
                continue
            alphas.append(float((M[m] * ri[m]).mean()))
        # EV (same construction as DL-SDF)
        for i in range(R_te.shape[1]):
            ri = R_te[:, i]
            m = np.isfinite(ri) & np.isfinite(rp)
            if m.sum() < 8 or np.var(rp[m]) < 1e-12:
                continue
            b = np.polyfit(rp[m], ri[m], 1)
            e = ri[m] - (b[0] * rp[m] + b[1])
            ss_res += float((e ** 2).sum())
            ss_tot += float(((ri[m] - ri[m].mean()) ** 2).sum())
    if not pooled_rp:
        return {"name": name, "error": "no windows"}
    rp_all = np.concatenate(pooled_rp)
    return {
        "name": name,
        "n_windows": n_windows,
        "n_oos_months": len(rp_all),
        "sharpe_pooled": sharpe_ann(rp_all),
        "sharpe_mean_win": float(np.mean(per_win_sharpe)),
        "r2": float(np.average(r2s, weights=r2_n)) if r2s else math.nan,
        "ev": (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "rms_alpha_pct": float(np.sqrt(np.mean(np.square(alphas)))) * 100 if alphas else math.nan,
        "max_alpha_pct": float(np.max(np.abs(alphas))) * 100 if alphas else math.nan,
        "n_lasso_selected": int((w != 0).sum()),
        "lasso_lambda": float(np.mean(lambdas)),
        "pooled_rp": rp_all,
    }


def hj_bound(R_exc, windows):
    """Max Sharpe of test assets: sqrt(mu' Sigma^-1 mu), train cov, per window avg.
    Only stocks with >=12 obs in the window; ridge-stabilized cov."""
    bounds = []
    for tr_idx, te_idx in windows:
        R_tr = R_exc[tr_idx]
        keep = np.isfinite(R_tr).sum(axis=0) >= 12
        Rt = R_tr[:, keep]
        Rt = np.where(np.isfinite(Rt), Rt, 0.0)
        mu = Rt.mean(axis=0)
        S = shrunk_cov(Rt)
        # ridge: floor the diagonal (thin stocks can still be near-singular)
        S += 1e-4 * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        try:
            b = float(np.sqrt(mu @ np.linalg.solve(S, mu)))
        except np.linalg.LinAlgError:
            continue
        if math.isfinite(b):
            bounds.append(b)
    return float(np.mean(bounds)) if bounds else math.nan


def main():
    print("Loading data ...")
    arr, dates, variables, tickers = load_npz()
    rf = load_rf()
    ff5 = load_factors_ff5()
    q = load_factors_q()

    # common months: npz dates ∩ FF5 ∩ q
    common = [d for d in dates if (int(d[:4]), int(d[5:7])) in ff5 and
              (int(d[:4]), int(d[5:7])) in q]
    print(f"  common period: {common[0]} .. {common[-1]}  ({len(common)} months)")
    T = len(common)
    pos = {d: i for i, d in enumerate(dates)}
    idx = [pos[d] for d in common]

    # excess returns [T x N]
    # NOTE: sentinel -99.99 stored as float32 -> float64 gives -99.98999786;
    # equality against -99.99 FAILS. Mask by threshold (real returns >= -1,
    # winsorized; z-chars clipped to [-10,10]).
    R_raw = arr[idx, :, 0].astype(float)
    R = R_raw.copy()
    R[R < -50.0] = np.nan
    rf_v = np.array([rf.get(d, np.nan) for d in common])
    R_exc = R - rf_v[:, None]
    N = R.shape[1]
    print(f"  {N} stocks, rf aligned, excess returns {R_exc.shape}")

    # z-chars [T x N x 20]
    X_raw = arr[idx, :, 1:].astype(float)
    X = X_raw.copy()
    X[X < -50.0] = np.nan

    # factor matrices
    F_ff5 = np.array([[ff5[(int(d[:4]), int(d[5:7]))][k] for k in
                       ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]] for d in common])
    F_q = np.array([[q[(int(d[:4]), int(d[5:7]))][k] for k in
                     ["Mkt_RF", "ME", "IA", "ROE"]] for d in common])
    F_mkt = F_ff5[:, :1]

    windows = list(rolling_windows(list(range(T)), train=60, test=12))
    print(f"  {len(windows)} rolling windows (train 60 / test 12)")

    # HJ bound of test assets
    hj = hj_bound(R_exc, windows)
    print(f"  HJ bound (max Sharpe of test assets): {hj:.3f}")

    results = []
    print("\n  running Market (CAPM reference) ...")
    results.append(eval_factor_model(F_mkt, R_exc, common, windows, "Market"))
    print("  running FF5 ...")
    results.append(eval_factor_model(F_ff5, R_exc, common, windows, "FF5"))
    print("  running q-factor ...")
    results.append(eval_factor_model(F_q, R_exc, common, windows, "q-factor"))
    print("  running PCA ...")
    # PCA factors from train returns (impute missing w/ 0), top-5
    rp_all_pca, r2s, r2_n, alphas, per_win = [], [], [], [], []
    ss_res = ss_tot = 0.0
    n_win = 0
    for tr_idx, te_idx in windows:
        R_tr, R_te = R_exc[tr_idx], R_exc[te_idx]
        Rt = np.where(np.isfinite(R_tr), R_tr, 0.0)
        mu_t = Rt.mean(axis=0)
        S = shrunk_cov(Rt)
        _, _, Vt = np.linalg.svd(S)
        V = Vt[:N_PCA].T  # N x K loadings
        F_tr = Rt @ V
        F_te = np.where(np.isfinite(R_te), R_te, 0.0) @ V
        w = max_sharpe_weights(F_tr)
        rp = F_te @ w
        rp = rp[np.isfinite(rp)]
        if len(rp) < 6:
            continue
        n_win += 1
        rp_all_pca.append(rp)
        per_win.append(sharpe_ann(rp))
        lam = F_tr.mean(axis=0)
        preds, reals = [], []
        for i in range(R_tr.shape[1]):
            v = np.nanvar(R_tr[:, i])
            if not np.isfinite(v) or v < 1e-6:
                continue
            beta = ols_betas(R_tr[:, i], F_tr)
            if beta is None:
                continue
            m_te = np.isfinite(R_te[:, i])
            if m_te.sum() < MIN_TEST_OBS:
                continue
            preds.append(beta[1:] @ lam)
            reals.append(R_te[m_te, i].mean())
        r2 = xs_r2(preds, reals)
        if not math.isnan(r2):
            r2s.append(r2)
            r2_n.append(len(reals))
        F_te_c = F_te - F_tr.mean(axis=0)
        M = 1.0 - F_te_c @ w
        for i in range(R_te.shape[1]):
            m = np.isfinite(R_te[:, i]) & np.isfinite(M)
            if m.sum() < MIN_TEST_OBS:
                continue
            alphas.append(float((M[m] * R_te[m, i]).mean()))
        for i in range(R_te.shape[1]):
            ri = R_te[:, i]
            m = np.isfinite(ri) & np.isfinite(rp)
            if m.sum() < 8 or np.var(rp[m]) < 1e-12:
                continue
            b = np.polyfit(rp[m], ri[m], 1)
            e = ri[m] - (b[0] * rp[m] + b[1])
            ss_res += float((e ** 2).sum())
            ss_tot += float(((ri[m] - ri[m].mean()) ** 2).sum())
    rp_pca = np.concatenate(rp_all_pca)
    results.append({
        "name": "PCA(5)",
        "n_windows": n_win,
        "n_oos_months": len(rp_pca),
        "sharpe_pooled": sharpe_ann(rp_pca),
        "sharpe_mean_win": float(np.mean(per_win)),
        "r2": float(np.average(r2s, weights=r2_n)) if r2s else math.nan,
        "ev": (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "rms_alpha_pct": float(np.sqrt(np.mean(np.square(alphas)))) * 100 if alphas else math.nan,
        "max_alpha_pct": float(np.max(np.abs(alphas))) * 100 if alphas else math.nan,
        "pooled_rp": rp_pca,
    })
    print("  running LASSO ...")
    lasso_res = eval_lasso(R_exc, X, common, windows, "LASSO")
    results.append(lasso_res)

    # ── write outputs ──────────────────────────────────────────────────
    fields = ["name", "n_windows", "n_oos_months", "sharpe_pooled",
              "sharpe_mean_win", "r2", "ev", "rms_alpha_pct", "max_alpha_pct"]
    with open(RES / "e1_benchmarks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            if "error" in r:
                continue
            w.writerow({k: r.get(k, "") for k in fields})
    with open(RES / "e1_pooled_series.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "oos_return"])
        for r in results:
            if "error" in r or "pooled_rp" not in r:
                continue
            for v in r["pooled_rp"]:
                w.writerow([r["name"], f"{v:.6f}"])

    # ── print table ────────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print(f"{'model':<10}{'win':>5}{'oosM':>6}{'Sharpe':>9}{'Sharpe(w)':>10}"
          f"{'XS-R2':>9}{'EV':>8}{'RMS_alpha%':>11}{'max_alpha%':>11}")
    print("-" * 104)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<10} ERROR {r['error']}")
            continue
        print(f"{r['name']:<10}{r['n_windows']:>5}{r['n_oos_months']:>6}"
              f"{r['sharpe_pooled']:>9.3f}{r['sharpe_mean_win']:>10.3f}"
              f"{r['r2']:>9.4f}{r['ev']:>8.4f}"
              f"{r['rms_alpha_pct']:>11.3f}{r['max_alpha_pct']:>11.3f}")
    if "lasso_lambda" in lasso_res:
        print(f"\n  LASSO: lambda={lasso_res['lasso_lambda']:.5f}, "
              f"nonzero chars={lasso_res.get('n_lasso_selected', '?')}/20")
    print(f"\n  HJ bound (avg max-Sharpe of test assets): {hj:.3f}")
    print("=" * 92)
    print(f"Saved -> {RES / 'e1_benchmarks.csv'} and {RES / 'e1_pooled_series.csv'}")


if __name__ == "__main__":
    main()
