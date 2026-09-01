#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ipca_proper.py — IPCA (Kelly, Pruitt & Su 2019) benchmark on the paper's protocol
=================================================================================
Replaces the earlier draft (ipca_benchmark.py), whose OOS construction was not
comparable to the paper's benchmarks.

Protocol — mirrors scripts/run_e1.py exactly:
  - Same months/windows: dates ∩ FF5 ∩ q, rolling train 60 / test 12.
  - Same panel: Char_all.npz, sentinel -99.99 -> NaN, >=30% char coverage,
    lag-aligned characteristics (x_{t-1} prices r_t), excess returns R - rf.
  - Same metric formulas as eval_factor_model (verbatim per-window copy),
    validated by reproducing the committed Market/FF5/q-factor rows and the
    inline PCA(5) row from results*/e1_benchmarks.csv before IPCA is run.

IPCA model (per window, train only):
  R_it = lambda_it' f_t + e_it,   lambda_it = Gamma' x_it  (x includes 1.0)
  Alternating least squares: given F, Gamma = lstsq([f_t (x) x_it]); given
  Gamma, f_t = (Lam'Lam)^-1 Lam'r_t per month. Init F: top-K PCA of the train
  return panel (missing -> cross-sectional mean). 100 iters, tol 1e-5.

OOS (no leakage):
  Gamma frozen at train values. Test-month factor: cross-sectional GLS update
  f_t = (Lam'Lam)^-1 Lam' r_t with Lam from test-month characteristics
  (point-in-time). Then F_tr/F_fe feed the SAME metric code as FF5/q/PCA:
  tangency weights on train factors, M = 1 - w'f, per-stock alphas, EV.

Outputs per country (results{,_tr,_pk}/):
  e1_ipca_results.csv      same schema as e1_benchmarks.csv (rows IPCA(1,3,5))
  e1_ipca_alpha_cells.csv  model,window,alpha  (rms_window_bootstrap-compatible)
  e1_ipca_pooled_series.csv
"""
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_core import (load_npz, load_rf, load_factors_ff5, load_factors_q,
                       lag_align, shrunk_cov, max_sharpe_weights, sharpe_ann,
                       xs_r2, ols_betas, rolling_windows)

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
_C = os.environ.get("DLAP_COUNTRY", "").upper()
if _C == "TR":
    DATA, RES = ROOT / "data_tr", ROOT / "results_tr"
elif _C == "PK":
    DATA, RES = ROOT / "data_pk", ROOT / "results_pk"
else:
    DATA, RES = ROOT / "data", ROOT / "results"

MIN_TEST_OBS = 6          # same as run_e1.py
KS = [1, 3, 5]


# ── verbatim per-window copy of eval_factor_model (run_e1.py L54-122) ──────
def eval_factor_window(F_tr, F_te, R_tr, R_te, widx):
    """One window. Same formulas as run_e1.eval_factor_model; returns raw
    per-window metrics plus the pieces needed for pooled aggregation."""
    w = max_sharpe_weights(F_tr)
    rp = F_te @ w
    rp = rp[np.isfinite(rp)]
    out = {"n_oos": len(rp), "sharpe": sharpe_ann(rp) if len(rp) >= 6 else math.nan,
           "pooled_rp": rp, "alpha_cells": [], "r2": math.nan, "r2_n": 0,
           "ss_res": 0.0, "ss_tot": 0.0, "alphas": []}
    if len(rp) < 6:
        return out

    # XS R^2 over stocks (identical construction)
    lam = F_tr.mean(axis=0)
    preds, reals = [], []
    for i in range(R_tr.shape[1]):
        ri_tr, ri_te = R_tr[:, i], R_te[:, i]
        if np.nanvar(ri_tr) < 1e-6:
            continue
        beta = ols_betas(ri_tr, F_tr)
        if beta is None:
            continue
        m_te = np.isfinite(ri_te)
        if m_te.sum() < MIN_TEST_OBS:
            continue
        preds.append(beta[1:] @ lam)
        reals.append(ri_te[m_te].mean())
    r2 = xs_r2(preds, reals)
    out["r2"] = r2
    out["r2_n"] = len(reals)

    # pricing errors: M_t = 1 - w'F_t (F excess, centered by train mean)
    F_te_c = F_te - F_tr.mean(axis=0)
    M = 1.0 - F_te_c @ w
    for i in range(R_te.shape[1]):
        ri = R_te[:, i]
        m = np.isfinite(ri) & np.isfinite(M)
        if m.sum() < MIN_TEST_OBS:
            continue
        a = float((M[m] * ri[m]).mean())
        out["alphas"].append(a)
        out["alpha_cells"].append((widx, a))

    # EV: test-sample OLS of stock returns on the SDF portfolio return
    for i in range(R_te.shape[1]):
        ri = R_te[:, i]
        m = np.isfinite(ri) & np.isfinite(rp)
        if m.sum() < 8 or np.var(rp[m]) < 1e-12:
            continue
        b = np.polyfit(rp[m], ri[m], 1)
        e = ri[m] - (b[0] * rp[m] + b[1])
        out["ss_res"] += float((e ** 2).sum())
        out["ss_tot"] += float(((ri[m] - ri[m].mean()) ** 2).sum())
    return out


def aggregate(per_win, name):
    """Pooled metrics — same formulas as run_e1's eval_factor_model return."""
    rps = [w_["pooled_rp"] for w_ in per_win if w_["n_oos"] >= 6]
    sharps = [w_["sharpe"] for w_ in per_win if w_["sharpe"] == w_["sharpe"]]
    r2s = [(w_["r2"], w_["r2_n"]) for w_ in per_win if w_["r2"] == w_["r2"]]
    alphas = [a for w_ in per_win for a in w_["alphas"]]
    ss_res = sum(w_["ss_res"] for w_ in per_win)
    ss_tot = sum(w_["ss_tot"] for w_ in per_win)
    if not rps:
        return {"name": name, "error": "no windows"}
    rp_all = np.concatenate(rps)
    cells = [(wi + 1, a) for wi, w_ in enumerate(per_win)
             for (_, a) in w_["alpha_cells"]]
    return {
        "name": name,
        "n_windows": len(rps),
        "n_oos_months": len(rp_all),
        "sharpe_pooled": sharpe_ann(rp_all),
        "sharpe_mean_win": float(np.mean(sharps)) if sharps else math.nan,
        "r2": float(np.average([r for r, _ in r2s], weights=[n for _, n in r2s])) if r2s else math.nan,
        "ev": (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan,
        "rms_alpha_pct": float(np.sqrt(np.mean(np.square(alphas)))) * 100 if alphas else math.nan,
        "max_alpha_pct": float(np.max(np.abs(alphas))) * 100 if alphas else math.nan,
        "pooled_rp": rp_all,
        "alpha_cells": cells,
    }


# ── IPCA core ──────────────────────────────────────────────────────────────
def fit_ipca(Xp, Rp, t_idx, K, T_tr, F_init, max_iter=100, tol=1e-5):
    """Xp (M,L+1) w/ intercept, Rp (M,), t_idx (M,) month position in train.
    T_tr = number of train months; F_init (T_tr, K) starting factors.
    Returns (Gamma (L+1, K), F (T_tr, K))."""
    M = len(Rp)
    F = F_init.copy()

    Gamma = None
    for it in range(max_iter):
        # Gamma step: R = [f_t (x) x_it]' vec(Gamma)
        Z = np.einsum("mk,ml->mkl", F[t_idx], Xp).reshape(M, K * Xp.shape[1])
        g, *_ = np.linalg.lstsq(Z, Rp, rcond=None)
        Gamma_new = g.reshape(Xp.shape[1], K)
        # F step: per month f = (Lam'Lam)^-1 Lam'r
        Lam_all = Xp @ Gamma_new                       # (M, K)
        F_new = np.zeros_like(F)
        for t in range(T_tr):
            m = t_idx == t
            if m.sum() < K + 2:
                continue
            Lam = Lam_all[m]
            r = Rp[m]
            S = Lam.T @ Lam
            try:
                F_new[t] = np.linalg.solve(S + 1e-10 * np.eye(K), Lam.T @ r)
            except np.linalg.LinAlgError:
                F_new[t] = np.linalg.pinv(S) @ Lam.T @ r
        if Gamma is not None:
            num = np.sqrt(np.mean((Gamma_new - Gamma) ** 2))
            den = np.sqrt(np.mean(Gamma ** 2)) + 1e-12
            if num / den < tol and np.max(np.abs(F_new - F)) < 1e-6:
                Gamma, F = Gamma_new, F_new
                break
        Gamma, F = Gamma_new, F_new
    return Gamma, F


def ipca_window(X_tr, R_tr, X_te, R_te, K):
    """Fit IPCA on train (complete-x stock-months), update factors OOS.
    Returns (F_tr, F_te) monthly factor matrices for the metric code."""
    T_tr, N, L = X_tr.shape
    T_te = X_te.shape[0]
    Xc = np.concatenate([np.ones((T_tr, N, 1)), X_tr], axis=2)
    Xe = np.concatenate([np.ones((T_te, N, 1)), X_te], axis=2)

    m_tr = np.isfinite(R_tr) & np.isfinite(Xc).all(axis=2)
    rows_x, rows_r, rows_t, rows_si = [], [], [], []
    for t in range(T_tr):
        idx = np.where(m_tr[t])[0]
        rows_x.append(Xc[t, idx])
        rows_r.append(R_tr[t, idx])
        rows_t.append(np.full(len(idx), t))
        rows_si.append(idx)
    if not rows_r:
        return None
    Xp = np.vstack(rows_x)
    Rp = np.concatenate(rows_r)
    t_idx = np.concatenate(rows_t)
    si_idx = np.concatenate(rows_si)
    if len(Rp) < 50 * K:
        return None
    # F init: top-K PCA of the ragged train panel (rows vary by month),
    # missing stock-months -> 0 after demeaning (same convention as the
    # PCA(5) benchmark's "impute missing with 0")
    R_pan = np.full((T_tr, N), np.nan)
    R_pan[t_idx, si_idx] = Rp
    col_mean = np.nanmean(R_pan, axis=0)
    R_pan = np.where(np.isfinite(R_pan), R_pan, np.where(np.isfinite(col_mean), col_mean, 0.0))
    R_pan_c = R_pan - np.nanmean(R_pan, axis=0, keepdims=True)
    R_pan_c = np.where(np.isfinite(R_pan_c), R_pan_c, 0.0)
    if R_pan_c.std() > 0:
        try:
            _, _, Vt = np.linalg.svd(R_pan_c, full_matrices=False)
            F_init = R_pan_c @ Vt[:K].T
        except np.linalg.LinAlgError:
            F_init = np.zeros((T_tr, K))
    else:
        F_init = np.zeros((T_tr, K))
    try:
        Gamma, F_tr = fit_ipca(Xp, Rp, t_idx, K, T_tr, F_init)
    except np.linalg.LinAlgError:
        return None
    if Gamma is None or not np.all(np.isfinite(Gamma)):
        return None

    # OOS factor update per test month (Gamma frozen, point-in-time chars).
    # The paper's factor benchmarks use EVERY test month (no month drops), so
    # IPCA must too: the update is a K-dimensional cross-sectional regression
    # with N >> K observations — it fails only if the month is nearly empty,
    # in which case fall back to the zero vector (factor orthogonal to that
    # month's returns; disclosed in the skipped-month count).
    F_te = np.full((T_te, K), np.nan)
    n_fallback = 0
    for t in range(T_te):
        m = np.isfinite(R_te[t]) & np.isfinite(Xe[t]).all(axis=1)
        if m.sum() < max(K + 5, 20):
            n_fallback += 1
            continue
        Lam = Xe[t, m] @ Gamma
        r = R_te[t, m]
        S = Lam.T @ Lam
        try:
            sol = np.linalg.solve(S + 1e-8 * np.eye(K), Lam.T @ r)
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(S) @ Lam.T @ r
        F_te[t] = np.where(np.isfinite(sol), sol, 0.0)
    F_te = np.where(np.isfinite(F_te), F_te, 0.0)
    if (T_te - n_fallback) < 6:
        return None
    return F_tr, F_te, n_fallback


# ── PCA(5) factors per window (verbatim logic from run_e1 main) ────────────
def pca_window(R_tr, R_te, N_PCA=5):
    Rt = np.where(np.isfinite(R_tr), R_tr, 0.0)
    S = shrunk_cov(Rt)
    _, _, Vt = np.linalg.svd(S)
    V = Vt[:N_PCA].T
    F_tr = Rt @ V
    F_te = np.where(np.isfinite(R_te), R_te, 0.0) @ V
    return F_tr, F_te


# ── driver ─────────────────────────────────────────────────────────────────
def main():
    print(f"== IPCA (proper) — country={_C or 'IR'} ==")
    arr, dates, variables, tickers = load_npz()
    rf = load_rf()
    ff5 = load_factors_ff5()
    q = load_factors_q()

    common = [d for d in dates if (int(d[:4]), int(d[5:7])) in ff5 and
              (int(d[:4]), int(d[5:7])) in q]
    T = len(common)
    pos = {d: i for i, d in enumerate(dates)}
    idx = [pos[d] for d in common]
    print(f"  common period {common[0]}..{common[-1]} ({T} months)")

    R_raw = arr[idx, :, 0].astype(float)
    R = R_raw.copy()
    R[R < -50.0] = np.nan
    rf_v = np.array([rf.get(d, np.nan) for d in common])
    R_exc = R - rf_v[:, None]

    X = arr[idx, :, 1:].astype(float)
    X[X < -50.0] = np.nan
    cov = np.mean(np.isfinite(X), axis=(0, 1))
    keep = np.where(cov >= 0.30)[0]
    dropped = [variables[1 + k] for k in range(X.shape[2]) if k not in keep]
    if dropped:
        print(f"  char coverage filter drops: {dropped}")
    X = X[:, :, keep]
    macro_dummy = np.zeros((T, 1))
    X, _ = lag_align(X, macro_dummy)

    F_ff5 = np.array([[ff5[(int(d[:4]), int(d[5:7]))][k] for k in
                       ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]] for d in common])
    F_q = np.array([[q[(int(d[:4]), int(d[5:7]))][k] for k in
                     ["Mkt_RF", "ME", "IA", "ROE"]] for d in common])
    F_mkt = F_ff5[:, :1]

    windows = list(rolling_windows(list(range(T)), train=60, test=12))
    print(f"  {len(windows)} windows (train 60 / test 12)")

    # ---- STEP 1: validate the evaluator against committed benchmark rows ----
    print("\n  [validation] reproducing committed e1_benchmarks.csv rows ...")
    committed = {}
    p_b = RES / "e1_benchmarks.csv"
    if p_b.exists():
        with open(p_b, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                committed[r["name"]] = r
    F_map = {"Market": F_mkt, "FF5": F_ff5, "q-factor": F_q}
    max_diff = 0.0
    for name, F_all in F_map.items():
        per = [eval_factor_window(F_all[tr], F_all[te], R_exc[tr], R_exc[te], wi)
               for wi, (tr, te) in enumerate(windows)]
        agg = aggregate(per, name)
        c = committed.get(name)
        if not c:
            print(f"    {name}: no committed row — skipped")
            continue
        diffs = {k: abs(float(agg[k]) - float(c[k])) for k in
                 ["sharpe_pooled", "sharpe_mean_win", "r2", "ev",
                  "rms_alpha_pct", "max_alpha_pct"] if c.get(k)}
        md = max(diffs.values())
        max_diff = max(max_diff, md)
        print(f"    {name}: max |diff| = {md:.2e}  {'OK' if md < 1e-6 else 'MISMATCH ⚠'}")
    # PCA(5) validation (inline construction in run_e1)
    per = []
    for wi, (tr, te) in enumerate(windows):
        F_tr, F_te = pca_window(R_exc[tr], R_exc[te], 5)
        per.append(eval_factor_window(F_tr, F_te, R_exc[tr], R_exc[te], wi))
    agg = aggregate(per, "PCA(5)")
    c = committed.get("PCA(5)")
    if c:
        md = max(abs(float(agg[k]) - float(c[k])) for k in
                 ["sharpe_pooled", "sharpe_mean_win", "r2", "ev",
                  "rms_alpha_pct", "max_alpha_pct"] if c.get(k))
        max_diff = max(max_diff, md)
        print(f"    PCA(5): max |diff| = {md:.2e}  {'OK' if md < 1e-6 else 'MISMATCH ⚠'}")
    if max_diff >= 1e-6:
        print("  VALIDATION FAILED — aborting (committed rows not reproduced).")
        sys.exit(2)
    print("  [validation] PASSED — evaluator is byte-identical to run_e1.\n")

    # ---- STEP 2: IPCA(K) ----
    results, all_cells, pooled_series = [], [], []
    for K in KS:
        per = []
        for wi, (tr, te) in enumerate(windows):
            got = ipca_window(X[tr], R_exc[tr], X[te], R_exc[te], K)
            if got is None:
                print(f"    IPCA({K}) window {wi}: skipped")
                continue
            F_tr, F_te, n_fb = got
            if n_fb:
                print(f"    IPCA({K}) window {wi}: {n_fb} month(s) used zero-factor fallback")
            per.append(eval_factor_window(F_tr, F_te, R_exc[tr], R_exc[te], wi))
        agg = aggregate(per, f"IPCA({K})")
        if "error" in agg:
            print(f"  IPCA({K}): FAILED — {agg['error']}")
            continue
        print(f"  IPCA({K}): n_win={agg['n_windows']} sharpe={agg['sharpe_pooled']:+.4f} "
              f"rms={agg['rms_alpha_pct']:.2f}% ev={agg['ev']:+.4f} max_a={agg['max_alpha_pct']:.1f}%")
        results.append(agg)
        all_cells.extend([(agg["name"], w, f"{a:.8f}") for w, a in agg["alpha_cells"]])
        pooled_series.extend([(agg["name"], f"{r:.8f}") for r in agg["pooled_rp"]])

    if not results:
        print("ERROR: all IPCA specs failed")
        sys.exit(1)

    fields = ["name", "n_windows", "n_oos_months", "sharpe_pooled",
              "sharpe_mean_win", "r2", "ev", "rms_alpha_pct", "max_alpha_pct"]
    with open(RES / "e1_ipca_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{r[k]:.10f}" if isinstance(r[k], float) else r[k])
                        for k in fields})
    with open(RES / "e1_ipca_alpha_cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "window", "alpha"])
        w.writerows(all_cells)
    with open(RES / "e1_ipca_pooled_series.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "oos_return"])
        w.writerows(pooled_series)
    print(f"\nSaved -> {RES}/e1_ipca_results.csv (+ alpha cells, pooled series)")


if __name__ == "__main__":
    main()
