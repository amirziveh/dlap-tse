#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ae_benchmark.py — Conditional Autoencoder (Gu, Kelly & Xiu 2020, JoE) benchmark
===============================================================================
Mirrors scripts/ipca_proper.py exactly:
  - Same months/windows: dates ∩ FF5 ∩ q, rolling train 60 / test 12.
  - Same panel: Char_all.npz, sentinel -99.99 -> NaN, >=30% char coverage,
    lag-aligned characteristics, excess returns R - rf.
  - Same metric code: eval_factor_window / aggregate are IMPORTED from
    ipca_proper (which validates them against committed e1_benchmarks.csv
    rows to <1e-6 before any model runs).

GKX conditional autoencoder (per window, train only):
  R_it = beta_i(z_it)' f_t + e_it,   beta_i = encoder(z_it)  (z includes 1.0)
  Autoencoder loss: sum_it (r_it - beta' f_t)^2 over observed train stock-months
  + L2 weight decay. Factors constrained to unit norm, zero cross-sectional
  mean (identified up to rotation; GKX Sec. II.B). Init: factors = top-K PCA
  of the train return panel (missing -> CS mean, same as IPCA here);
  encoder = MLP [L+1 -> 32 -> 16 -> K] with tanh, linear output (GKX layer-1).
  ALS-style alternation, 200 epochs Adam(1e-3), batch = all rows.

OOS (no leakage):
  Encoder frozen at train values. Test-month factor: cross-sectional OLS
  f_t = (B'B)^-1 B' r_t with B = encoder(test-month chars), point-in-time.
  Then F_tr/F_te feed the SAME metric code as FF5/q/PCA/IPCA.

Outputs per country (results{,_tr,_pk}/):
  e1_ae_results.csv      same schema as e1_ipca_results.csv (rows AE(1,3,5))
  e1_ae_alpha_cells.csv  model,window,alpha  (rms_window_bootstrap-compatible)
  e1_ae_pooled_series.csv
"""
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_core import load_npz, load_rf, load_factors_ff5, load_factors_q, \
    lag_align, rolling_windows
from ipca_proper import eval_factor_window, aggregate, pca_window

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
_C = os.environ.get("DLAP_COUNTRY", "").upper()
if _C == "TR":
    DATA, RES = ROOT / "data_tr", ROOT / "results_tr"
elif _C == "PK":
    DATA, RES = ROOT / "data_pk", ROOT / "results_pk"
else:
    DATA, RES = ROOT / "data", ROOT / "results"

KS = [1, 3, 5]
HIDDEN = [32, 16]
SEED = 42
EPOCHS = 200
LR = 1e-3
WD = 1e-4
TORCH_SEED = 0


class Encoder(torch.nn.Module):
    """GKX layer-1: characteristics -> factor loadings (nonlinear)."""
    def __init__(self, n_in, ks):
        super().__init__()
        layers, d = [], n_in
        for h in HIDDEN:
            layers += [torch.nn.Linear(d, h), torch.nn.Tanh()]
            d = h
        layers.append(torch.nn.Linear(d, ks))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def _fit_ae(Xp, Rp, t_idx, T_tr, K, F_init):
    """Xp (M, L+1) incl. intercept column FIRST, Rp (M,), t_idx month index.
    Returns (F (T_tr, K) unit-norm zero-CS-mean, trained encoder state)."""
    M = len(Rp)
    enc = Encoder(Xp.shape[1], K)
    opt = torch.optim.Adam(enc.parameters(), lr=LR, weight_decay=WD)
    Zt = torch.from_numpy(Xp.astype(np.float32))
    Rt = torch.from_numpy(Rp.astype(np.float32))
    ti = torch.from_numpy(t_idx.astype(np.int64))
    F = torch.from_numpy(F_init.astype(np.float32))
    F = F / F.norm(dim=0).clamp(min=1e-8)

    for ep in range(EPOCHS):
        enc.train()
        B = enc(Zt)                                   # (M, K)
        # unit-norm + zero CS mean constraints on factors (GKX identification)
        F = F - F.mean(dim=0, keepdim=True)
        F = F / F.norm(dim=0).clamp(min=1e-8)
        pred = (B * F[ti]).sum(dim=1)
        loss = ((pred - Rt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        # factor step: closed-form OLS given current loadings (per month)
        with torch.no_grad():
            enc.eval()
            B = enc(Zt).numpy()
            F_np = np.zeros((T_tr, K), dtype=np.float64)
            for t in np.unique(t_idx):
                m = t_idx == t
                if m.sum() >= K + 2:
                    Bm, rm = B[m], Rp[m]
                    S = Bm.T @ Bm
                    try:
                        F_np[t] = np.linalg.solve(S + 1e-8 * np.eye(K), Bm.T @ rm)
                    except np.linalg.LinAlgError:
                        F_np[t] = np.linalg.pinv(S) @ Bm.T @ rm
            F = torch.from_numpy(F_np.astype(np.float32))
            F = F - F.mean(dim=0, keepdim=True)
            F = F / F.norm(dim=0).clamp(min=1e-8)
    # Final factor refit at the NATURAL closed-form scale (no normalization):
    # the unit-norm/zero-mean constraints identify the encoder during training,
    # but the paper's metric code (tangency weights from train factor means,
    # M = 1 - w'F) is degenerate on zero-mean factors. IPCA delivers the same
    # raw (Lam'Lam)^-1 Lam'r factors, so AE must too for comparability.
    with torch.no_grad():
        enc.eval()
        B = enc(Zt).numpy()
        F_final = np.zeros((T_tr, K), dtype=np.float64)
        for t in np.unique(t_idx):
            m = t_idx == t
            if m.sum() >= K + 2:
                Bm, rm = B[m], Rp[m]
                S = Bm.T @ Bm
                try:
                    F_final[t] = np.linalg.solve(S + 1e-8 * np.eye(K), Bm.T @ rm)
                except np.linalg.LinAlgError:
                    F_final[t] = np.linalg.pinv(S) @ Bm.T @ rm
    return F_final, {k: v.clone() for k, v in enc.state_dict().items()}


def ae_window(X_tr, R_tr, X_te, R_te, K):
    """Fit AE on train (complete-x stock-months), update factors OOS.
    Returns (F_tr, F_te, n_fallback) or None."""
    T_tr, N, L = X_tr.shape
    T_te = X_te.shape[0]
    Xc = np.concatenate([np.ones((T_tr, N, 1)), X_tr], axis=2)
    Xe = np.concatenate([np.ones((T_te, N, 1)), X_te], axis=2)

    m_tr = np.isfinite(R_tr) & np.isfinite(Xc).all(axis=2)
    rows_x, rows_r, rows_t = [], [], []
    for t in range(T_tr):
        idx = np.where(m_tr[t])[0]
        rows_x.append(Xc[t, idx])
        rows_r.append(R_tr[t, idx])
        rows_t.append(np.full(len(idx), t))
    if not rows_r:
        return None
    Xp = np.vstack(rows_x)
    Rp = np.concatenate(rows_r)
    t_idx = np.concatenate(rows_t)
    if len(Rp) < 50 * K:
        return None

    # F init: top-K PCA of the ragged train panel (same convention as IPCA)
    R_pan = np.full((T_tr, N), np.nan)
    for t in range(T_tr):
        R_pan[t, m_tr[t]] = R_tr[t, m_tr[t]]
    col_mean = np.nanmean(R_pan, axis=0)
    R_pan = np.where(np.isfinite(R_pan), R_pan,
                     np.where(np.isfinite(col_mean), col_mean, 0.0))
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

    torch.manual_seed(TORCH_SEED)
    try:
        F_tr, enc_state = _fit_ae(Xp, Rp, t_idx, T_tr, K, F_init)
    except Exception as e:
        print(f"    AE({K}) train failed: {type(e).__name__}: {e}")
        return None
    if not np.all(np.isfinite(F_tr)) or np.allclose(F_tr, 0.0):
        return None

    # OOS factor update per test month (encoder frozen, point-in-time chars).
    # Same fallback convention as IPCA: months with too few complete rows use
    # the zero factor (disclosed via the skipped-month count).
    F_te = np.full((T_te, K), np.nan)
    n_fallback = 0
    enc = Encoder(Xe.shape[2], K)
    enc.load_state_dict(enc_state)
    enc.eval()
    with torch.no_grad():
        for t in range(T_te):
            m = np.isfinite(R_te[t]) & np.isfinite(Xe[t]).all(axis=1)
            if m.sum() < max(K + 5, 20):
                n_fallback += 1
                continue
            B = enc(torch.from_numpy(Xe[t, m].astype(np.float32))).numpy()
            r = R_te[t, m]
            S = B.T @ B
            try:
                sol = np.linalg.solve(S + 1e-8 * np.eye(K), B.T @ r)
            except np.linalg.LinAlgError:
                sol = np.linalg.pinv(S) @ B.T @ r
            F_te[t] = np.where(np.isfinite(sol), sol, 0.0)
    F_te = np.where(np.isfinite(F_te), F_te, 0.0)
    if (T_te - n_fallback) < 6:
        return None
    return F_tr, F_te, n_fallback


# module-level holder removed: encoder state now returned from _fit_ae


def main():
    print(f"== Conditional Autoencoder (GKX) — country={_C or 'IR'} ==")
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
    print("  [validation] PASSED — evaluator identical to run_e1/ipca_proper.\n")

    # ---- STEP 2: AE(K) ----
    results, all_cells, pooled_series = [], [], []
    for K in KS:
        per = []
        for wi, (tr, te) in enumerate(windows):
            got = ae_window(X[tr], R_exc[tr], X[te], R_exc[te], K)
            if got is None:
                print(f"    AE({K}) window {wi}: skipped")
                continue
            F_tr, F_te, n_fb = got
            if n_fb:
                print(f"    AE({K}) window {wi}: {n_fb} month(s) zero-factor fallback")
            per.append(eval_factor_window(F_tr, F_te, R_exc[tr], R_exc[te], wi))
        agg = aggregate(per, f"AE({K})")
        if "error" in agg:
            print(f"  AE({K}): FAILED — {agg['error']}")
            continue
        print(f"  AE({K}): n_win={agg['n_windows']} sharpe={agg['sharpe_pooled']:+.4f} "
              f"rms={agg['rms_alpha_pct']:.2f}% ev={agg['ev']:+.4f} max_a={agg['max_alpha_pct']:.1f}%")
        results.append(agg)
        all_cells.extend([(agg["name"], w, f"{a:.8f}") for w, a in agg["alpha_cells"]])
        pooled_series.extend([(agg["name"], f"{r:.8f}") for r in agg["pooled_rp"]])

    if not results:
        print("ERROR: all AE specs failed")
        sys.exit(1)

    fields = ["name", "n_windows", "n_oos_months", "sharpe_pooled",
              "sharpe_mean_win", "r2", "ev", "rms_alpha_pct", "max_alpha_pct"]
    with open(RES / "e1_ae_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{r[k]:.10f}" if isinstance(r[k], float) else r[k])
                        for k in fields})
    with open(RES / "e1_ae_alpha_cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "window", "alpha"])
        w.writerows(all_cells)
    with open(RES / "e1_ae_pooled_series.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "oos_return"])
        w.writerows(pooled_series)
    print(f"\nSaved -> {RES}/e1_ae_results.csv (+ alpha cells, pooled series)")


if __name__ == "__main__":
    main()
