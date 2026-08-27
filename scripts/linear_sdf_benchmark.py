#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear_sdf_benchmark.py — common LINEAR SDF benchmark (CPZ's linear specification)
====================================================================================
The conditional linear-in-characteristics SDF that CPZ (2024) use as their
benchmark ("linear SDF"): a COMMON kernel with weights linear in firm
characteristics,

    omega_t(i) = theta' x_{i,t}            (theta in R^F, estimated per window)
    M_t        = 1 - (1/N_t) * sum_i omega_t(i) * R^e_{t,i} * mean(N_t)
    alpha_i    = (1/T_i) * sum_t M_t * R^e_{t,i}

Estimation: weighted least squares in closed form. With
    mu_i = E[R^e_{i,t}],  c_i = E[ R^e_{i,t} * (1/N_t) sum_j x_{j,t} R^e_{j,t} ]
we have alpha_i(theta) = mu_i - theta' c_i, and the official weighted loss
    L(theta) = mean_i (count_i / max_count) * alpha_i^2
is minimized by theta = (C' W C + ridge)^-1 C' W mu   (C: N x F rows c_i).

Same rolling protocol (train 48 / valid 12 / test 12), same metrics as the
deep SDF (Sharpe of the SDF portfolio r_p,t = sum_i omega R / sum_i |omega|,
EV, RMS/max alpha). Runs write results/linear_sdf_results.csv plus pooled
series per specification.

This benchmark closes the loop on the referee point: the deep (nonlinear,
common) SDF should beat its own linear-in-characteristics restriction.
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_core import load_rf, load_factors_ff5, load_factors_q, \
    sharpe_ann, rolling_windows, lag_align, load_npz

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data"
RES = ROOT / {"TR": "results_tr", "PK": "results_pk"}.get(os.environ.get("DLAP_COUNTRY", "").upper(), "results")

SY_INDICES = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
RIDGE = 1e-6


def load_data():
    arr, dates, variables, tickers = load_npz()
    rf = load_rf()
    ff5 = load_factors_ff5()
    q = load_factors_q()
    common = [d for d in dates if (int(d[:4]), int(d[5:7])) in ff5 and
              (int(d[:4]), int(d[5:7])) in q]
    pos = {d: i for i, d in enumerate(dates)}
    idx = [pos[d] for d in common]
    R = arr[idx, :, 0].astype(float)
    R[R < -50] = np.nan
    rf_v = np.array([rf.get(d, np.nan) for d in common])
    R_exc = R - rf_v[:, None]
    X = arr[idx, :, 1:].astype(float)
    X[X < -50] = np.nan
    X, _ = lag_align(X, np.zeros((len(common), 1)))
    return R_exc, X, common


def estimate_theta(R, X, mask, ridge=RIDGE, lam=0.0):
    """Weighted LS for the common linear SDF on TRAIN months.
    R: (T,N) X: (T,N,F) mask: (T,N) -> theta (F,)
    c_i = mean_t [ R_i,t * (1/N_t) sum_j x_j,t R_j,t ]  (over valid obs)

    lam: KNS-style SDF-variance penalty  lam * mean_t (M_t - 1)^2, i.e.
    theta' G theta with G = mean_t (managed_t managed_t') — keeps the SDF
    scale near unity on fat-tailed markets (TSE) where the unpenalized
    linear SDF lets M_t = 1 - sum(omega R)/N_t explode out of sample."""
    T, N, F = X.shape
    valid_t = mask.sum(axis=1) > 0
    wr = np.where(mask[:, :, None], X * R[:, :, None], 0.0)  # (T,N,F)
    n_t = mask.sum(axis=1).clip(min=1)                   # (T,)
    managed = wr.sum(axis=1) / n_t[:, None]              # (T,F) = (1/N_t) sum_j x_j R_j
    cnt = mask.sum(axis=0).clip(min=1)                   # (N,)
    # c_i = mean_t [ R_i,t * managed_t ]  (only valid t)
    mr = np.where(mask[:, :, None], R[:, :, None] * managed[:, None, :], 0.0)
    C = mr.sum(axis=0) / cnt[:, None]                    # (N,F)
    mu = np.where(mask, R, 0.0).sum(axis=0) / cnt        # (N,)
    w = cnt / cnt.max()                                  # official weighted loss
    W = np.diag(w)
    A = C.T @ W @ C + ridge * np.trace(C.T @ W @ C) * np.eye(F) / F
    if lam > 0:
        G = (managed[valid_t].T @ managed[valid_t]) / valid_t.sum()
        A = A + lam * G
    b = C.T @ W @ mu
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # singular A: typically an all-NaN char column in a country layout
        # (e.g. dy on PK) collinear/degenerate -> bump ridge and retry once
        A2 = A + max(RIDGE, 1e-4) * np.trace(C.T @ W @ C) * np.eye(F) / F
        return np.linalg.solve(A2, b)


def pricing_loss_for(R, X, mask, theta):
    """Weighted squared pricing errors (official loss form) for a given theta."""
    wr = np.where(mask[:, :, None], X * R[:, :, None], 0.0)
    n_t = mask.sum(axis=1).clip(min=1)
    managed = wr.sum(axis=1) / n_t[:, None]
    cnt = mask.sum(axis=0).clip(min=1)
    mr = np.where(mask[:, :, None], R[:, :, None] * managed[:, None, :], 0.0)
    C = mr.sum(axis=0) / cnt[:, None]
    mu = np.where(mask, R, 0.0).sum(axis=0) / cnt
    alpha = mu - C @ theta
    valid = cnt >= 6
    w = cnt[valid] / cnt[valid].max()
    return float((w * alpha[valid] ** 2).mean())


RIDGE_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
# KNS-style SDF-variance penalty grid (lam * mean_t (M_t - 1)^2)
LAM_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1,
            1.0, 3.0, 10.0, 30.0, 100.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charset", choices=["sy", "all"], default="sy")
    args = ap.parse_args()

    _, _, _variables, _ = load_npz()
    _VARS = [str(v) for v in _variables]
    _NF = len(_VARS) - 1  # minus return

    sy_names = ["mom", "ag", "ac", "noa", "nsi", "gp", "cei", "ita", "ig",
                "dist", "oscore"]
    if args.charset == "sy":
        variables = [str(v) for v in _VARS]
        avail = [v for v in variables if v in sy_names]
        feat_idx = [variables.index(n) - 1 for n in avail]  # npz col0 = return
    else:
        feat_idx = list(range(_NF))
    out_name = "lin11" if args.charset == "sy" else "lin20"
    print(f"== LINEAR SDF [{out_name}]: {len(feat_idx)} chars, common kernel ==")

    R_exc, X, common = load_data()
    X = np.clip(X[:, :, feat_idx], -10.0, 10.0)
    T = len(common)
    windows = list(rolling_windows(list(range(T)), train=60, test=12))

    pooled_rp, all_alphas, per_win_sharpe = [], [], []
    pooled_rp_aligned = []
    n_windows = 0
    for wi, (w_tr, w_te) in enumerate(windows):
        w_va = w_tr[-12:]
        w_tr = w_tr[:-12]
        R_tr, X_tr = R_exc[w_tr], X[w_tr]
        R_va, X_va = R_exc[w_va], X[w_va]
        R_te, X_te = R_exc[w_te], X[w_te]

        keep = np.isfinite(R_tr).sum(axis=0) >= 12
        R_tr, X_tr = R_tr[:, keep], X_tr[:, keep]
        R_va, X_va = R_va[:, keep], X_va[:, keep]
        R_te, X_te = R_te[:, keep], X_te[:, keep]
        if R_tr.shape[1] < 50:
            print(f"  window {wi}: skipped ({R_tr.shape[1]} stocks)")
            continue

        # PK/early-window quirk: requiring ALL chars finite zeroes out entire
        # months (mom/nsi lookbacks missing at panel start). Soft handling:
        # mask on returns only; missing chars are ZERO-FILLED (deep models'
        # treatment), so no NaN propagates into the LS system.
        mask_tr = np.isfinite(R_tr)
        X_tr = np.nan_to_num(X_tr, nan=0.0)
        # Pure weighted LS (CPZ linear specification). A tiny ridge guards
        # against exact singularity; no penalty is needed now that the SDF
        # scale is the published form M_t = 1 - mean_i(omega R) (the old
        # official-code '/N_t * mean(N_t)' rescale blew up the SDF variance).
        theta = estimate_theta(R_tr, X_tr, mask_tr, ridge=RIDGE, lam=0.0)

        mask_te = np.isfinite(R_te)
        X_te = np.nan_to_num(X_te, nan=0.0)
        omega_te = np.where(mask_te, np.nan_to_num(X_te @ theta), 0.0)  # (12,N)
        num = (omega_te * np.nan_to_num(R_te)).sum(axis=1)
        den = np.abs(omega_te).sum(axis=1)
        rp = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), np.nan)
        rp = rp[np.isfinite(rp)]

        # M_t on test months (common kernel, published CPZ form: 1 - mean_i(omega R))
        wr = np.where(mask_te, np.nan_to_num(X_te @ theta) * np.nan_to_num(R_te), 0.0)
        n_t = mask_te.sum(axis=1).clip(min=1)
        M_te = 1.0 - wr.sum(axis=1) / n_t
        mr = np.where(mask_te, M_te[:, None] * np.nan_to_num(R_te), 0.0)
        cnt_te = mask_te.sum(axis=0).clip(min=1)
        alpha_te = mr.sum(axis=0) / cnt_te
        alpha_te = alpha_te[cnt_te >= 6]

        if len(rp) >= 6:
            # keep a test-month-aligned copy (nan where filtered) for EV
            rp_aligned = np.full(len(w_te), np.nan)
            rp_aligned[np.isfinite(rp) if len(rp) == len(w_te) else slice(0, 0)] = rp if len(rp) == len(w_te) else np.nan
            pooled_rp.append(rp)
            pooled_rp_aligned.append(rp_aligned)
            per_win_sharpe.append(sharpe_ann(rp))
            all_alphas.extend(list(alpha_te))
            n_windows += 1
        print(f"  window {wi} [{common[w_te[0]]}..{common[w_te[-1]]}]: "
              f"sharpe={sharpe_ann(rp):.3f}")

    if not pooled_rp:
        print("FATAL: no windows completed")
        sys.exit(1)
    rp_all = np.concatenate(pooled_rp)
    sharpe = sharpe_ann(rp_all)
    rms_alpha = float(np.sqrt(np.mean(np.square(all_alphas)))) * 100
    max_alpha = float(np.max(np.abs(all_alphas))) * 100

    # EV (same construction as deep SDF)
    ss_res = ss_tot = 0.0
    for wi in range(n_windows):
        w_tr, w_te = windows[wi]
        keep = np.isfinite(R_exc[w_tr[:-12]]).sum(axis=0) >= 12
        R_te_w = R_exc[w_te][:, keep]
        rp_w = pooled_rp_aligned[wi]
        for i in range(R_te_w.shape[1]):
            ri = R_te_w[:, i]
            m = np.isfinite(ri) & np.isfinite(rp_w)
            if m.sum() < 8 or np.var(rp_w[m]) < 1e-12:
                continue
            b = np.polyfit(rp_w[m], ri[m], 1)
            e = ri[m] - (b[0] * rp_w[m] + b[1])
            ss_res += float((e ** 2).sum())
            ss_tot += float(((ri[m] - ri[m].mean()) ** 2).sum())
    ev = (1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    print(f"\nLINEAR SDF [{out_name}] pooled OOS: sharpe={sharpe:.3f} "
          f"EV={ev:.4f} rms_alpha={rms_alpha:.3f}% max_alpha={max_alpha:.3f}%")

    out_csv = RES / "linear_sdf_results.csv"
    header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["name", "charset", "n_features", "n_windows",
                        "n_oos_months", "sharpe_pooled", "ev",
                        "rms_alpha_pct", "max_alpha_pct"])
        w.writerow([out_name, args.charset, len(feat_idx), n_windows,
                    len(rp_all), f"{sharpe:.4f}", f"{ev:.4f}",
                    f"{rms_alpha:.4f}", f"{max_alpha:.4f}"])
    with open(RES / f"linear_sdf_{out_name}_pooled_series.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["oos_return"])
        for v in rp_all:
            w.writerow([f"{v:.6f}"])
    print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
