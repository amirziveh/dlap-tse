#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_e2_lag.py — Robustness check: lag-aligned deep SDF (E2LAG)
================================================================
Same as E2 (11 SY chars, LSTM macro states, rolling 60/12/12) but with
PROPERLY LAGGED alignment: the return of month t is priced by
characteristics and macro state observed at month t-1 (the canonical
CPZ/GKX x_t -> r_{t+1} convention), instead of same-month x_t -> r_t.

Only the alignment changes; everything else (seed, windows, hyperparams,
portfolio construction) is identical to train_e2.py --charset sy.

Outputs: results/e2lag_results.csv, results/e2lag_pooled_series.csv
"""
import argparse
import csv
import os
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_core import sharpe_ann, rolling_windows
from train_e2 import (load_data, train_window, SY_INDICES, sdf_values,
                      STATE_DIM, LR, MAX_EPOCHS, PATIENCE, MIN_OBS_ALPHA)

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
RES = ROOT / "results"

torch.manual_seed(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charset", choices=["sy", "all"], default="sy")
    ap.add_argument("--states", choices=["lstm", "const"], default="lstm")
    ap.add_argument("--critic", action="store_true")
    ap.add_argument("--liq-filter", action="store_true")
    args = ap.parse_args()

    out_name = "e2lag"
    feat_idx = SY_INDICES if args.charset == "sy" else list(range(20))
    n_features = len(feat_idx)
    print(f"== {out_name.upper()}: deep SDF, {n_features} chars, "
          f"LAGGED alignment (x_{{t-1}} -> r_t) ==")

    R_exc, X, macro, common = load_data()
    X_full = X
    T = len(common)

    # ── THE ONLY CHANGE: lag the characteristics & macro by one month ──
    X = np.concatenate([np.full((1, X.shape[1], X.shape[2]), np.nan), X[:-1]], axis=0)
    macro = np.concatenate([np.full((1, macro.shape[1]), np.nan), macro[:-1]], axis=0)
    macro[0] = macro[1]  # keep LSTM input finite (row 0 is inside train only)
    # ────────────────────────────────────────────────────────────────────

    windows = list(rolling_windows(list(range(T)), train=60, test=12))
    print(f"  {len(windows)} windows, period {common[0]}..{common[-1]}")

    X = np.clip(X[:, :, feat_idx], -10.0, 10.0)
    turnover_full = X_full[:, :, 2]

    pooled_rp, all_alphas, per_win_sharpe = [], [], []
    n_windows = 0
    for wi, (w_tr, w_te) in enumerate(windows):
        w_va = w_tr[-12:]
        w_tr = w_tr[:-12]
        R_tr, X_tr, mac_tr = R_exc[w_tr], X[w_tr], macro[w_tr]
        R_va, X_va, mac_va = R_exc[w_va], X[w_va], macro[w_va]
        R_te, X_te, mac_te = R_exc[w_te], X[w_te], macro[w_te]

        keep = np.isfinite(R_tr).sum(axis=0) >= 12
        if args.liq_filter:
            tr_turn = np.nanmean(turnover_full[w_tr][:, keep], axis=0)
            thr = np.nanpercentile(tr_turn, 5)
            keep = keep.copy()
            keep[keep] = keep[keep] & (tr_turn > thr)
        R_tr, X_tr = R_tr[:, keep], X_tr[:, keep]
        R_va, X_va = R_va[:, keep], X_va[:, keep]
        R_te, X_te = R_te[:, keep], X_te[:, keep]
        if R_tr.shape[1] < 50:
            print(f"  window {wi}: skipped ({R_tr.shape[1]} stocks)")
            continue

        znet, mnet, cnet, val_loss, epochs_used = train_window(
            R_tr, X_tr, mac_tr, R_va, X_va, mac_va, n_features,
            states=args.states, critic=args.critic)
        print(f"  window {wi} [{common[w_te[0]]}..{common[w_te[-1]]}]: "
              f"val_loss={val_loss:.2e} epochs={epochs_used}")

        znet.eval(); mnet.eval()
        mu = mac_tr.mean(axis=0); sd = mac_tr.std(axis=0) + 1e-12
        mac_all = torch.from_numpy(
            (np.concatenate([mac_tr, mac_va, mac_te]) - mu) / sd).float()
        X_all = np.concatenate([X_tr, X_va, X_te], axis=0)
        R_all = np.concatenate([R_tr, R_va, R_te], axis=0)
        X_all_t = torch.from_numpy(np.nan_to_num(X_all, nan=0.0)).float()
        R_all_t = torch.from_numpy(np.nan_to_num(R_all, nan=0.0)).float()
        with torch.no_grad():
            z_all = znet(mac_all)
            M_all = sdf_values(mnet(z_all), X_all_t).numpy()
        M_te = M_all[-len(w_te):]
        mask_te = np.isfinite(R_te) & np.isfinite(X_te).all(axis=2)
        num = np.where(mask_te, M_te * R_te, 0.0).sum(axis=1)
        den = np.where(mask_te, M_te, 0.0).sum(axis=1)
        rp = np.where(den != 0, num / np.where(den != 0, den, 1.0), np.nan)
        rp = rp[np.isfinite(rp)]
        alphas = [float((M_te[mask_te[:, i], i] * R_te[mask_te[:, i], i]).mean())
                  for i in range(R_te.shape[1])
                  if mask_te[:, i].sum() >= MIN_OBS_ALPHA]
        if len(rp) >= 6:
            pooled_rp.append(rp)
            per_win_sharpe.append(sharpe_ann(rp))
            all_alphas.extend(alphas)
            n_windows += 1

    rp_all = np.concatenate(pooled_rp)
    sharpe = sharpe_ann(rp_all)
    rms_alpha = float(np.sqrt(np.mean(np.square(all_alphas)))) * 100
    max_alpha = float(np.max(np.abs(all_alphas))) * 100

    print(f"\n{out_name.upper()} pooled OOS (LAGGED): n_windows={n_windows} "
          f"sharpe={sharpe:.3f} rms_alpha={rms_alpha:.3f}% max_alpha={max_alpha:.3f}%")

    with open(RES / f"{out_name}_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "sharpe", "rms_alpha_pct", "max_alpha_pct",
                    "n_windows", "note"])
        w.writerow([out_name, f"{sharpe:.4f}", f"{rms_alpha:.4f}",
                    f"{max_alpha:.4f}", n_windows, "11ch LSTM, LAGGED align"])
    with open(RES / f"{out_name}_pooled_series.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "oos_return"])
        for v in rp_all:
            w.writerow([out_name, f"{v:.6f}"])


if __name__ == "__main__":
    main()
