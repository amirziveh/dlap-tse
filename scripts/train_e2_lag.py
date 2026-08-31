#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_e2_lag.py — SUPERSEDED (kept for the audit trail only)
============================================================
NOTE (2026-08-03, true-CPZ re-implementation): this script is no longer part
of the pipeline. The lagged alignment (x_{t-1} -> r_t) is now applied INSIDE
train_e2.load_data() via eval_core.lag_align (the canonical CPZ/GKX
convention), so E2 in train_e2.py IS the lagged specification. Running this
script would double-lag the data (x_{t-2} -> r_t). The e2lag CSV currently in
results/ predates the alignment fix and is identical to E2; it feeds nothing
in the manuscript. Do not re-run.
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
from train_e2 import (load_data, train_window, make_tensors, SY_INDICES,
                      STATE_DIM, LR, MAX_EPOCHS, PATIENCE, MIN_OBS_ALPHA)
from sdf_models import common_sdf, sdf_portfolio_return, pricing_errors_common

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
# country-aware results dir (same pattern as e7/seed_sensitivity)
_C = os.environ.get("DLAP_COUNTRY", "").upper()
RES = ROOT / {"TR": "results_tr", "PK": "results_pk"}.get(_C, "results")
torch.manual_seed(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charset", choices=["sy", "all"], default="sy")
    ap.add_argument("--states", choices=["lstm", "const"], default="lstm")
    ap.add_argument("--critic", action="store_true")
    ap.add_argument("--liq-filter", action="store_true")
    args = ap.parse_args()

    out_name = "e2lag"
    # country-aware charset: derive from load_data()'s npz variables so PK's
    # ig-dropped layout works (IR keeps 11 sy chars; PK has 10)
    R_exc, X, macro, common = load_data()
    X_full = X
    T = len(common)

    # ── THE ONLY CHANGE: lag the characteristics & macro by one month ──
    X = np.concatenate([np.full((1, X.shape[1], X.shape[2]), np.nan), X[:-1]], axis=0)
    macro = np.concatenate([np.full((1, macro.shape[1]), np.nan), macro[:-1]], axis=0)
    macro[0] = macro[1]  # keep LSTM input finite (row 0 is inside train only)
    # ────────────────────────────────────────────────────────────────────

    # country-aware sy charset: match SY_NAMES against the npz variable list
    # (PK drops ig -> 10 sy chars; IR keeps 11). load_npz re-read is cheap.
    if args.charset == "sy":
        from eval_core import load_npz as _lnpz
        variables = [str(v) for v in _lnpz()[2]]
        sy_names = ["mom", "ag", "ac", "noa", "nsi", "gp", "cei", "ita", "ig",
                    "dist", "oscore"]
        avail = [v for v in variables if v in sy_names]
        # npz col 0 = return; X = arr[:,:,1:] -> npz index-1 = X column index
        feat_idx = [variables.index(n) - 1 for n in avail]
        n_features = len(feat_idx)
    else:
        feat_idx = list(range(X.shape[2]))
        n_features = len(feat_idx)
    print(f"  sy feat_idx={feat_idx} n_features={n_features}")

    windows = list(rolling_windows(list(range(T)), train=60, test=12))
    print(f"  {len(windows)} windows, period {common[0]}..{common[-1]}")

    # slice to the charset FIRST; core_idx for make_tensors is then identity
    X = np.clip(X[:, :, feat_idx], -10.0, 10.0)
    turnover_full = X_full[:, :, 2]
    core_idx = list(range(n_features))

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

        znet, sdfnet, cnet, val_loss, epochs_used, _tr_rp_mean, _oos_r2_inputs = train_window(
            R_tr, X_tr, mac_tr, R_va, X_va, mac_va, n_features,
            states=args.states, critic=args.critic, core_idx=core_idx)
        print(f"  window {wi} [{common[w_te[0]]}..{common[w_te[-1]]}]: "
              f"val_loss={val_loss:.2e} epochs={epochs_used}")

        znet.eval(); sdfnet.eval()
        mu = mac_tr.mean(axis=0); sd = mac_tr.std(axis=0) + 1e-12
        mac_all = torch.from_numpy(
            (np.concatenate([mac_tr, mac_va, mac_te]) - mu) / sd).float()
        X_all = np.concatenate([X_tr, X_va, X_te], axis=0)
        R_all = np.concatenate([R_tr, R_va, R_te], axis=0)
        R_all_t, X_all_t, mask_all_t = make_tensors(R_all, X_all, core_idx)
        with torch.no_grad():
            z_all = znet(mac_all)
            omega_all = sdfnet(z_all, X_all_t)
            M_all = common_sdf(omega_all, R_all_t, mask_all_t).numpy()
        omega_te = omega_all[-len(w_te):].numpy()
        M_te = M_all[-len(w_te):]
        mask_te = np.isfinite(R_te) & np.isfinite(X_te).all(axis=2)
        mask_te_t = torch.from_numpy(mask_te)
        R_te_t = torch.from_numpy(np.nan_to_num(R_te, nan=0.0)).float()
        # CPZ eval: SDF-portfolio return + common-SDF pricing errors (as train_e2)
        rp = sdf_portfolio_return(
            torch.from_numpy(omega_te).float(), R_te_t, mask_te_t).numpy()
        rp = rp[np.isfinite(rp)]
        alpha_te = pricing_errors_common(
            torch.from_numpy(M_te).float(), R_te_t, mask_te_t).numpy()
        alphas = [a for a in alpha_te if not np.isnan(a)]
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
