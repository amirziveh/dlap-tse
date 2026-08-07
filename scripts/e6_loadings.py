#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e6_loadings.py — DLAP-TSE E6: which characteristics price TSE stocks
======================================================================
Trains the deep SDF (same protocol as E2/E3, CPZ common-SDF architecture)
and collects the SDF weight vectors omega_t(i) on OOS test months.

Loading definition (CPZ-style): for each OOS month t, run the cross-sectional
regression of the SDF weights on the characteristics,

    omega_t(i) = a_t + beta_t' x_{i,t} + eps_{i,t}     (OLS over stocks)

and report the Fama-MacBeth average of beta_t. Since characteristics are
cross-sectionally z-scored (std 1 per month), beta_j IS the standardized
loading: how much SDF portfolio weight (omega) characteristic j commands.

Interpretation: a positive beta_j means high-x_j stocks receive more weight in
the SDF portfolio, i.e. must earn higher expected returns -> characteristic j
carries a positive risk premium (and vice versa).

Sign convention: omega is sign-normalized per window (SDF portfolio mean
return positive) — the squared-pricing-error loss pins the sign only up to
local-optimum flips across windows.

Outputs (same format as before, so loadings_bootstrap.py works unchanged):
  results/e6_loadings_sy.csv  (11 SY signals)
  results/e6_loadings_all.csv (20 characteristics)
  results/e6_weights_sy.csv / e6_weights_all.csv (monthly loading series)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_e2 as T  # noqa: E402
from sdf_models import common_sdf, sdf_portfolio_return  # noqa: E402

CHARS_20 = ["size", "st_rev", "turnover", "vol", "bm", "mom", "roe", "ag", "ac",
            "noa", "nsi", "gp", "cei", "ita", "ig", "dist", "oscore",
            "investment", "cbop", "dy"]
SY_CHARS = [CHARS_20[i] for i in T.SY_INDICES]


def run(charset, seed=42):
    torch.manual_seed(seed)
    R_exc, X, macro, common = T.load_data()
    _, _, variables_all, _ = T.load_npz()
    feat_idx = T.charset_indices(variables_all) if charset == "sy" else list(range(X.shape[2]))
    core_pos = [feat_idx.index(i) for i in T.charset_indices(variables_all)]
    char_names = [v for v in variables_all[1:] if v in (T.SY_NAMES if charset == "sy" else variables_all)]
    label = "sy" if charset == "sy" else "all"
    print(f"== E6 loadings (cpz): {label} ({len(feat_idx)} chars) ==")

    windows = list(T.rolling_windows(list(range(len(common))), train=60, test=12))
    X = np.clip(X[:, :, feat_idx], -10.0, 10.0)

    rows = []  # (month, beta vector)
    for wi, (w_tr, w_te) in enumerate(windows):
        w_va = w_tr[-12:]
        w_tr = w_tr[:-12]
        R_tr, X_tr, mac_tr = R_exc[w_tr], X[w_tr], macro[w_tr]
        R_va, X_va, mac_va = R_exc[w_va], X[w_va], macro[w_va]
        R_te, X_te, mac_te = R_exc[w_te], X[w_te], macro[w_te]
        keep = np.isfinite(R_tr).sum(axis=0) >= 12
        R_tr, X_tr = R_tr[:, keep], X_tr[:, keep]
        R_va, X_va = R_va[:, keep], X_va[:, keep]
        R_te, X_te = R_te[:, keep], X_te[:, keep]
        if R_tr.shape[1] < 50:
            print(f"  window {wi}: skipped ({R_tr.shape[1]} stocks)")
            continue
        znet, sdfnet, cnet, val_loss, epochs = T.train_window(
            R_tr, X_tr, mac_tr, R_va, X_va, mac_va, len(feat_idx),
            arch="cpz", core_idx=core_pos)
        znet.eval(); sdfnet.eval()
        mu = mac_tr.mean(axis=0); sd = mac_tr.std(axis=0) + 1e-12
        mac_all = torch.from_numpy(
            (np.concatenate([mac_tr, mac_va, mac_te]) - mu) / sd).float()
        X_all = np.concatenate([X_tr, X_va, X_te], axis=0)
        R_all = np.concatenate([R_tr, R_va, R_te], axis=0)
        R_all_t, X_all_t, mask_all_t = T.make_tensors(R_all, X_all, core_pos)
        with torch.no_grad():
            z_all = znet(mac_all)
            omega_all = sdfnet(z_all, X_all_t)
            M_all = common_sdf(omega_all, R_all_t, mask_all_t).numpy()
        omega_te = omega_all[-len(w_te):].numpy()
        M_te = M_all[-len(w_te):]
        mask_te = np.isfinite(R_te) & np.isfinite(X_te).all(axis=2)
        mask_te_t = torch.from_numpy(mask_te)
        R_te_t = torch.from_numpy(np.nan_to_num(R_te, nan=0.0)).float()
        rp = sdf_portfolio_return(
            torch.from_numpy(omega_te).float(), R_te_t, mask_te_t).numpy()
        if rp.mean() < 0:
            omega_te = -omega_te  # sign convention: SDF portfolio mean return >= 0
        # per-month UNIVARIATE cross-sectional OLS of omega on each characteristic
        # (CPZ-style: characteristic by characteristic; avoids the near-collinearity
        # of investment-like signals in the multivariate regression, e.g. ag vs I/A)
        for k, mi in enumerate(w_te):
            xm = X_te[k]                    # (N,F)
            wm = omega_te[k]                # (N,)
            m = mask_te[k]                  # (N,)
            if m.sum() < 10:
                continue
            betas = []
            for j in range(xm.shape[1]):
                xj = xm[m, j]
                wv = wm[m]
                if np.var(xj) < 1e-12:
                    betas.append(0.0)
                    continue
                Xd = np.column_stack([np.ones(len(xj)), xj])
                b, *_ = np.linalg.lstsq(Xd, wv, rcond=None)
                betas.append(b[1])
            rows.append((common[mi], np.array(betas)))
        print(f"  window {wi} [{common[w_te[0]]}..{common[w_te[-1]]}]: "
              f"val_loss={val_loss:.2e} epochs={epochs}")

    if not rows:
        print("FATAL: no months collected")
        sys.exit(1)
    W = np.array([w for _, w in rows])  # (M, F)
    months = [m for m, _ in rows]
    M = W.shape[0]
    mean_w = W.mean(axis=0)
    std_w = W.std(axis=0)
    t_stat = mean_w / (std_w / np.sqrt(M))
    mean_abs = np.abs(W).mean(axis=0)

    order = np.argsort(-mean_abs)
    print(f"\n{'char':<14}{'mean_w':>10}{'std_w':>9}{'t-stat':>9}{'mean|w|':>9}")
    print("-" * 54)
    for j in order:
        print(f"{char_names[j]:<14}{mean_w[j]:>10.4f}{std_w[j]:>9.4f}"
              f"{t_stat[j]:>9.2f}{mean_abs[j]:>9.4f}")

    # ── save (seed != 42 keeps a suffixed copy for stability checks) ──
    sfx = "" if seed == 42 else f"_s{seed}"
    with open(T.RES / f"e6_loadings_{label}{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["char", "mean_w", "std_w", "t_stat", "mean_abs_w", "n_months"])
        for j in range(len(char_names)):
            w.writerow([char_names[j], f"{mean_w[j]:.5f}", f"{std_w[j]:.5f}",
                        f"{t_stat[j]:.3f}", f"{mean_abs[j]:.5f}", M])
    with open(T.RES / f"e6_weights_{label}{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month"] + char_names)
        for m, vec in zip(months, W):
            w.writerow([m] + [f"{v:.5f}" for v in vec])
    print(f"\nSaved -> e6_loadings_{label}{sfx}.csv, e6_weights_{label}{sfx}.csv "
          f"({M} OOS months)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--charset", choices=["sy", "all"], default="sy")
    ap.add_argument("--seed", type=int, default=42,
                    help="torch seed (multi-seed stability checks)")
    args = ap.parse_args()
    run(args.charset, seed=args.seed)
