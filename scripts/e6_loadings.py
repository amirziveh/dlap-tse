#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e6_loadings.py — DLAP-TSE E6: which characteristics price TSE stocks
======================================================================
Trains the deep SDF (same protocol as E2/E3) and collects the SDF weight
vectors w(z_t) on OOS test months. Since characteristics are cross-sectionally
z-scored (std 1 per month), w_j IS the standardized loading of characteristic j
in the SDF  M = 1 - w(z)'x.

Interpretation: a positive w_j means high-x_j stocks receive a lower SDF value,
i.e. must earn higher expected returns -> characteristic j carries a positive
risk premium (and vice versa).

Outputs:
  results/e6_loadings_sy.csv  (11 SY signals)
  results/e6_loadings_all.csv (20 characteristics)
  results/e6_weights_sy.csv / e6_weights_all.csv (monthly weight series)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_e2 as T  # noqa: E402

CHARS_20 = ["size", "st_rev", "turnover", "vol", "bm", "mom", "roe", "ag", "ac",
            "noa", "nsi", "gp", "cei", "ita", "ig", "dist", "oscore",
            "investment", "cbop", "dy"]
SY_CHARS = [CHARS_20[i] for i in T.SY_INDICES]


def run(charset):
    feat_idx = T.SY_INDICES if charset == "sy" else list(range(20))
    char_names = SY_CHARS if charset == "sy" else CHARS_20
    label = "sy" if charset == "sy" else "all"
    print(f"== E6 loadings: {label} ({len(feat_idx)} chars) ==")

    R_exc, X, macro, common = T.load_data()
    windows = list(T.rolling_windows(list(range(len(common))), train=60, test=12))
    X = np.clip(X[:, :, feat_idx], -10.0, 10.0)

    rows = []  # (month, w vector)
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
            continue
        znet, mnet, cnet, val_loss, epochs = T.train_window(
            R_tr, X_tr, mac_tr, R_va, X_va, mac_va, len(feat_idx))
        znet.eval(); mnet.eval()
        mu = mac_tr.mean(axis=0); sd = mac_tr.std(axis=0) + 1e-12
        mac_all = torch.from_numpy(
            (np.concatenate([mac_tr, mac_va, mac_te]) - mu) / sd).float()
        with torch.no_grad():
            w_all = mnet(znet(mac_all)).numpy()
        for k, mi in enumerate(w_te):
            rows.append((common[mi], w_all[-len(w_te) + k]))
        print(f"  window {wi} [{common[w_te[0]]}..{common[w_te[-1]]}]: "
              f"val_loss={val_loss:.2e} epochs={epochs}")

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

    # ── save ───────────────────────────────────────────────────────────
    with open(T.RES / f"e6_loadings_{label}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["char", "mean_w", "std_w", "t_stat", "mean_abs_w", "n_months"])
        for j in range(len(char_names)):
            w.writerow([char_names[j], f"{mean_w[j]:.5f}", f"{std_w[j]:.5f}",
                        f"{t_stat[j]:.3f}", f"{mean_abs[j]:.5f}", M])
    with open(T.RES / f"e6_weights_{label}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month"] + char_names)
        for m, vec in zip(months, W):
            w.writerow([m] + [f"{v:.5f}" for v in vec])
    print(f"\nSaved -> e6_loadings_{label}.csv, e6_weights_{label}.csv "
          f"({M} OOS months)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--charset", choices=["sy", "all"], default="sy")
    args = ap.parse_args()
    run(args.charset)
