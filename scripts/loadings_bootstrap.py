#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loadings_bootstrap.py — time-series-aware inference for the E6 characteristic
loadings (P4B of the audit).

The naive t-stat in e6_loadings.py treats the 144 monthly weight observations
as independent. They are not: the 12 months of each rolling window share one
trained network, and the LSTM state induces serial dependence. This script
reports, per characteristic:

  mean_w        pooled mean weight (same as e6_loadings)
  naive t       mean / (sd / sqrt(M))                     (as currently reported)
  boot t        t under a moving-block bootstrap (block 6) of the monthly
                weight series (10,000 reps, seed 42)
  boot CI       ~~~~~~~~~~~ 95% percentile interval
  signif_boot   |boot t| > 2

Outputs: results/e6_loadings_boot_all.csv, results/e6_loadings_boot_sy.csv
"""
import os
import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# country-aware results dir (same pattern as e7/seed_sensitivity)
_C = os.environ.get("DLAP_COUNTRY", "").upper()
RES = ROOT / {"TR": "results_tr", "PK": "results_pk"}.get(_C, "results")
BLOCK = 6
N_BOOT = 10_000
SEED = 42

CHARS_20 = ["size", "st_rev", "turnover", "vol", "bm", "mom", "roe", "ag", "ac",
            "noa", "nsi", "gp", "cei", "ita", "ig", "dist", "oscore",
            "investment", "cbop", "dy"]
SY_INDICES = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def _char_list(label):
    """Country-aware char names: derive from the weights file header minus
    metadata cols, so PK (ig-dropped) layouts match automatically."""
    fname = RES / f"e6_weights_{label}.csv"
    with open(fname, encoding="utf-8-sig", newline="") as f:
        hdr = next(csv.reader(f))
    meta = {"window", "test_period", "month", "date", "seed"}
    return [c for c in hdr if c.lower() not in meta]


def load_weights(label):
    fname = RES / f"e6_weights_{label}.csv"
    chars = _char_list(label)
    W = []
    with open(fname, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            W.append([float(row[c]) for c in chars])
    return np.array(W), chars


def boot_t_stat(w, rng):
    """Moving-block bootstrap of a monthly weight series: t = mean / boot_sd."""
    T = len(w)
    n_blocks = int(np.ceil(T / BLOCK))
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, T - BLOCK + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
        means[i] = w[idx].mean()
    sd = means.std(ddof=1)
    return w.mean() / sd if sd > 0 else np.nan, np.percentile(means, [2.5, 97.5])


def run(label):
    W, chars = load_weights(label)
    M = W.shape[0]
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"== loadings bootstrap: {label} ({len(chars)} chars, M={M}) ==")
    print(f"{'char':<14}{'mean_w':>9}{'naive_t':>9}{'boot_t':>9}{'boot_CI':>18}")
    for j, c in enumerate(chars):
        w = W[:, j]
        mean_w = w.mean()
        naive_t = mean_w / (w.std(ddof=1) / np.sqrt(M))
        bt, (lo, hi) = boot_t_stat(w, rng)
        rows.append({"char": c, "mean_w": f"{mean_w:.5f}",
                     "naive_t": f"{naive_t:.3f}", "boot_t": f"{bt:.3f}",
                     "boot_ci_lo": f"{lo:.5f}", "boot_ci_hi": f"{hi:.5f}",
                     "signif_boot": int(abs(bt) > 2)})
        print(f"{c:<14}{mean_w:>9.4f}{naive_t:>9.2f}{bt:>9.2f}"
              f"[{lo:+.4f},{hi:+.4f}]")
    with open(RES / f"e6_loadings_boot_{label}.csv", "w", newline="",
              encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"Saved -> e6_loadings_boot_{label}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--charset", choices=["sy", "all"], default="all")
    args = ap.parse_args()
    run(args.charset)
