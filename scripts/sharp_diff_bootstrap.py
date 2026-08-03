#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sharp_diff_bootstrap.py — paired moving-block bootstrap of pooled OOS Sharpe
differences (E2 vs each benchmark), matching the procedure described in
manuscript Section 5: block length 6, 10,000 replications, seed 42, same
block indices resampled for both legs.

Inputs:  results/e2_pooled_series.csv, results/e1_pooled_series.csv
Outputs: results/sharp_diff_bootstrap.csv
"""
import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
RES = ROOT / "results"

BLOCK = 6
N_BOOT = 10_000
SEED = 42


def load_series(fname):
    out = {}
    with open(RES / fname, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if "model" in row:  # e1_pooled_series: model,oos_return
                out.setdefault(row["model"], []).append(float(row["oos_return"]))
            else:  # e2_pooled_series: oos_return
                out.setdefault("E2", []).append(float(row["oos_return"]))
    return {k: np.array(v) for k, v in out.items()}


def sharpe_ann(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(12)


def block_boot_diff(a, b, rng):
    """One moving-block resample of the paired difference, block length 6."""
    T = len(a)
    n_blocks = int(np.ceil(T / BLOCK))
    idx = np.empty(T, dtype=int)
    starts = rng.integers(0, T - BLOCK + 1, size=n_blocks)
    pos = 0
    for s in starts:
        idx[pos:pos + BLOCK] = np.arange(s, s + BLOCK)
        pos += BLOCK
    idx = idx[:T]
    return sharpe_ann(a[idx]) - sharpe_ann(b[idx])


def main():
    series = load_series("e2_pooled_series.csv")
    bench = load_series("e1_pooled_series.csv")
    e2 = series["E2"]
    rng = np.random.default_rng(SEED)
    pairs = ["FF5", "q-factor", "LASSO", "PCA(5)", "Market"]
    rows = []
    for b in pairs:
        if b not in bench or len(bench[b]) != len(e2):
            print(f"  skip {b}: series length mismatch "
                  f"({len(bench.get(b, []))} vs {len(e2)})")
            continue
        diff_hat = sharpe_ann(e2) - sharpe_ann(bench[b])
        diffs = np.array([block_boot_diff(e2, bench[b], rng)
                          for _ in range(N_BOOT)])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"pair": f"E2 vs {b}",
                     "sharpe_e2": f"{sharpe_ann(e2):.4f}",
                     "sharpe_bench": f"{sharpe_ann(bench[b]):.4f}",
                     "diff": f"{diff_hat:.4f}",
                     "ci_lo": f"{lo:.4f}",
                     "ci_hi": f"{hi:.4f}",
                     "zero_excluded": int(not (lo <= 0 <= hi))})
        print(f"  E2 vs {b:<8} diff={diff_hat:+.4f}  "
              f"95% [{lo:+.4f}, {hi:+.4f}]  zero_excluded={int(not (lo <= 0 <= hi))}")
    with open(RES / "sharp_diff_bootstrap.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {RES / 'sharp_diff_bootstrap.csv'}")


if __name__ == "__main__":
    main()
