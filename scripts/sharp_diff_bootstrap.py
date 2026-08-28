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
# country-aware results dir (same pattern as e7/seed_sensitivity)
_C = os.environ.get("DLAP_COUNTRY", "").upper()
RES = ROOT / {"TR": "results_tr", "PK": "results_pk"}.get(_C, "results")
BLOCK = int(os.environ.get("DLAP_BLOCK", "6"))
N_BOOT = 10_000
SEED = 42
OUT_SUFFIX = os.environ.get("DLAP_OUT_SUFFIX", "")


def load_series(fname):
    default_key = Path(fname).stem.split("_")[0].upper()  # e2/e8/... -> E2/E8
    out = {}
    with open(RES / fname, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if "model" in row:  # e1_pooled_series: model,oos_return
                out.setdefault(row["model"], []).append(float(row["oos_return"]))
            else:  # eN_pooled_series: oos_return
                out.setdefault(default_key, []).append(float(row["oos_return"]))
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="e2", help="deep-SDF spec series name (e.g. e2, e8)")
    ap.add_argument("--spec-label", default=None, help="label for the deep spec (default: spec upper)")
    args = ap.parse_args()
    spec = args.spec
    label = (args.spec_label or spec.upper())
    series = load_series(f"{spec}_pooled_series.csv")
    bench = load_series("e1_pooled_series.csv")
    sdf_full = series[label]
    rng = np.random.default_rng(SEED)
    pairs = ["FF5", "q-factor", "LASSO", "PCA(5)", "Market"]
    rows = []
    for b in pairs:
        if b not in bench:
            print(f"  skip {b}: benchmark series missing")
            continue
        bs = bench[b]
        if len(bs) != len(sdf_full):
            # a deep spec may skip an untrainable window (PK w0: nsi missing
            # before 2018-10) -> align on the COMMON TAIL (most recent OOS
            # months for both legs)
            n = min(len(bs), len(sdf_full))
            sdf = sdf_full[-n:]
            bs = bs[-n:]
            print(f"  align {b}: tail-overlap n={n} "
                  f"(deep {len(sdf_full)}, bench {len(bench[b])})")
        else:
            sdf = sdf_full
        diff_hat = sharpe_ann(sdf) - sharpe_ann(bs)
        diffs = np.array([block_boot_diff(sdf, bs, rng)
                          for _ in range(N_BOOT)])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"pair": f"{label} vs {b}",
                     "sharpe_sdf": f"{sharpe_ann(sdf):.4f}",
                     "sharpe_bench": f"{sharpe_ann(bs):.4f}",
                     "diff": f"{diff_hat:.4f}",
                     "ci_lo": f"{lo:.4f}",
                     "ci_hi": f"{hi:.4f}",
                     "zero_excluded": int(not (lo <= 0 <= hi))})
        print(f"  {label} vs {b:<8} diff={diff_hat:+.4f}  "
              f"95% [{lo:+.4f}, {hi:+.4f}]  zero_excluded={int(not (lo <= 0 <= hi))}")
    out = RES / f"sharp_diff_bootstrap_{label.lower()}{OUT_SUFFIX}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
