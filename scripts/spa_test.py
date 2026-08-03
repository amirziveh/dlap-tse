#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spa_test.py — DLAP-TSE v0.6: SPA and Reality Check for SDF-portfolio Sharpe
============================================================================
Responds to the reviewer request for established forecast-comparison
procedures alongside the paired moving-block bootstrap:

  * Hansen (2005) SPA  — studentized, consistent recentering
  * White (2000) Reality Check — same statistic, no recentering (conservative)

For each target model (E2, E8) the loss differentials against the full
benchmark set {FF5, q-factor, PCA(5), LASSO, Market} are bootstrapped jointly
with a moving block bootstrap (block 6 and 12, 10,000 replications, seed 42,
identical block indices across models). Two loss conventions:

  * raw excess returns          -> mean return outperformance
  * volatility-normalized       -> Sharpe-ratio comparison (reviewer request)

Outputs: results/spa_test.csv
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
RES = ROOT / "results"

BLOCKS = (6, 12)
N_BOOT = 10_000
SEED = 42
BENCHMARKS = ["FF5", "q-factor", "PCA(5)", "LASSO", "Market"]
TARGETS = ["E2", "E8"]


def load_series(fname):
    """e1_pooled_series: model,oos_return; eN_pooled_series: oos_return."""
    default_key = Path(fname).stem.split("_")[0].upper()
    out = {}
    with open(RES / fname, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if "model" in row:
                out.setdefault(row["model"], []).append(float(row["oos_return"]))
            else:
                out.setdefault(default_key, []).append(float(row["oos_return"]))
    return {k: np.array(v, float) for k, v in out.items()}


def mbb_indices(n, block, rng):
    """One moving-block bootstrap index pattern (length n)."""
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nblocks)
    idx = np.concatenate([starts[i] + np.arange(block) for i in range(nblocks)])
    return idx[:n]


def studentized_spa(d, block, rng, n_boot=N_BOOT):
    """d: (n, k) loss differentials (positive = target better). Returns (spa_p, rc_p)."""
    n, k = d.shape
    mu = d.mean(axis=0)
    # bootstrap distribution for consistent std of sqrt(n)*mean
    idx = np.array([mbb_indices(n, block, rng) for _ in range(n_boot)])
    means = np.empty((n_boot, k))
    for j in range(n_boot):
        means[j] = d[idx[j]].mean(axis=0)
    sigma = means.std(axis=0) * np.sqrt(n)  # std of sqrt(n)*mean
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    T = float(np.max(np.sqrt(n) * mu / sigma))
    # consistent recentering (Hansen 2005)
    a_n = sigma * np.sqrt(2.0 * np.log(np.log(n))) / np.sqrt(n)
    recenter = np.where(mu < -a_n, 0.0, mu)
    # bootstrap statistics
    spa_exc = (means - recenter) * np.sqrt(n) / sigma
    rc_exc = means * np.sqrt(n) / sigma
    spa_p = float(np.mean(spa_exc.max(axis=1) >= T))
    rc_p = float(np.mean(rc_exc.max(axis=1) >= T))
    return spa_p, rc_p


def main():
    series = load_series("e1_pooled_series.csv")
    for t in TARGETS:
        series[t] = load_series(f"{t.lower()}_pooled_series.csv")[t]
    n = len(series["E2"])
    assert all(len(series[m]) == n for m in series), "series length mismatch"

    print(f"models: {list(series)}  n={n}")
    rows = []
    for target in TARGETS:
        r_t = series[target]
        for norm in ("ret", "sharpe"):
            if norm == "sharpe":
                r_t_n = r_t / (r_t.std() + 1e-12)
                bench_n = {b: series[b] / (series[b].std() + 1e-12)
                           for b in BENCHMARKS}
            else:
                r_t_n, bench_n = r_t, {b: series[b] for b in BENCHMARKS}
            d = np.column_stack([r_t_n - bench_n[b] for b in BENCHMARKS])
            diffs = {b: float((r_t_n - bench_n[b]).mean()) for b in BENCHMARKS}
            best = max(diffs, key=diffs.get)
            for block in BLOCKS:
                rng = np.random.default_rng(SEED)
                spa_p, rc_p = studentized_spa(d, block, rng)
                rows.append({
                    "target": target, "loss": norm, "block": block,
                    "n_models": len(BENCHMARKS),
                    "spa_p": f"{spa_p:.4f}", "rc_p": f"{rc_p:.4f}",
                    "best_benchmark": best,
                    "best_mean_diff": f"{diffs[best]:.4f}",
                })
                print(f"{target} [{norm}, block {block}]: SPA p={spa_p:.4f} "
                      f"RC p={rc_p:.4f}  best bench={best} "
                      f"(mean diff {diffs[best]:.4f})")
    with open(RES / "spa_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {RES/'spa_test.csv'}")


if __name__ == "__main__":
    sys.exit(main())
