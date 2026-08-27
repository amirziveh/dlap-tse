#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
placebo_summary.py — DLAP-TSE v0.6: summarize the placebo tests.
Placebo A (random 5% drop) vs Placebo B (noisy 5% drop: highest train-window
return volatility) vs E2 (full cross-section) vs E8 (liquidity-filtered),
across seeds 42/43/44. The question: does dropping the same NUMBER of stocks
by a non-liquidity rule also eliminate the catastrophic windows?

Outputs: results/placebo_summary.csv
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
# country-aware results dir (same pattern as e7/seed_sensitivity)
_C = os.environ.get("DLAP_COUNTRY", "").upper()
RES = ROOT / {"TR": "results_tr", "PK": "results_pk"}.get(_C, "results")
def load_series(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return np.array([float(r["oos_return"]) for r in csv.DictReader(f)])


def wsharpe(x):
    return [float(np.nanmean(x[i*12:(i+1)*12]) / np.nanstd(x[i*12:(i+1)*12])
                  * np.sqrt(12)) for i in range(12)]


def pooled_sharpe(x):
    return float(np.nanmean(x) / np.nanstd(x) * np.sqrt(12))


def main():
    specs = ["e2", "e8", "prandom", "pnoisy"]
    rows = []
    for seed in [42, 43, 44]:
        d = RES if seed == 42 else RES / f"seed{seed}"
        for spec in specs:
            if spec in ("prandom", "pnoisy"):
                p = RES / "placebo" / spec / (f"seed{seed}" if seed != 42 else ".") / \
                    f"{spec}_pooled_series.csv"
            else:
                p = d / f"{spec}_pooled_series.csv"
            if not p.exists():
                print(f"missing {p}")
                continue
            s = load_series(p)
            ws = wsharpe(s)
            n_neg = sum(1 for v in ws if v < 0)
            worst = min(ws)
            rows.append({"spec": spec.upper(), "seed": seed,
                         "pooled_sharpe": f"{pooled_sharpe(s):.3f}",
                         "n_neg_windows": n_neg,
                         "worst_window_sharpe": f"{worst:+.2f}",
                         "windows": " ".join(f"{v:+.2f}" for v in ws)})
            print(f"{spec.upper():<9} seed {seed}: pooled {pooled_sharpe(s):.3f}  "
                  f"neg-windows {n_neg}/12  worst {worst:+.2f}")
    with open(RES / "placebo_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {RES/'placebo_summary.csv'}")


if __name__ == "__main__":
    sys.exit(main())
