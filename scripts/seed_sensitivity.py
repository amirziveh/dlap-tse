#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_sensitivity.py — aggregate multi-seed robustness results
=============================================================
Reads results/{e2,e3,e8}_results.csv (seed 42) and results/seed{43,44}/
and e6_loadings_all{,_s43,_s44}.csv; writes results/seed_sensitivity.csv
for the manuscript robustness paragraph.
"""
import csv
import os
from pathlib import Path

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
RES = ROOT / "results"


def kv(path):
    d = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                d[row[0]] = row[1]
    return d


def main():
    rows = []
    for spec in ["e2", "e3", "e4a", "e4b", "e5a", "e5b", "e8", "e8b"]:
        for seed in [42, 43, 44]:
            if seed == 42:
                p = RES / f"{spec}_results.csv"
            else:
                p = RES / f"seed{seed}" / f"{spec}_results.csv"
            if not p.exists():
                continue
            d = kv(p)
            rows.append({"spec": spec.upper(), "seed": seed,
                         "sharpe": d.get("sharpe_pooled", ""),
                         "ev": d.get("ev", ""),
                         "rms_alpha_pct": d.get("rms_alpha_pct", "")})
    # loadings across seeds
    for seed in [42, 43, 44]:
        sfx = "" if seed == 42 else f"_s{seed}"
        p = RES / f"e6_loadings_all{sfx}.csv"
        if not p.exists():
            continue
        ld = {}
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                ld[r["char"]] = float(r["mean_w"])
        rows.append({"spec": "loadings_max|mean|", "seed": seed,
                     "sharpe": f"{max(abs(v) for v in ld.values()):.4f}",
                     "ev": f"{sum(abs(v) for v in ld.values())/len(ld):.4f}",
                     "rms_alpha_pct": ""})
    with open(RES / "seed_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["spec", "seed", "sharpe", "ev", "rms_alpha_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"{'spec':<20}{'seed':>5}{'sharpe':>9}{'EV':>8}{'RMS_alpha':>10}")
    for r in rows:
        print(f"{r['spec']:<20}{r['seed']:>5}{r['sharpe']:>9}{r['ev']:>8}{r['rms_alpha_pct']:>10}")
    print(f"\nSaved -> results/seed_sensitivity.csv")


if __name__ == "__main__":
    main()
