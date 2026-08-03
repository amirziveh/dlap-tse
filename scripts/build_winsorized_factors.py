#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_winsorized_factors.py — Symmetric data prep for benchmarks
================================================================
The deep-SDF panel winsorizes monthly stock returns at the 1%/99%
percentile within each month (build_characteristics.py). The FF5 and
q-factor benchmark portfolios were previously built from RAW returns,
so benchmark inputs contained capital-increase artifacts (e.g. +198.7%
P_CMA_SC in 2009-09) that the deep SDF never saw — asymmetric data prep.

This script rebuilds both factor sets from the SAME winsorized returns:
  1. monthly_returns_winsorized.csv  (per-month 1/99 clip, same rule as
     build_characteristics.py, written for reproducibility)
  2. factors_winsorized/factors_2x3.csv   (FF5 via fama-five construct_factors)
  3. factors_winsorized/factors_q.csv     (HXZ q via build_qfactors)

Outputs live in DLAP_ROOT/data/factors_winsorized/; the original raw-based
files are left untouched (eval_core is pointed at the winsorized ones).
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMA = Path(os.environ.get("FAMA_ROOT", str(Path.home() / "research/fama-five/data")))
ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
OUT = ROOT / "data" / "factors_winsorized"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def winsorize_monthly_returns():
    """Per-month 1%/99% winsorization, identical rule to build_characteristics.py."""
    rows = read_csv(FAMA / "processed" / "monthly_returns.csv")
    by_month = defaultdict(list)
    for r in rows:
        try:
            v = float(r["ret_monthly"])
        except (ValueError, TypeError):
            continue
        if np.isfinite(v):
            by_month[(r["year"], r["month"])].append(r)
    n_clip = 0
    for ym, rs in by_month.items():
        vals = sorted(float(r["ret_monthly"]) for r in rs)
        lo = vals[max(0, int(len(vals) * 0.01) - 1)]
        hi = vals[min(len(vals) - 1, int(len(vals) * 0.99))]
        for r in rs:
            x = float(r["ret_monthly"])
            if x < lo or x > hi:
                r["ret_monthly"] = str(float(np.clip(x, lo, hi)))
                n_clip += 1
    out = ROOT / "data" / "monthly_returns_winsorized.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"winsorized {n_clip} returns ({n_clip / len(rows) * 100:.2f}%) "
          f"-> {out.name}")


def build_ff5():
    """FF5 2x3 factors from winsorized returns (monkey-patch construct_factors)."""
    sys.path.insert(0, str(Path.home() / "research/fama-five/scripts"))
    import construct_factors as cf

    # patch: winsorized returns
    def load_monthly_returns_ws():
        returns = defaultdict(dict)
        with open(ROOT / "data" / "monthly_returns_winsorized.csv") as f:
            for r in csv.DictReader(f):
                t = r["ticker"]
                ym = (int(r["year"]), int(r["month"]))
                ret = float(r["ret_monthly"])
                n_days = int(r["n_days"])
                if n_days >= 3:
                    returns[t][ym] = ret
        return returns

    cf.load_monthly_returns = load_monthly_returns_ws
    cf.FACTORS_DIR = OUT
    cf.PORTFOLIO_DIR = OUT / "portfolio_returns"
    cf.PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    cf.construct_factors("2x3")
    print("FF5 winsorized ->", OUT / "factors_2x3.csv")


def build_q():
    """HXZ q factors from winsorized returns (monkey-patch build_qfactors)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_qfactors as bq

    # patch: winsorized returns + winsorized Mkt_RF + output dir
    def load_returns_mcap_ws():
        returns = defaultdict(dict)
        mcap = defaultdict(dict)
        for row in bq.read_csv(ROOT / "data" / "monthly_returns_winsorized.csv"):
            t = row["ticker"]
            try:
                y, m = int(row["year"]), int(row["month"])
                returns[t][(y, m)] = float(row["ret_monthly"])
            except (ValueError, KeyError):
                continue
        for row in bq.read_csv(FAMA / "processed" / "market_cap_monthly.csv"):
            t = row["ticker"]
            try:
                y, m = int(row["year"]), int(row["month"])
                mcap[t][(y, m)] = float(row["market_cap"])
            except (ValueError, KeyError):
                continue
        return returns, mcap

    bq.load_returns_mcap = load_returns_mcap_ws
    bq.OUT = OUT

    # patch Mkt_RF source inside main by intercepting read_csv for factors file
    orig_read = bq.read_csv

    def read_csv_patched(path):
        p = Path(path)
        if str(p).endswith("factors_2x3.csv") and "factors_winsorized" not in str(p):
            p = OUT / "factors_2x3.csv"
        return orig_read(p)

    bq.read_csv = read_csv_patched
    bq.main()
    print("q winsorized ->", OUT / "factors_q.csv")


if __name__ == "__main__":
    winsorize_monthly_returns()
    build_ff5()
    build_q()
