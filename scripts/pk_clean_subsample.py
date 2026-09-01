#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_clean_subsample.py — Pakistan high-confidence data robustness
================================================================
Filters PK stock-months to a "clean" subsample where ≥min_chars of 19
characteristics are non-missing, then reruns E2 and E8 to check whether
portfolio instability (Sharpe dispersion across seeds) persists.

Usage:
  python scripts/pk_clean_subsample.py --min-chars 15
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data_pk"
RES  = ROOT / "results_pk"
VENV = Path("/home/ubuntu/venvs/dlap-tse/bin/python")

CHARS_19 = [
    "ac", "ag", "bm", "cbop", "cei", "dist", "dy", "gp", "investment",
    "ita", "mom", "noa", "nsi", "oscore", "roe", "size", "st_rev",
    "turnover", "vol"
]


def build_clean_panel(min_chars=15):
    """Read characteristics_panel.csv, filter to high-coverage stock-months,
    write cleaned version, rebuild Char_all.npz."""
    csv_path = DATA / "characteristics_panel.csv"
    out_csv = DATA / "characteristics_panel_clean.csv"
    out_npz = DATA / "Char_all_clean.npz"

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    kept = []
    for r in rows:
        n_valid = sum(1 for c in CHARS_19 if r.get(c, "") != "")
        if n_valid >= min_chars:
            kept.append(r)

    pct = 100 * len(kept) / total
    print(f"Full: {total} rows → Clean (≥{min_chars} chars): {len(kept)} rows ({pct:.1f}%)")

    # Coverage comparison
    for c in CHARS_19:
        full_pct = sum(1 for r in rows if r.get(c, "") != "") / total * 100
        clean_pct = sum(1 for r in kept if r.get(c, "") != "") / len(kept) * 100 if kept else 0
        delta = clean_pct - full_pct
        if abs(delta) > 5:
            print(f"  {c:15s}: {full_pct:5.1f}% → {clean_pct:5.1f}% (+{delta:.1f}pp)")

    # Write clean CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(kept)
    print(f"Wrote: {out_csv}")

    # Rebuild NPZ in same format as Char_all.npz
    dates = sorted(set(f"{r['year']}-{int(r['month']):02d}" for r in kept))
    tickers = sorted(set(r["ticker"] for r in kept))
    # Determine variable list from CSV columns (exclude metadata)
    meta_cols = {"ticker", "year", "month", "ret_monthly"}
    variables = [c for c in rows[0].keys() if c not in meta_cols]

    date_idx = {d: i for i, d in enumerate(dates)}
    tick_idx = {t: i for i, t in enumerate(tickers)}
    var_idx = {v: i for i, v in enumerate(variables)}

    data = np.full((len(dates), len(tickers), len(variables)), np.nan)
    for r in kept:
        ds = f"{r['year']}-{int(r['month']):02d}"
        di = date_idx[ds]
        ti = tick_idx[r["ticker"]]
        for v in variables:
            val = r.get(v, "")
            if val != "":
                data[di, ti, var_idx[v]] = float(val)

    np.savez_compressed(out_npz, data=data, date=np.array(dates),
                        variable=np.array(variables), ticker=np.array(tickers))
    print(f"Wrote: {out_npz} shape={data.shape}")

    # Also need to rebuild risk_free_rate.csv and factors if they don't exist in data_pk
    # Check if they exist
    for needed in ["risk_free_rate.csv", "factors_2x3.csv"]:
        if not (DATA / needed).exists():
            print(f"  WARNING: {needed} not found in {DATA}")

    return out_csv, out_npz


def run_e2_e8(seeds=(42, 43, 44)):
    """Run E2 + E8 for given seeds using the clean NPZ (swapped in via env)."""
    for seed in seeds:
        for spec, extra in [("E2", ""), ("E8", "--liq-filter")]:
            print(f"\n--- {spec} seed={seed} (clean subsample) ---")
            cmd = [
                str(VENV), "scripts/train_e2.py",
                "--charset", "sy", "--states", "lstm",
                "--seed", str(seed)
            ]
            if extra:
                cmd.append(extra)
            env = os.environ.copy()
            env["DLAP_COUNTRY"] = "PK"
            # Temporarily swap Char_all.npz
            orig = DATA / "Char_all.npz"
            clean = DATA / "Char_all_clean.npz"
            backup = DATA / "Char_all_backup.npz"
            try:
                if orig.exists():
                    orig.rename(backup)
                clean.rename(orig)
                result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                        cwd=str(ROOT), timeout=600)
                print(result.stdout[-500:] if result.stdout else "")
                if result.returncode != 0:
                    print(f"STDERR: {result.stderr[-300:]}")
            finally:
                # Restore
                if orig.exists():
                    orig.rename(clean)
                if backup.exists():
                    backup.rename(orig)

    # Collect results
    print("\n=== Clean subsample results ===")
    for spec in ["e2", "e8"]:
        for seed in seeds:
            if seed == 42:
                p = RES / f"{spec}_results.csv"
            else:
                p = RES / f"seed{seed}" / f"{spec}_results.csv"
            if p.exists():
                with open(p) as f:
                    lines = f.readlines()
                sharpe = [l.split(",")[1].strip() for l in lines if l.startswith("OOS_Sharpe")]
                print(f"  {spec.upper()} seed={seed}: {sharpe}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-chars", type=int, default=15,
                    help="Minimum non-missing characteristics per stock-month (default 15/19)")
    ap.add_argument("--skip-build", action="store_true",
                    help="Skip panel rebuild, go straight to training")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    if not args.skip_build:
        build_clean_panel(min_chars=args.min_chars)
    run_e2_e8(seeds=tuple(args.seeds))


if __name__ == "__main__":
    main()
