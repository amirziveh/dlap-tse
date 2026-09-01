#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common_calendar.py — Common-calendar robustness for DLAP-TSE
=============================================================
Reruns E2+E8 on all 3 countries using only the overlapping period
(2014-01 to 2026-07, 149 months → 5 OOS windows from 2019-01).

This eliminates the apples-to-oranges complaint from unequal OOS periods.

Usage:
  python scripts/common_calendar.py
"""
import os
import subprocess
import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
VENV = Path("/home/ubuntu/venvs/dlap-tse/bin/python")

# Common period: 2014-01 to 2026-07 (149 months, 5 OOS windows from 2019-01)
COMMON_START = "2014-01"
COMMON_END   = "2026-07"


def truncate_panel(country_dir, out_dir, start=COMMON_START, end=COMMON_END):
    """Truncate characteristics_panel.csv and rebuild NPZ for the common period."""
    src_csv = country_dir / "characteristics_panel.csv"
    if not src_csv.exists():
        print(f"  WARNING: {src_csv} not found, skipping")
        return False

    out_dir.mkdir(exist_ok=True, parents=True)
    out_csv = out_dir / "characteristics_panel.csv"

    with open(src_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    filtered = []
    for r in rows:
        ds = f"{r['year']}-{int(r['month']):02d}"
        if start <= ds <= end:
            filtered.append(r)

    if not filtered:
        print(f"  No rows in period {start}..{end}")
        return False

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(filtered)
    print(f"  {out_csv.name}: {len(filtered)} rows ({start}..{end})")

    # Rebuild NPZ
    dates = sorted(set(f"{r['year']}-{int(r['month']):02d}" for r in filtered))
    tickers = sorted(set(r["ticker"] for r in filtered))
    meta_cols = {"ticker", "year", "month", "ret_monthly"}
    variables = [c for c in rows[0].keys() if c not in meta_cols]

    date_idx = {d: i for i, d in enumerate(dates)}
    tick_idx = {t: i for i, t in enumerate(tickers)}
    var_idx = {v: i for i, v in enumerate(variables)}

    data = np.full((len(dates), len(tickers), len(variables)), np.nan)
    for r in filtered:
        ds = f"{r['year']}-{int(r['month']):02d}"
        di = date_idx[ds]
        ti = tick_idx[r["ticker"]]
        for v in variables:
            val = r.get(v, "")
            if val != "":
                data[di, ti, var_idx[v]] = float(val)

    np.savez_compressed(out_dir / "Char_all.npz",
                        data=data, date=np.array(dates),
                        variable=np.array(variables), ticker=np.array(tickers))

    # Copy risk_free_rate.csv if it exists and isn't already in the trunc dir
    rf_src = country_dir / "risk_free_rate.csv"
    rf_dst = out_dir / "risk_free_rate.csv"
    if rf_src.exists() and not rf_dst.exists():
        import shutil
        shutil.copy(rf_src, rf_dst)

    print(f"  Char_all.npz: {data.shape}")
    return True


def run_country(country_code, data_dir, seeds=(42,)):
    """Run E2+E8 on a truncated country panel."""
    for seed in seeds:
        for spec, extra in [("E2", ""), ("E8", "--liq-filter")]:
            cmd = [str(VENV), "scripts/train_e2.py",
                   "--charset", "sy", "--states", "lstm", "--seed", str(seed)]
            if extra:
                cmd.append(extra)

            env = os.environ.copy()
            # Override paths to use truncated data
            env["DLAP_COUNTRY"] = country_code
            # Temporarily point to the truncated data dir
            orig_data = ROOT / (f"data_{country_code.lower()}" if country_code != "IR" else "data")
            backup_data = ROOT / f"_data_{country_code.lower()}_backup"
            try:
                orig_data.rename(backup_data)
                data_dir.rename(orig_data)
                result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                        cwd=str(ROOT), timeout=600)
                # Print summary line
                for line in (result.stdout or "").split("\n"):
                    if "pooled" in line.lower() or "e2 " in line.lower() or "e8 " in line.lower():
                        print(f"    {line.strip()}")
            finally:
                # Restore
                if orig_data.exists():
                    orig_data.rename(data_dir)
                if backup_data.exists():
                    backup_data.rename(orig_data)


def collect_results():
    """Collect and compare original vs common-calendar results."""
    print("\n=== Common-calendar comparison ===")
    print(f"Period: {COMMON_START} to {COMMON_END}")
    print(f"{'Country':<8} {'Spec':<6} {'Sharpe':>8} {'EV':>8} {'RMS%':>8}")
    print("-" * 42)

    for country, res_dir in [("IR", ROOT / "results"),
                              ("TR", ROOT / "results_tr"),
                              ("PK", ROOT / "results_pk")]:
        for spec in ["e2_results.csv", "e8_results.csv"]:
            # Original results (from full period)
            orig_path = res_dir / spec
            # Common-calendar results (from truncated run)
            cc_path = res_dir / "common_calendar" / spec
            for label, path in [("full", orig_path), ("CC", cc_path)]:
                if path.exists():
                    with open(path) as f:
                        lines = f.readlines()
                    data = {}
                    for l in lines:
                        parts = l.strip().split(",")
                        if len(parts) == 2:
                            data[parts[0].strip()] = parts[1].strip()
                    model = data.get("model", spec.replace("_results.csv", "").upper())
                    # Try to find sharpe, ev, rms in various formats
                    shp = data.get("OOS_Sharpe", data.get("sharpe", "?"))
                    ev = data.get("OOS_EV", data.get("ev", "?"))
                    rms = data.get("OOS_RMS_alpha_pct", data.get("rms_alpha_pct", "?"))
                    print(f"{country:<8} {label} {model:<15} {shp:>8} {ev:>8} {rms:>8}")


def main():
    print(f"=== Common-calendar experiment: {COMMON_START} to {COMMON_END} ===")

    countries = [
        ("IR", ROOT / "data",         ROOT / "_common_cal_ir"),
        ("TR", ROOT / "data_tr",      ROOT / "_common_cal_tr"),
        ("PK", ROOT / "data_pk",      ROOT / "_common_cal_pk"),
    ]

    for code, src, trunc_dir in countries:
        print(f"\n--- {code}: truncating ---")
        ok = truncate_panel(src, trunc_dir)
        if ok:
            print(f"--- {code}: running E2+E8 ---")
            run_country(code, trunc_dir, seeds=(42,))

    collect_results()


if __name__ == "__main__":
    main()
