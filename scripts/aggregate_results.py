#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_results.py — Collect all seed expansion, IPCA, clean subsample,
and common-calendar results into summary tables for the manuscript.
"""
import csv
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))


def read_results(path):
    """Read key-value results CSV into dict."""
    if not path.exists():
        return None
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                d[parts[0].strip()] = parts[1].strip()
    return d


def collect_seeds():
    """Collect seed sensitivity across all countries."""
    countries = {"IR": ROOT / "results", "TR": ROOT / "results_tr", "PK": ROOT / "results_pk"}
    specs = ["e2", "e8"]
    seeds = list(range(42, 45)) + list(range(100, 110))  # original + new

    rows = []
    for country, res_dir in countries.items():
        for spec in specs:
            for seed in seeds:
                if seed == 42:
                    p = res_dir / f"{spec}_results.csv"
                else:
                    p = res_dir / f"seed{seed}" / f"{spec}_results.csv"
                d = read_results(p)
                if d is None:
                    continue
                try:
                    shp = float(d.get("sharpe_pooled", d.get("sharpe", 0)))
                    ev = float(d.get("ev", d.get("OOS_EV", 0)))
                    rms = float(d.get("rms_alpha_pct", d.get("OOS_RMS_alpha_pct", 0)))
                    # Sanity check: reject absurd values
                    if abs(shp) > 10 or rms > 100:
                        continue
                    rows.append({
                        "country": country, "spec": spec.upper(),
                        "seed": seed, "sharpe": shp, "ev": ev, "rms": rms
                    })
                except (ValueError, TypeError):
                    continue

    # Summary stats per country × spec
    print("=" * 80)
    print("SEED SENSITIVITY SUMMARY (10 seeds: 42-44, 100-109)")
    print("=" * 80)
    print(f"{'Country':<6} {'Spec':<5} {'N':>3} {'Median Shp':>10} {'Mean Shp':>10} "
          f"{'Std':>8} {'IQR':>14} {'%Positive':>9} {'Med RMS%':>9}")
    print("-" * 80)

    for country in ["IR", "TR", "PK"]:
        for spec in ["E2", "E8"]:
            subset = [r for r in rows if r["country"] == country and r["spec"] == spec]
            if len(subset) < 3:
                continue
            sharpes = np.array([r["sharpe"] for r in subset])
            rms_vals = np.array([r["rms"] for r in subset])
            med_shp = np.median(sharpes)
            mean_shp = np.mean(sharpes)
            std_shp = np.std(sharpes)
            q25, q75 = np.percentile(sharpes, [25, 75])
            pct_pos = np.mean(sharpes > 0) * 100
            med_rms = np.median(rms_vals)
            print(f"{country:<6} {spec:<5} {len(subset):>3} {med_shp:>10.4f} {mean_shp:>10.4f} "
                  f"{std_shp:>8.4f} [{q25:.3f},{q75:.3f}] {pct_pos:>8.1f}% {med_rms:>8.2f}%")

    return rows


def collect_ipca():
    """Collect IPCA results."""
    print("\n" + "=" * 80)
    print("IPCA BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Country':<6} {'Best K':>6} {'Sharpe':>10} {'RMS%':>10}")
    print("-" * 40)

    for country, res_dir in [("IR", ROOT / "results"), ("TR", ROOT / "results_tr"), ("PK", ROOT / "results_pk")]:
        d = read_results(res_dir / "ipca_results.csv")
        if d is None:
            print(f"{country:<6} {'N/A':>6}")
            continue
        model = d.get("model", "?")
        # Find best K from the model name
        best_k = model.split("(")[1].split(")")[0] if "(" in model else "?"
        # Find best sharpe
        best_shp = 0
        best_rms = 0
        for k in [3, 5, 8]:
            shp = d.get(f"IPCA({k})_sharpe")
            rms = d.get(f"IPCA({k})_rms_pct")
            if shp and float(shp) > best_shp:
                best_shp = float(shp)
                best_rms = float(rms) if rms else 0
        print(f"{country:<6} {best_k:>6} {best_shp:>10.4f} {best_rms:>10.2f}%")


def collect_common_calendar():
    """Collect common-calendar results."""
    print("\n" + "=" * 80)
    print("COMMON-CALENDAR RESULTS (2014-01 to 2026-07)")
    print("=" * 80)
    print(f"{'Country':<6} {'Spec':<5} {'Full Shp':>9} {'CC Shp':>9} {'Full RMS%':>10} {'CC RMS%':>9}")
    print("-" * 60)

    for country, res_dir, cc_dir in [
        ("IR", ROOT / "results", ROOT / "results" / "common_calendar"),
        ("TR", ROOT / "results_tr", ROOT / "results_tr" / "common_calendar"),
        ("PK", ROOT / "results_pk", ROOT / "results_pk" / "common_calendar"),
    ]:
        for spec in ["e2", "e8"]:
            full = read_results(res_dir / f"{spec}_results.csv")
            cc = read_results(cc_dir / f"{spec}_results.csv")
            full_shp = float(full.get("sharpe_pooled", 0)) if full else 0
            full_rms = float(full.get("rms_alpha_pct", 0)) if full else 0
            cc_shp = float(cc.get("sharpe_pooled", 0)) if cc else 0
            cc_rms = float(cc.get("rms_alpha_pct", 0)) if cc else 0
            if cc:
                print(f"{country:<6} {spec.upper():<5} {full_shp:>+9.4f} {cc_shp:>+9.4f} "
                      f"{full_rms:>9.2f}% {cc_rms:>8.2f}%")
        if not (cc_dir / "e2_results.csv").exists():
            print(f"{country:<6} {'N/A (no CC data)'}")


def collect_pk_clean():
    """Collect PK clean subsample results."""
    print("\n" + "=" * 80)
    print("PAKISTAN CLEAN SUBSAMPLE (≥15/19 chars, 72.1% retention)")
    print("=" * 80)
    print(f"{'Spec':<5} {'Full Shp':>9} {'Clean Shp':>10} {'Full RMS%':>10} {'Clean RMS%':>11}")
    print("-" * 50)

    pk_dir = ROOT / "results_pk"
    clean_dir = pk_dir / "clean_subsample"
    full_e2 = read_results(pk_dir / "e2_results.csv")
    full_e8 = read_results(pk_dir / "e8_results.csv")
    clean_e2 = read_results(clean_dir / "e2_results.csv")
    clean_e8 = read_results(clean_dir / "e8_results.csv")

    for spec, full, clean in [("E2", full_e2, clean_e2), ("E8", full_e8, clean_e8)]:
        if not clean:
            print(f"{spec:<5} clean results missing")
            continue
        fs = float(full.get("sharpe_pooled", 0)) if full else 0
        fr = float(full.get("rms_alpha_pct", 0)) if full else 0
        cs = float(clean.get("sharpe_pooled", 0))
        cr = float(clean.get("rms_alpha_pct", 0))
        print(f"{spec:<5} {fs:>+9.4f} {cs:>+10.4f} {fr:>9.2f}% {cr:>10.2f}%")


def main():
    print("DLAP-TSE: Aggregated Results Summary")
    print("=" * 80)
    rows = collect_seeds()
    collect_ipca()
    collect_common_calendar()
    collect_pk_clean()

    # Save master seed CSV
    out = ROOT / "results" / "seed_sensitivity_expanded.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["country", "spec", "seed", "sharpe", "ev", "rms"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved seed data -> {out}")


if __name__ == "__main__":
    main()
