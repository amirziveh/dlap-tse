#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master_results.py — consolidate all model results into master_results.csv
===============================================================================
Reads the per-spec CSVs (deep SDF cpz + charscore robustness, linear SDF,
benchmarks) and writes a single master table used by the manuscript:

  results/master_results.csv  (model, sharpe, ev, rms_alpha_pct, max_alpha_pct, note)
"""
import csv
import os
from pathlib import Path

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
RES = ROOT / "results"


def read_kv(path):
    d = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                d[row[0]] = row[1]
    return d


def read_e1():
    out = {}
    with open(RES / "e1_benchmarks.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[r["name"]] = r
    return out


def main():
    e1 = read_e1()
    rows = []
    for m in ["Market", "FF5", "q-factor", "PCA(5)", "LASSO"]:
        r = e1[m]
        rows.append({"model": m, "sharpe": r["sharpe_pooled"],
                     "ev": r["ev"], "rms_alpha_pct": r["rms_alpha_pct"],
                     "max_alpha_pct": r.get("max_alpha_pct", ""),
                     "note": "benchmark"})
    # linear SDF benchmark (common, linear-in-characteristics)
    with open(RES / "linear_sdf_results.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"model": f"Linear SDF ({'11ch' if r['charset']=='sy' else '20ch'})",
                         "sharpe": r["sharpe_pooled"], "ev": r["ev"],
                         "rms_alpha_pct": r["rms_alpha_pct"],
                         "max_alpha_pct": r["max_alpha_pct"],
                         "note": "linear SDF (common, omega=theta'x)"})
    # deep SDF specs (cpz)
    for spec, note in [("e2", "11ch LSTM"), ("e3", "20ch LSTM"),
                       ("e4a", "20ch const"), ("e4b", "11ch const"),
                       ("e5a", "11ch LSTM critic"), ("e5b", "20ch LSTM critic"),
                       ("e8", "11ch LSTM liq-filter"), ("e8b", "20ch LSTM liq-filter")]:
        p = RES / f"{spec}_results.csv"
        if not p.exists():
            continue
        d = read_kv(p)
        rows.append({"model": spec.upper(), "sharpe": d.get("sharpe_pooled", ""),
                     "ev": d.get("ev", ""), "rms_alpha_pct": d.get("rms_alpha_pct", ""),
                     "max_alpha_pct": d.get("max_alpha_pct", ""),
                     "note": note + " (cpz common SDF)"})
    # charscore robustness (legacy per-stock linear characteristic SDF)
    for spec, note in [("e2", "11ch LSTM"), ("e3", "20ch LSTM")]:
        p = RES / "charscore" / f"{spec}_results.csv"
        if not p.exists():
            continue
        d = read_kv(p)
        rows.append({"model": f"{spec.upper()}-CS", "sharpe": d.get("sharpe_pooled", ""),
                     "ev": d.get("ev", ""), "rms_alpha_pct": d.get("rms_alpha_pct", ""),
                     "max_alpha_pct": d.get("max_alpha_pct", ""),
                     "note": note + " (characteristic-score SDF)"})

    with open(RES / "master_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "sharpe", "ev",
                                          "rms_alpha_pct", "max_alpha_pct", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/master_results.csv ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['model']:<20} Sharpe={r['sharpe']:>8} EV={r['ev']:>8} "
              f"RMS_alpha={r['rms_alpha_pct']:>8}%")


if __name__ == "__main__":
    main()
