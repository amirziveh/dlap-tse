#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rms_window_bootstrap.py — paired window-level bootstrap of RMS pricing-error
differences (revision P2, 2026-08-31).

Design
------
The paper's RMS alpha is a window-level statistic: within each OOS window w,
RMS_w = sqrt( mean_i( alpha_{w,i}^2 ) ) over stocks i with valid cells; the
pooled RMS is sqrt( mean_w( RMS_w^2 ) ) = sqrt( mean_w mean_i alpha^2 ).
Because each window's 12 OOS months share one trained model (and benchmark
windows share estimation data), months are NOT independent; the defensible
resampling unit is the WINDOW (cluster bootstrap, paired: same resampled
window indices for both legs). Model selection, weights, and errors are all
frozen at their released values — no re-estimation.

For each pair (deep spec, benchmark) and seed s:
  d* = [ RMS_deep(B) - RMS_bench(B) ] over B = resampled window multiset,
  10,000 replicates, seed 42 for the RNG (reproducible).
Reported: point diff, 2.5/97.5 percentile CI, P(d* <= 0), P(d* >= 0).

Inputs:  results{,_tr,_pk}/{<spec>_alpha_cells.csv, e1_alpha_cells.csv,
         linear_sdf_{lin11,lin20}_alpha_cells.csv}
Output:  results{,_tr,_pk}/rms_window_bootstrap.csv
"""
import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
_C = os.environ.get("DLAP_COUNTRY", "").upper()
RES = ROOT / {"TR": "results_tr", "PK": "results_pk"}.get(_C, "results")
TAG = {"TR": "TR", "PK": "PK"}.get(_C, "IR")
N_BOOT = 10_000
SEED = 42

DEEP_SPECS = ["e2", "e3", "e4a", "e4b", "e5a", "e5b", "e8", "e8b"]
BENCH = {
    "Market": "e1_alpha_cells.csv:Market",
    "q-factor": "e1_alpha_cells.csv:q-factor",
    "LASSO": "e1_alpha_cells.csv:LASSO",
    "PCA(5)": "e1_alpha_cells.csv:PCA(5)",
    "Linear SDF (SY)": "linear_sdf_lin11_alpha_cells.csv",
    "Linear SDF (all)": "linear_sdf_lin20_alpha_cells.csv",
}


def load_cells(path, model=None):
    """-> dict window -> np.array of alphas. Window keys normalized to 0-based
    (e1 writer uses a 1-based counter; deep specs use the 0-based window id)."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if model is not None and row["model"] != model:
                continue
            out.setdefault(int(row["window"]), []).append(float(row["alpha"]))
    keys = sorted(out)
    if keys and min(keys) == 1 and max(keys) - min(keys) + 1 == len(keys):
        out = {k - 1: v for k, v in out.items()}
    return {k: np.asarray(v) for k, v in out.items()}


def rms_by_window(cells):
    """RMS_w per window, aligned to the sorted union of windows."""
    ws = sorted(cells)
    return ws, np.asarray([float(np.sqrt(np.mean(cells[w] ** 2))) for w in ws])


def boot_rms_diff(a_cells, b_cells, rng):
    """Paired window-cluster bootstrap of RMS difference."""
    # intersect windows (both legs must exist in a resampled window set)
    ws = sorted(set(a_cells) & set(b_cells))
    A = np.asarray([float(np.sqrt(np.mean(a_cells[w] ** 2))) for w in ws])
    B = np.asarray([float(np.sqrt(np.mean(b_cells[w] ** 2))) for w in ws])
    n = len(ws)
    point = float(np.sqrt(np.mean(A ** 2)) - np.sqrt(np.mean(B ** 2))) * 100  # -> percentage points
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        diffs[b] = (np.sqrt(np.mean(A[idx] ** 2)) - np.sqrt(np.mean(B[idx] ** 2))) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_le0 = float(np.mean(diffs <= 0))
    p_ge0 = float(np.mean(diffs >= 0))
    return point, float(lo), float(hi), p_le0, p_ge0, n


def main():
    rng = np.random.default_rng(SEED)
    # benchmark cells
    e1_path = RES / "e1_alpha_cells.csv"
    bench_cells = {}
    if e1_path.exists():
        for name in ["Market", "q-factor", "LASSO", "PCA(5)"]:
            bench_cells[name] = load_cells(e1_path, model=name)
    for label, fn in [("Linear SDF (SY)", "linear_sdf_lin11_alpha_cells.csv"),
                      ("Linear SDF (all)", "linear_sdf_lin20_alpha_cells.csv")]:
        p = RES / fn
        if p.exists():
            bench_cells[label] = load_cells(p)
    ae_p = RES / "e1_ae_alpha_cells.csv"
    if ae_p.exists():
        for name in ["AE(1)", "AE(3)", "AE(5)"]:
            bench_cells[name] = load_cells(ae_p, model=name)

    rows = []
    for spec in DEEP_SPECS:
        p = RES / f"{spec}_alpha_cells.csv"
        if not p.exists():
            continue
        deep_cells = load_cells(p)
        for bench_name, b_cells in bench_cells.items():
            point, lo, hi, ple, pge, n = boot_rms_diff(deep_cells, b_cells, rng)
            common_ws = sorted(set(deep_cells) & set(b_cells))
            deep_pool = np.concatenate([deep_cells[w] ** 2 for w in common_ws]) if common_ws else np.array([np.nan])
            bench_pool = np.concatenate([b_cells[w] ** 2 for w in common_ws]) if common_ws else np.array([np.nan])
            rows.append({"market": TAG, "spec": spec.upper(), "benchmark": bench_name,
                         "rms_deep_pct": round(float(np.sqrt(np.mean(deep_pool)) * 100), 4),
                         "rms_bench_pct": round(float(np.sqrt(np.mean(bench_pool)) * 100), 4),
                         "diff_pp": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "p_diff_le0": round(ple, 4), "p_diff_ge0": round(pge, 4),
                         "n_windows": n})

    out = RES / "rms_window_bootstrap.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} pairs)")
    for r in rows:
        if r["spec"] in ("E2", "E3"):
            print(f"  {r['spec']:>4} vs {r['benchmark']:<16} "
                  f"diff={r['diff_pp']:+.2f}pp CI[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] "
                  f"P(le0)={r['p_diff_le0']:.3f}")


if __name__ == "__main__":
    main()
