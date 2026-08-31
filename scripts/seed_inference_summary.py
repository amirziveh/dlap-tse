#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_inference_summary.py — per-seed formal inference for the revision's
Priority 1 (no seed-42-only formal inference) + Priority 8 input (2026-08-31).

For each market (IR/TR/PK), seed s ∈ {42,43,44}, and deep spec (E2, E8):
  1. pooled OOS Sharpe of the as-trained series;
  2. paired moving-block bootstrap (block 6, 10k, seed 42) Sharpe difference
     vs every benchmark (Market, FF5, q-factor, PCA(5), LASSO) — same block
     indices for both legs, tail-overlap alignment as in sharp_diff_bootstrap;
  3. sign-symmetry loss-gap diagnostics from <spec>_sign_symmetry.csv;
  4. zero-exclusion classification per pair (does the 95% CI exclude 0?).

Also aggregates E2 sign-normalized Sharpe per seed (diagnostic, clearly
labeled ex-post).

Output: results/seed_inference_summary.csv (+ stdout table)
"""
import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
CC = {"IR": "results", "TR": "results_tr", "PK": "results_pk"}
BLOCK = 6
N_BOOT = 10_000
SEED = 42


def sharpe_ann(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(12)


def load_series(path):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    if rows and rows[0] and rows[0][0] == "model":
        out = {}
        for r in rows[1:]:
            if len(r) >= 2:
                out.setdefault(r[0], []).append(float(r[1]))
        return {k: np.asarray(v) for k, v in out.items()}
    return {"SERIES": np.asarray([float(r[0]) for r in rows[1:] if r and r[0]])}


def boot_diff_paired(a, b, rng, block=BLOCK):
    """Moving-block bootstrap of Sharpe(a) - Sharpe(b); resampled indices shared."""
    T = min(len(a), len(b))
    a, b = np.asarray(a[-T:], float), np.asarray(b[-T:], float)  # tail alignment
    n_blocks = int(np.ceil(T / block))
    diffs = np.empty(N_BOOT)
    for k in range(N_BOOT):
        starts = rng.integers(0, T - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        diffs[k] = sharpe_ann(a[idx]) - sharpe_ann(b[idx])
    point = sharpe_ann(a) - sharpe_ann(b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


def main():
    out_rows = []
    for tag, res in CC.items():
        e1 = load_series(res and ROOT / res / "e1_pooled_series.csv")
        for spec in ["e2", "e8"]:
            for seed in [42, 43, 44]:
                sub = ROOT / res / ("" if seed == 42 else f"seed{seed}")
                p = sub / f"{spec}_pooled_series.csv"
                if not p.exists():
                    continue
                deep = load_series(p)["SERIES"]
                row = {"market": tag, "spec": spec.upper(), "seed": seed,
                       "sharpe": round(sharpe_ann(deep), 4)}
                rng = np.random.default_rng(SEED + hash((tag, spec)) % 1000)
                for bname, bser in e1.items():
                    pt, lo, hi = boot_diff_paired(deep, bser, rng)
                    row[f"d_{bname}"] = round(pt, 4)
                    row[f"ci_{bname}"] = f"[{lo:.3f},{hi:.3f}]"
                    row[f"zeroex_{bname}"] = int((lo > 0) or (hi < 0))
                # sign-symmetry diagnostic (seed-42 file sits at top level)
                sym_p = ROOT / res / f"{spec}_sign_symmetry.csv"
                if seed == 42 and sym_p.exists():
                    gaps = [float(r["rel_gap"]) for r in
                            csv.DictReader(open(sym_p, encoding="utf-8-sig"))]
                    row["sym_median_gap"] = round(float(np.median(gaps)), 3)
                out_rows.append(row)

    out = ROOT / "results" / "seed_inference_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        if out_rows:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    print(f"wrote {out} ({len(out_rows)} rows)\n")
    hdr = f"{'mkt':<4}{'spec':<5}{'seed':<6}{'Sharpe':>8}  {'vs strongest-bench zero-excl?':>10}"
    print(hdr)
    for r in out_rows:
        # count how many benchmark CIs exclude zero and in which direction
        zx = [(k[7:], r[k]) for k in r if k.startswith("zeroex_") and r[k]]
        dirs = []
        for name, _ in zx:
            ci = r[f"ci_{name}"]
            lo = float(ci.strip("[]").split(",")[0])
            dirs.append(f"{name}({'+' if lo > 0 else '-'})")
        print(f"{r['market']:<4}{r['spec']:<5}{r['seed']:<6}{r['sharpe']:>8}  "
              f"{','.join(dirs) if dirs else 'none':<10}")


if __name__ == "__main__":
    main()
