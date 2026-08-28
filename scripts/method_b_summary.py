#!/usr/bin/env python3
"""Method B (ex-ante sign identification, --pin-lambda) summary.

Scans results*/e2pin<L>*_{results,pooled_series}.csv (+ seed43/seed44),
compares the ex-ante pinned E2 series against (a) the as-trained series and
(b) the manuscript's post-hoc per-window sign convention, and writes
results/method_b_summary.json consumed by render_manuscript_3c.py.

Usage: python3 scripts/method_b_summary.py [--lambdas 1.0,10.0]
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COUNTRIES = {"IR": ROOT / "results", "TR": ROOT / "results_tr",
             "PK": ROOT / "results_pk"}
OUT = ROOT / "results" / "method_b_summary.json"


def load_series(p: Path) -> np.ndarray:
    rows = [r for r in p.read_text(encoding="utf-8-sig").splitlines()[1:]
            if r.strip()]
    return np.array([float(r.split(",")[0]) for r in rows])


def wmeans(v: np.ndarray, w: int) -> np.ndarray:
    return np.array([v[i * w:(i + 1) * w].mean() for i in range(len(v) // w)])


def sharpe(v: np.ndarray) -> float:
    return float(v.mean() / v.std(ddof=0) * math.sqrt(12))


def sign_norm(v: np.ndarray, w: int) -> float:
    """Post-hoc per-window sign convention (manuscript Table sign)."""
    segs = [v[i * w:(i + 1) * w] if v[i * w:(i + 1) * w].mean() >= 0
            else -v[i * w:(i + 1) * w] for i in range(len(v) // w)]
    return sharpe(np.concatenate(segs))


def collect(lam: float) -> dict:
    tag_ = f"{lam:g}"
    out = {}
    for c, base in COUNTRIES.items():
        rec = {}
        for sub in (".", "seed43", "seed44"):
            pre = base / sub
            pin_csv = pre / f"e2pin{tag_}_results.csv"
            if not pin_csv.exists():
                continue
            d = {r[0]: r[1] for r in csv.reader(
                pin_csv.open(encoding="utf-8-sig")) if len(r) >= 2}
            pin = load_series(pre / f"e2pin{tag_}_pooled_series.csv")
            raw = load_series(pre / "e2_pooled_series.csv")
            w = len(pin) // int(d["n_windows"])
            mr, mp = wmeans(raw, w), wmeans(pin, w)
            rec[sub] = {
                "seed": {".": 42, "seed43": 43, "seed44": 44}[sub],
                "sharpe": float(d["sharpe_pooled"]),
                "ev": float(d["ev"]),
                "rms_alpha_pct": float(d["rms_alpha_pct"]),
                "n_windows": int(d["n_windows"]),
                "raw_sharpe": sharpe(raw),
                "sign_norm_sharpe": sign_norm(raw, w),
                "pin_sign_norm_sharpe": sign_norm(pin, w),
                "window_sign_agreement": int(
                    (np.sign(mr) == np.sign(mp)).sum()),
            }
        if rec:
            vals = [r["sharpe"] for r in rec.values()]
            out[c] = {
                "per_seed": rec,
                "sharpe_range": [min(vals), max(vals)],
                "sharpe_mean": float(np.mean(vals)),
                "agreement_mean": float(np.mean(
                    [r["window_sign_agreement"] / r["n_windows"]
                     for r in rec.values()])),
                "n_seeds": len(vals),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", default="1.0,10.0",
                    help="comma-separated --pin-lambda values to summarize")
    args = ap.parse_args()
    out = {}
    for lam in (float(x) for x in args.lambdas.split(",")):
        data = collect(lam)
        if data:
            out[f"{lam:g}"] = data
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
