#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tr_build_npz.py — DLAP-TSE Turkey (BIST) Phase 4
=================================================
Converts TR CSV panels into npz files (same layout as build_npz.py / pk_build_npz.py):

  Char_all.npz:   data = [T x N x (1 + F)] float32, return + z-scored chars
  Macro_all.npz:  data = [T x M] float64 raw macro levels

Same return winsorization (1%/99% per month) as PK — BIST raw closes carry
split-style jumps (max +702% in the panel).
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TR = Path("/home/ubuntu/research/dlap-tse/data_tr")
UNKNOWN = -99.99

CHARS = ["size", "st_rev", "turnover", "vol", "bm", "mom", "roe", "ag", "ac",
         "noa", "nsi", "gp", "cei", "ita", "ig", "dist", "oscore",
         "cbop", "dy"]  # 19: investment(I/A) removed 2026-08-31 (identical to ag)
MACROS = ["cbrate", "cpi", "usd_try", "tbill3m", "brent", "m2"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_csv(TR / "characteristics_z.csv")
    months = sorted({(int(r["year"]), int(r["month"])) for r in rows})
    tickers = sorted({r["ticker"] for r in rows})
    T, N, F = len(months), len(tickers), len(CHARS)
    print(f"chars: {T} months x {N} tickers x {F} features")

    t_idx = {m: i for i, m in enumerate(months)}
    n_idx = {t: i for i, t in enumerate(tickers)}

    data = np.full((T, N, 1 + F), UNKNOWN, dtype=np.float32)
    for r in rows:
        i = t_idx[(int(r["year"]), int(r["month"]))]
        j = n_idx[r["ticker"]]
        try:
            ret = float(r["ret_monthly"])
        except (TypeError, ValueError):
            ret = math.nan
        data[i, j, 0] = UNKNOWN if not math.isfinite(ret) else ret
        for k, c in enumerate(CHARS):
            v = r.get(c, "")
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = math.nan
            data[i, j, 1 + k] = UNKNOWN if not math.isfinite(v) else v

    # winsorize monthly returns 1%/99% per month (same as PK; BIST split jumps)
    for i in range(T):
        col = data[i, :, 0]
        vals = col[col != UNKNOWN]
        if len(vals) < 20:
            continue
        lo, hi = np.percentile(vals, [1, 99])
        col[col != UNKNOWN] = np.clip(vals, lo, hi)

    np.savez_compressed(TR / "Char_all.npz", data=data,
                        date=[f"{y}-{m:02d}" for y, m in months],
                        variable=["return"] + CHARS, ticker=tickers)
    print(f"Char_all.npz: {data.shape}")

    mrows = read_csv(TR / "macro_panel.csv")
    macro = np.full((T, len(MACROS)), np.nan, dtype=np.float64)
    for r in mrows:
        key = (int(r["month"][:4]), int(r["month"][5:7]))
        if key not in t_idx:
            continue
        i = t_idx[key]
        for k, c in enumerate(MACROS):
            v = r.get(c, "")
            try:
                macro[i, k] = float(v)
            except (TypeError, ValueError):
                pass
    np.savez_compressed(TR / "Macro_all.npz", data=macro,
                        date=[f"{y}-{m:02d}" for y, m in months],
                        variable=MACROS)
    print(f"Macro_all.npz: {macro.shape}")


if __name__ == "__main__":
    main()
