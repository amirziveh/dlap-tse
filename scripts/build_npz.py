#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_npz.py — DLAP-TSE Phase 1
================================
Converts the CSV panels into npz files mirroring the official CPZ layout
(see code/official_cpz + code/torch_port torch_version/data_loader.py):

  individual npz:  data  = [T x N x (1 + F)] float32
                   [:, :, 0]   = monthly return (missing = -99.99)
                   [:, :, 1:]  = z-scored characteristics (missing = -99.99)
                   date    = list of 'YYYY-MM' (length T)
                   variable= list of names (length 1+F)
                   ticker  = list of Persian tickers (length N)
  macro npz:       data  = [T x M] float64 raw macro levels
                   date, variable

One full-history file set (the training script will slice rolling 60/12/12 windows):
  data/Char_all.npz, data/Macro_all.npz, data/meta.json
"""
import csv
import os
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse"))) / "data"
UNKNOWN = -99.99

CHARS = ["size", "st_rev", "turnover", "vol", "bm", "mom", "roe", "ag", "ac",
         "noa", "nsi", "gp", "cei", "ita", "ig", "dist", "oscore",
         "cbop", "dy"]  # investment(I/A) removed 2026-08-31: byte-identical to ag
MACROS = ["cbirate", "cpi", "usd_official", "brent", "gold_coin", "usd_market"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    # --- characteristics (z-scored) ---
    rows = read_csv(OUT / "characteristics_z.csv")
    months = sorted({(int(r["year"]), int(r["month"])) for r in rows})
    tickers = sorted({r["ticker"] for r in rows})
    T, N = len(months), len(tickers)
    F = len(CHARS)
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

    date_list = [f"{y}-{m:02d}" for y, m in months]
    variable_list = ["return"] + CHARS
    np.savez(OUT / "Char_all.npz",
             data=data,
             date=date_list,
             variable=variable_list,
             ticker=tickers)
    print(f"  saved Char_all.npz  shape={data.shape}  unknown_fraction="
          f"{(data == UNKNOWN).mean():.3f}")

    # --- macro ---
    mrows = read_csv(OUT / "macro_panel.csv")
    macro_dates = [r["month"] for r in mrows]
    # align macro dates to char dates (macro may extend beyond)
    mdata = np.full((T, len(MACROS)), np.nan)
    m_by_date = {r["month"]: r for r in mrows}
    for i, d in enumerate(date_list):
        r = m_by_date.get(d)
        if r is None:
            continue
        for k, c in enumerate(MACROS):
            v = r.get(c, "")
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = math.nan
            mdata[i, k] = v
    np.savez(OUT / "Macro_all.npz",
             data=mdata,
             date=date_list,
             variable=MACROS)
    print(f"  saved Macro_all.npz shape={mdata.shape}  coverage="
          f"{(~np.isnan(mdata)).mean():.3f}")

    meta = {
        "months": date_list,
        "tickers": tickers,
        "characteristics": CHARS,
        "macros": MACROS,
        "unknown_val": UNKNOWN,
        "n_months": T,
        "n_tickers": N,
        "note": "Chars are per-month cross-sectional z-scores (winsorized 1%/99%). "
                "Macro = raw levels, normalized at train time (train mean/std). "
                "Rolling windows 60/12/12 are sliced by the training script.",
    }
    with open(OUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  saved meta.json ({N} tickers, {T} months)")


if __name__ == "__main__":
    main()
