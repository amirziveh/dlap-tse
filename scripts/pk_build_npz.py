#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_build_npz.py — DLAP-TSE Pakistan (PSX) Phase 4
==================================================
Converts PK CSV panels into npz files mirroring the official CPZ layout
(same as build_npz.py for TSE):

  Char_all.npz:   data  = [T x N x (1 + F)] float32
                  [:, :, 0]  = monthly return (missing = -99.99)
                  [:, :, 1:] = z-scored characteristics (missing = -99.99)
  Macro_all.npz:  data  = [T x M] float64 raw macro levels

Outputs (data_pk/): Char_all.npz, Macro_all.npz, meta.json
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PK = Path("/home/ubuntu/research/dlap-tse/data_pk")
UNKNOWN = -99.99

CHARS = ["size", "st_rev", "turnover", "vol", "bm", "mom", "roe", "ag", "ac",
         "noa", "nsi", "gp", "cei", "ita", "ig", "dist", "oscore",
         "investment", "cbop", "dy"]  # 20 (ig from LT investments)
MACROS = ["policy_rate", "tbill", "cpi_ix", "cpi_yoy", "usd_pkr", "brent"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_csv(PK / "characteristics_z.csv")
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

    # ---- winsorize monthly returns 1%/99% per month -------------------------
    # PSX raw closes carry split/right-issue jumps (up to +1396%); the factor
    # construction already winsorizes; align the SDF panel to the same series
    # (documented in data_pk/macro_sources.md).
    for i in range(T):
        col = data[i, :, 0]
        vals = col[col != UNKNOWN]
        if len(vals) < 20:
            continue
        lo, hi = np.percentile(vals, [1, 99])
        clipped = np.clip(vals, lo, hi)
        col[col != UNKNOWN] = clipped

    np.savez_compressed(PK / "Char_all.npz", data=data,
                        date=[f"{y}-{m:02d}" for y, m in months],
                        variable=["return"] + CHARS, ticker=tickers)
    print(f"Char_all.npz: {data.shape}")

    # macro
    mrows = read_csv(PK / "macro_panel.csv")
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
    np.savez_compressed(PK / "Macro_all.npz", data=macro,
                        date=[f"{y}-{m:02d}" for y, m in months],
                        variable=MACROS)
    print(f"Macro_all.npz: {macro.shape}")

    meta = {"months": [f"{y}-{m:02d}" for y, m in months],
            "tickers": tickers, "chars": CHARS, "macros": MACROS,
            "n_months": T, "n_tickers": N, "n_chars": F}
    json.dump(meta, open(PK / "meta.json", "w"), indent=1)
    print("meta.json saved")


if __name__ == "__main__":
    main()
