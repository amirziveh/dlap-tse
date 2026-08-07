#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_build_factors.py — DLAP-TSE Pakistan (PSX) Phase 3
======================================================
Builds FF5 (2x3) and HXZ q-factor benchmarks for Pakistan from the raw
characteristics panel, same conventions as TSE/Turkey:
  - per-month 1%/99% winsorized returns
  - size = exp(size char) -> median split; bm/roe/ag -> 30/70
  - value-weighted (mc weights)
Outputs (data_pk/factors_winsorized/): factors_2x3.csv, factors_q.csv
"""
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

PK = Path("/home/ubuntu/research/dlap-tse/data_pk")
OUT = PK / "factors_winsorized"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_panel():
    rf = {}
    for r in read_csv(PK / "risk_free_rate.csv"):
        y, m = r["month"].split("-")
        rf[(int(y), int(m))] = float(r["monthly_rate_pct"])
    bym = defaultdict(list)
    for r in read_csv(PK / "characteristics_panel.csv"):
        try:
            y, m = int(r["year"]), int(r["month"])
            ret = float(r["ret_monthly"])
            size = float(r["size"])
        except (ValueError, TypeError):
            continue
        mc = np.exp(size) if np.isfinite(size) else np.nan
        def f(c):
            try:
                v = float(r[c])
                return v if np.isfinite(v) else np.nan
            except (ValueError, TypeError):
                return np.nan
        bym[(y, m)].append({"ret": ret, "mc": mc, "size": size,
                            "bm": f("bm"), "roe": f("roe"), "ag": f("ag")})
    return bym, rf


def winsorize_rets(bym):
    for ym, rs in bym.items():
        vals = sorted(x["ret"] for x in rs if np.isfinite(x["ret"]))
        if len(vals) < 20:
            continue
        lo = vals[max(0, int(len(vals) * 0.01) - 1)]
        hi = vals[min(len(vals) - 1, int(len(vals) * 0.99))]
        for x in rs:
            x["ret"] = float(np.clip(x["ret"], lo, hi))


def vw(rs):
    wsum = sum(x["mc"] for x in rs if np.isfinite(x["mc"]) and np.isfinite(x["ret"]))
    if wsum <= 0:
        return np.nan
    return sum(x["ret"] * x["mc"] for x in rs
               if np.isfinite(x["mc"]) and np.isfinite(x["ret"])) / wsum


def build():
    bym, rf = load_panel()
    winsorize_rets(bym)
    ff5_rows, q_rows = [], []
    for ym in sorted(bym):
        rs = bym[ym]
        rfv = rf.get(ym, 0.0)
        valid = [x for x in rs if np.isfinite(x["ret"]) and np.isfinite(x["mc"])]
        if len(valid) < 15:
            continue
        def split(var, pct):
            vals = sorted(x[var] for x in valid if np.isfinite(x[var]))
            if not vals:
                return None
            return vals[int(len(vals) * pct)]
        size_cut = split("size", 0.5)
        bm_cut, bm_cut2 = split("bm", 0.3), split("bm", 0.7)
        roe_cut, roe_cut2 = split("roe", 0.3), split("roe", 0.7)
        ag_cut, ag_cut2 = split("ag", 0.3), split("ag", 0.7)
        if None in (size_cut, bm_cut, bm_cut2, roe_cut, roe_cut2, ag_cut, ag_cut2):
            continue
        def cell(pred):
            return [x for x in valid if pred(x)]
        SL = cell(lambda x: x["size"] <= size_cut and x["bm"] <= bm_cut)
        SH = cell(lambda x: x["size"] <= size_cut and x["bm"] > bm_cut2)
        BL = cell(lambda x: x["size"] > size_cut and x["bm"] <= bm_cut)
        BH = cell(lambda x: x["size"] > size_cut and x["bm"] > bm_cut2)
        SMB = (vw(SL) + vw(SH) - vw(BL) - vw(BH)) / 2
        HML = (vw(SH) + vw(BH) - vw(SL) - vw(BL)) / 2
        S_R = vw(cell(lambda x: x["size"] <= size_cut and x["roe"] > roe_cut2))
        S_W = vw(cell(lambda x: x["size"] <= size_cut and x["roe"] <= roe_cut))
        B_R = vw(cell(lambda x: x["size"] > size_cut and x["roe"] > roe_cut2))
        B_W = vw(cell(lambda x: x["size"] > size_cut and x["roe"] <= roe_cut))
        RMW = ((S_R + B_R) - (S_W + B_W)) / 2
        S_C = vw(cell(lambda x: x["size"] <= size_cut and x["ag"] <= ag_cut))
        S_A = vw(cell(lambda x: x["size"] <= size_cut and x["ag"] > ag_cut2))
        B_C = vw(cell(lambda x: x["size"] > size_cut and x["ag"] <= ag_cut))
        B_A = vw(cell(lambda x: x["size"] > size_cut and x["ag"] > ag_cut2))
        CMA = ((S_C + B_C) - (S_A + B_A)) / 2
        mkt = vw(valid)
        ff5_rows.append({"year": ym[0], "month": ym[1],
                         "Mkt_RF": f"{mkt - rfv:.6f}", "SMB": f"{SMB:.6f}",
                         "HML": f"{HML:.6f}", "RMW": f"{RMW:.6f}",
                         "CMA": f"{CMA:.6f}"})
        r_me = SMB
        r_ia = ((S_C + B_C) - (S_A + B_A)) / 2
        r_roe = RMW
        q_rows.append({"year": ym[0], "month": ym[1],
                       "Mkt_RF": f"{mkt - rfv:.6f}", "ME": f"{r_me:.6f}",
                       "IA": f"{r_ia:.6f}", "ROE": f"{r_roe:.6f}"})
    for fname, rows, cols in [("factors_2x3.csv", ff5_rows,
                               ["year", "month", "Mkt_RF", "SMB", "HML", "RMW", "CMA"]),
                              ("factors_q.csv", q_rows,
                               ["year", "month", "Mkt_RF", "ME", "IA", "ROE"])]:
        with open(OUT / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"{fname}: {len(rows)} months")
    print("saved to", OUT)


if __name__ == "__main__":
    build()
