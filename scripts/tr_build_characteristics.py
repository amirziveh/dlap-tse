#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tr_build_characteristics.py — DLAP-TSE Turkey (BIST) Phase 2
==============================================================
Builds the 20-characteristic monthly panel for Turkey, mirroring the TSE
constructions (scripts/build_characteristics.py + fama-five anomaly signals)
with the İş Yatırım XI_29 industrial chart.

Outputs (in dlap-tse/data_tr/):
  characteristics_panel.csv  — raw panel: ticker, year, month, ret, 20 chars
  characteristics_z.csv      — winsorized (1%/99%) + cross-sectionally z-scored

Conventions (identical to TSE):
  - formation year: month (y,m), m>=7 -> formation y; m<7 -> formation y-1
  - financials for fiscal year f apply from July of year f+1 (announced ~March)
  - financial firms excluded (non-industrial charts + GYO real-estate trusts)
"""
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
DATA = ROOT / "data_tr"
PROC = DATA / "processed"
FIN = DATA / "financials"

CHARS = ["size", "st_rev", "turnover", "vol", "bm", "mom", "roe", "ag", "ac",
         "noa", "nsi", "gp", "cei", "ita", "ig", "dist", "oscore",
         "investment", "cbop", "dy"]

# ---- İş Yatırım XI_29 industrial chart: canonical item codes -------------
C_CA = "1A"      # current assets
C_CASH = "1AA"   # cash and cash equivalents
C_STINV = "1AB"  # short-term financial investments
C_TA = "1BL"     # total assets
C_PPE = "1BG"    # tangible fixed assets
C_LTINV = "1BC"  # long-term financial investments (equity-method in 1BD)
C_LTINV2 = "1BD" # investments with equity method
C_CL = "2A"      # short-term liabilities
C_STDEBT = "2AA" # short-term financial loans
C_LTDEBT = "2B"  # long-term liabilities
C_DEFTAX = "2BG" # deferred tax liabilities
C_EQUITY = "2N"  # shareholders' equity
C_MINOR = "2ODA" # minority interests
C_TLEQ = "2ODB"  # total liabilities + shareholders' equity
C_REV = "3C"     # net sales
C_COGS = "3CA"   # cost of sales
C_GP = "3D"      # gross profit
C_NI = "3L"      # net profit after taxes
C_DIV = "4CBB"   # dividends paid


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_returns():
    out = defaultdict(dict)  # ticker -> (y,m) -> ret
    for r in read_csv(PROC / "monthly_returns.csv"):
        if r["ret_monthly"]:
            out[r["ticker"]][(int(r["year"]), int(r["month"]))] = float(r["ret_monthly"])
    return out


def load_mcaps():
    out = defaultdict(dict)  # ticker -> (y,m) -> market cap
    for r in read_csv(PROC / "market_cap_monthly.csv"):
        out[r["ticker"]][(int(r["year"]), int(r["month"]))] = float(r["market_cap"])
    return out


def load_capital():
    out = defaultdict(dict)  # ticker -> (y,m) -> capital (SERMAYE, TL)
    for r in read_csv(PROC / "shares_panel.csv"):
        out[r["ticker"]][(int(r["year"]), int(r["month"]))] = float(r["capital"])
    return out


def load_volumes():
    out = defaultdict(dict)
    for r in read_csv(PROC / "volume_monthly.csv"):
        out[r["ticker"]][(int(r["year"]), int(r["month"]))] = float(r["volume_tl"])
    return out


def load_financials():
    """ticker -> fiscal_year -> {code: value}  (annual period-12 values)"""
    out = defaultdict(dict)
    for tdir in FIN.iterdir():
        if not tdir.is_dir():
            continue
        tkr = tdir.name
        for yf in tdir.glob("*.json"):
            year = int(yf.stem)
            try:
                d = json_load(yf)
            except Exception:
                continue
            v = d.get("value") or []
            if len(v) not in (110, 139, 145, 147):
                continue  # non-industrial charts (banks 192 / insurance 405 / misc)
            # chart versions: 110 (FY2013-15), 139 (FY2016-17), 147 (FY2018-23),
            # 145 (FY2024+); canonical codes are identical across versions
            # (110-chart lacks only 4CBB dividends)
            rec = {}
            for r in v:
                code = r["itemCode"]
                val = r.get("value1")
                if val in (None, "", "-"):
                    continue
                try:
                    rec[code] = float(val)
                except (TypeError, ValueError):
                    continue
            if rec:
                out[tkr][year] = rec
    return out


import json


def json_load(p):
    return json.load(open(p, encoding="utf-8"))


def get(rec, code):
    v = rec.get(code)
    return v if v is not None else 0.0


def annual_signals(tkr, years, fin, mcaps, caps):
    """Formation-year signals for one ticker (mirrors fama-five constructions)."""
    out = {}  # formation_year -> {char: value}
    ys = sorted(years)
    for i, fy in enumerate(ys):
        d = fin[fy]
        prev = fin.get(fy - 1, {})
        if not d:
            continue
        formation = fy + 1
        ta, ta_p = get(d, C_TA), get(prev, C_TA)
        tl = get(d, C_TLEQ) - get(d, C_EQUITY)
        tl_p = get(prev, C_TLEQ) - get(prev, C_EQUITY)
        be = get(d, C_EQUITY) + get(d, C_DEFTAX) - get(d, C_MINOR)
        be_p = get(prev, C_EQUITY) + get(prev, C_DEFTAX) - get(prev, C_MINOR)
        ni = get(d, C_NI)
        ni_p = get(prev, C_NI)
        s = {}
        # nsi: YoY change in registered capital (December of fiscal year)
        cap_t = (caps.get(tkr, {}).get((fy, 12)) or
                 caps.get(tkr, {}).get((fy, 11)) or 0)
        cap_p = (caps.get(tkr, {}).get((fy - 1, 12)) or
                 caps.get(tkr, {}).get((fy - 1, 11)) or 0)
        s["nsi"] = (cap_t - cap_p) / cap_p if cap_p > 0 else None
        # cei: (equity_t - equity_{t-1}) / TA_{t-1}
        s["cei"] = (be - be_p) / ta_p if ta_p and ta_p > 0 else None
        # NOA = (TA - cash - st_inv) - (TL - st_debt - lt_debt)
        def noa_of(rec, rec_p):
            ta_v = get(rec, C_TA)
            if not ta_v:
                return None
            op_a = ta_v - get(rec, C_CASH) - get(rec, C_STINV)
            tl_v = get(rec, C_TLEQ) - get(rec, C_EQUITY)
            op_l = tl_v - get(rec, C_STDEBT) - get(rec, C_LTDEBT)
            return op_a - op_l
        noa_t = noa_of(d, prev)
        noa_p = noa_of(prev, d)
        # ac: ΔNOA / TA_{t-1}
        s["ac"] = ((noa_t - noa_p) / ta_p) if (noa_t is not None and noa_p is not None and ta_p and ta_p > 0) else None
        # noa: NOA_t / TA_t
        s["noa"] = (noa_t / ta) if (noa_t is not None and ta and ta > 0) else None
        # ag: ΔTA / TA_{t-1}
        s["ag"] = (ta - ta_p) / ta_p if (ta and ta_p and ta_p > 0) else None
        # investment (q-factor I/A): same ΔTA/TA
        s["investment"] = s["ag"]
        # ita: ΔPP&E / TA_{t-1}
        ppe, ppe_p = get(d, C_PPE), get(prev, C_PPE)
        s["ita"] = (ppe - ppe_p) / ta_p if (ppe and ppe_p and ta_p and ta_p > 0) else None
        # ig: YoY % change in long-term investments
        li = get(d, C_LTINV) + get(d, C_LTINV2)
        li_p = get(prev, C_LTINV) + get(prev, C_LTINV2)
        s["ig"] = (li - li_p) / li_p if (li and li_p and li_p > 0) else None
        # dist: TLMTA - NIMTA (Campbell-style, with market cap)
        me = mcaps.get(tkr, {}).get((formation, 6)) or mcaps.get(tkr, {}).get((formation, 7))
        if me and me > 0 and ta and ta > 0:
            denom = ta + me
            s["dist"] = tl / denom - ni / denom
        else:
            s["dist"] = None
        # oscore: Ohlson (1980)
        if ta and ta > 0:
            ca, cl = get(d, C_CA), get(d, C_CL)
            size = math.log(max(ta, 1.0))
            tlta = tl / ta
            wcta = (ca - cl) / ta
            clca = (cl / ca) if ca and ca > 0 else 0.0
            oeneg = 1.0 if be < 0 else 0.0
            nita = ni / ta
            futl = (ni / tl) if tl and tl > 0 else 0.0
            intwo = 1.0 if (ni < 0 and (ni_p or 0) < 0) else 0.0
            chin = 0.0
            denom_c = abs(ni) + abs(ni_p or 0)
            if denom_c > 0 and ni_p is not None:
                chin = (ni - ni_p) / denom_c
            s["oscore"] = (-1.32 - 0.407 * size + 6.03 * tlta - 1.43 * wcta
                           + 0.0757 * clca - 1.83 * oeneg - 2.37 * nita
                           + 0.0058 * futl + 0.286 * intwo - 0.522 * chin)
        else:
            s["oscore"] = None
        # gp: gross profit / TA
        s["gp"] = (get(d, C_GP) / ta) if ta and ta > 0 else None
        # roe: NI / BE
        s["roe"] = (ni / be) if be and be > 0 else None
        # bm: BE / ME (June market cap of formation year)
        s["bm"] = (be / me) if (me and me > 0 and be) else None
        # cbop: (GP - ΔNOA) / BE  (balance-sheet accruals)
        if noa_t is not None and noa_p is not None:
            s["cbop"] = ((get(d, C_GP) - (noa_t - noa_p)) / be) if be and be > 0 else None
        else:
            s["cbop"] = None
        # dy: dividends paid / ME
        div = abs(get(d, C_DIV))
        s["dy"] = (div / me) if (me and me > 0 and div > 0) else None
        out[formation] = s
    return out


def monthly_chars(tkr, rets, mcaps, caps, vols, closes):
    """Monthly chars per ticker: size, st_rev, turnover, vol (+ mom annual done later)."""
    out = {}
    ret_series = {m: rets[tkr][m] for m in sorted(rets.get(tkr, {}))}
    keys = sorted(ret_series)
    prev_ret = None
    vol_win = []  # rolling window of the most recent monthly returns
    for k in keys:
        y, m = k
        r = ret_series[k]
        mc = mcaps.get(tkr, {}).get(k)
        cap = caps.get(tkr, {}).get(k)
        vol_tl = vols.get(tkr, {}).get(k)
        last_close = closes.get(tkr, {}).get(k)
        s = {}
        s["size"] = math.log(mc) if mc and mc > 0 else None
        s["st_rev"] = prev_ret  # lagged 1-month return (X is lag-aligned downstream)
        s["turnover"] = None
        if vol_tl and last_close and cap and cap > 0 and last_close > 0:
            shares = vol_tl / last_close          # share volume
            s["turnover"] = shares / cap
        vol_win.append(r)
        if len(vol_win) > 12:
            vol_win.pop(0)
        s["vol"] = float(np.std(vol_win)) if len(vol_win) >= 8 else None
        out[k] = s
        prev_ret = r
    return out


def mom_at_formation(tkr, rets, formation):
    """Annual momentum (TSE convention): cumulative return from April(formation)-12
    to April(formation)-2 (skip most recent month), >=8 of 12 months."""
    target = formation * 12 + 4
    vals = []
    for (y, m), r in rets.get(tkr, {}).items():
        seq = y * 12 + m
        if (target - 12) <= seq <= (target - 2):
            vals.append(r)
    if len(vals) >= 8:
        cum = 1.0
        for v in vals:
            cum *= (1 + v)
        return cum - 1
    return None


def load_closes():
    out = defaultdict(dict)
    for r in read_csv(PROC / "monthly_returns.csv"):
        if r["last_close"]:
            out[r["ticker"]][(int(r["year"]), int(r["month"]))] = float(r["last_close"])
    return out


def build():
    rets = load_returns()
    mcaps = load_mcaps()
    caps = load_capital()
    vols = load_volumes()
    closes = load_closes()
    fin = load_financials()
    print(f"financials: {len(fin)} tickers")

    rows = []
    import json as _json
    _excl_path = DATA / "financial_exclude.json"
    EXCLUDE = set()
    if _excl_path.exists():
        EXCLUDE = set(_json.load(open(_excl_path)))
    for tkr, fyears in sorted(fin.items()):
        if tkr.endswith("GYO"):
            continue
        if tkr in EXCLUDE:
            continue
        ann = annual_signals(tkr, fyears, fin[tkr], mcaps, caps)
        mon = monthly_chars(tkr, rets, mcaps, caps, vols, closes)
        for (y, m), r in rets.get(tkr, {}).items():
            if r is None:
                continue
            formation = y if m >= 7 else y - 1
            a = ann.get(formation, {})
            b = mon.get((y, m), {})
            if not a and not b:
                continue
            row = {"ticker": tkr, "year": y, "month": m, "ret_monthly": r}
            for c in CHARS:
                if c in b:
                    row[c] = b[c]
                elif c == "mom":
                    row[c] = mom_at_formation(tkr, rets, formation)
                else:
                    row[c] = a.get(c)
            rows.append(row)
    print(f"panel rows: {len(rows)}")

    # ---- write RAW panel FIRST (before z-score mutation) -------------------
    with open(DATA / "characteristics_panel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "year", "month", "ret_monthly"] + CHARS)
        w.writeheader()
        w.writerows(rows)

    # ---- winsorize + z-score (per month, cross-sectional) ------------------
    # missing values stay missing (NaN) so the training mask drops them
    from collections import defaultdict as dd
    bym = dd(list)
    for i, row in enumerate(rows):
        bym[(row["year"], row["month"])].append(i)
    out_rows = []
    for (y, m), idxs in sorted(bym.items()):
        for c in CHARS:
            vals = np.array([rows[i][c] if rows[i].get(c) is not None else np.nan
                             for i in idxs], float)
            lo, hi = np.nanpercentile(vals, [1, 99])
            vals = np.clip(vals, lo, hi)
            mu, sd = np.nanmean(vals), np.nanstd(vals)
            if np.isfinite(sd) and sd > 0:
                z = (vals - mu) / sd
            else:
                z = np.full_like(vals, np.nan)
            z[~np.isfinite(vals)] = np.nan
            for j, i in enumerate(idxs):
                rows[i][c] = z[j]
        for i in idxs:
            out_rows.append(rows[i])
    with open(DATA / "characteristics_z.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "year", "month", "ret_monthly"] + CHARS)
        w.writeheader()
        for row in out_rows:
            out = dict(row)
            for c in CHARS:
                v = out.get(c)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    out[c] = ""
            w.writerow(out)
    print(f"saved characteristics_panel.csv (raw) and characteristics_z.csv "
          f"({len(out_rows)} rows)")


if __name__ == "__main__":
    build()
