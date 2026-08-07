#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_build_characteristics.py — DLAP-TSE Pakistan (PSX) Phase 2
==============================================================
Builds the 20-characteristic monthly panel for Pakistan, mirroring the TSE
and Turkey constructions (scripts/build_characteristics.py,
scripts/tr_build_characteristics.py) with PSX data.

Outputs (data_pk/):
  characteristics_panel.csv — raw panel: ticker, year, month, ret, 20 chars
  characteristics_z.csv     — winsorized (1%/99%) + cross-sectionally z-scored

Conventions (Pakistan-specific, documented in notes/3country_data_status.md):
  - PK fiscal year ends June 30; annual reports published ~Sep-Oct
    -> formation year = calendar year y if month m >= 10, else y-1
       (financials for FY f apply from October of year f)
  - BE = Eq (no deferred-tax / minority data in VLM extraction)
  - NOA proxy = (TA - Cash) - (TL - CL)  [ST/LT debt not extracted -> documented]
  - nsi = d(shares)/shares  (registered-capital proxy, par value constant)
  - ig  = missing (no long-term-investment data) -> dropped from PK char set
  - bm/dist ME measured at (formation, 10) or (formation, 11) [report month]
  - momentum: TSE convention target = formation*12 + 4 (Apr), window -12..-2
  - turnover = share volume / shares outstanding (annual stepped)
"""
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

PK = Path("/home/ubuntu/research/dlap-tse/data_pk")
PROC = PK / "processed"

CHARS = ["size", "st_rev", "turnover", "vol", "bm", "mom", "roe", "ag", "ac",
         "noa", "nsi", "gp", "cei", "ita", "dist", "oscore",
         "investment", "cbop", "dy"]  # 19 chars: ig dropped (no LT-inv data)

FIN_FILE = PK / "financials_annual_no_financials.csv"


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_returns():
    out = defaultdict(dict)
    for r in read_csv(PROC / "monthly_returns.csv"):
        if r["ret_monthly"]:
            out[r["ticker"]][(int(r["year"]), int(r["month"]))] = float(r["ret_monthly"])
    return out


def load_mcaps():
    out = defaultdict(dict)
    for r in read_csv(PK / "market_cap_monthly.csv"):
        out[r["ticker"]][(int(r["year"]), int(r["month"]))] = float(r["mktcap"])
    return out


def load_shares():
    out = defaultdict(dict)  # ticker -> year -> shares
    for r in read_csv(PK / "shares_annual.csv"):
        out[r["ticker"]][int(r["year"])] = int(r["shares"])
    return out


def load_volumes():
    out = defaultdict(dict)
    for r in read_csv(PROC / "volume_monthly.csv"):
        out[r["ticker"]][(int(r["year"]), int(r["month"]))] = float(r["volume_shares"])
    return out


def load_financials():
    """symbol -> fiscal_year -> {field: value} (already unit-normalized)."""
    out = defaultdict(dict)
    for r in read_csv(FIN_FILE):
        sym, year = r["symbol"], int(r["year"])
        rec = {}
        for f in ["TA", "TL", "Eq", "CA", "CL", "Sales", "COGS", "GP",
                  "PAT", "Cash", "Inv", "PPE", "dividends"]:
            v = fnum(r.get(f))
            if v is not None:
                rec[f] = v
        if rec:
            out[sym][year] = rec
    return out


def annual_signals(tkr, years, fin, mcaps, shares):
    """Formation-year signals for one ticker."""
    out = {}  # formation_year -> {char: value}
    ys = sorted(years)
    for i, fy in enumerate(ys):
        d = fin[fy]
        prev = fin.get(fy - 1, {})
        if not d:
            continue
        formation = fy  # PK: FY f data applies from October of year f
        ta, ta_p = d.get("TA"), prev.get("TA")
        tl, tl_p = d.get("TL"), prev.get("TL")
        be, be_p = d.get("Eq"), prev.get("Eq")
        ni, ni_p = d.get("PAT"), prev.get("PAT")
        ca, cl = d.get("CA"), d.get("CL")
        cash, inv = d.get("Cash"), d.get("Inv")
        s = {}
        # nsi: YoY change in shares outstanding (registered-capital proxy)
        sh_t = (shares.get(tkr, {}).get(fy) or 0)
        sh_p = (shares.get(tkr, {}).get(fy - 1) or 0)
        s["nsi"] = (sh_t - sh_p) / sh_p if sh_p > 0 else None
        # cei: (BE_t - BE_{t-1}) / TA_{t-1}
        s["cei"] = ((be - be_p) / ta_p) if (be is not None and be_p is not None and ta_p and ta_p > 0) else None
        # NOA proxy = (TA - Cash) - (TL - CL); ac = dNOA/TA_{t-1}; noa = NOA_t/TA_t
        def noa_of(rec):
            if not rec.get("TA"):
                return None
            op_a = rec["TA"] - (rec.get("Cash") or 0.0)
            op_l = (rec.get("TL") or 0.0) - (rec.get("CL") or 0.0)
            return op_a - op_l
        noa_t = noa_of(d)
        noa_p = noa_of(prev)
        s["ac"] = ((noa_t - noa_p) / ta_p) if (noa_t is not None and noa_p is not None and ta_p and ta_p > 0) else None
        s["noa"] = (noa_t / ta) if (noa_t is not None and ta and ta > 0) else None
        # ag / investment: dTA / TA_{t-1}
        s["ag"] = (ta - ta_p) / ta_p if (ta and ta_p and ta_p > 0) else None
        s["investment"] = s["ag"]
        # ita: dPPE / TA_{t-1}
        ppe, ppe_p = d.get("PPE"), prev.get("PPE")
        s["ita"] = (ppe - ppe_p) / ta_p if (ppe and ppe_p and ta_p and ta_p > 0) else None
        # dist: TLMTA - NIMTA (Campbell-style)
        me = mcaps.get(tkr, {}).get((formation, 10)) or mcaps.get(tkr, {}).get((formation, 11))
        if me and me > 0 and ta and ta > 0 and tl is not None and ni is not None:
            denom = ta + me
            s["dist"] = tl / denom - ni / denom
        else:
            s["dist"] = None
        # oscore: Ohlson (1980) — size = ln(TA)
        if ta and ta > 0 and tl is not None and ca is not None and cl is not None and ni is not None:
            size = math.log(max(ta, 1.0))
            tlta = tl / ta
            wcta = (ca - cl) / ta
            clca = (cl / ca) if ca and ca > 0 else 0.0
            oeneg = 1.0 if (be or 0) < 0 else 0.0
            nita = ni / ta
            futl = (ni / tl) if tl and tl > 0 else 0.0
            intwo = 1.0 if (ni is not None and ni < 0 and (ni_p or 0) < 0) else 0.0
            chin = 0.0
            denom_c = abs(ni or 0) + abs(ni_p or 0)
            if denom_c > 0 and ni_p is not None and ni is not None:
                chin = (ni - ni_p) / denom_c
            s["oscore"] = (-1.32 - 0.407 * size + 6.03 * tlta - 1.43 * wcta
                           + 0.0757 * clca - 1.83 * oeneg - 2.37 * nita
                           + 0.0058 * futl + 0.286 * intwo - 0.522 * chin)
        else:
            s["oscore"] = None
        # gp: GP / TA
        s["gp"] = (d.get("GP") / ta) if (d.get("GP") is not None and ta and ta > 0) else None
        # roe: NI / BE
        s["roe"] = (ni / be) if (ni is not None and be and be > 0) else None
        # bm: BE / ME
        s["bm"] = (be / me) if (me and me > 0 and be) else None
        # cbop: (GP - dNOA) / BE
        if noa_t is not None and noa_p is not None:
            s["cbop"] = ((d.get("GP", 0.0) - (noa_t - noa_p)) / be) if (be and be > 0) else None
        else:
            s["cbop"] = None
        # dy: dividends / ME
        div = abs(d.get("dividends") or 0.0)
        s["dy"] = (div / me) if (me and me > 0 and div > 0) else None
        out[formation] = s
    return out


def monthly_chars(tkr, rets, mcaps, shares, vols):
    """Monthly chars: size, st_rev, turnover, vol."""
    out = {}
    ret_series = {m: rets[tkr][m] for m in sorted(rets.get(tkr, {}))}
    keys = sorted(ret_series)
    prev_ret = None
    vol_win = []
    for k in keys:
        y, m = k
        r = ret_series[k]
        mc = mcaps.get(tkr, {}).get(k)
        vol_sh = vols.get(tkr, {}).get(k)
        sh = shares.get(tkr, {}).get(y)
        s = {}
        s["size"] = math.log(mc) if mc and mc > 0 else None
        s["st_rev"] = prev_ret
        s["turnover"] = None
        if vol_sh and sh and sh > 0:
            s["turnover"] = vol_sh / sh
        vol_win.append(r)
        if len(vol_win) > 12:
            vol_win.pop(0)
        s["vol"] = float(np.std(vol_win)) if len(vol_win) >= 8 else None
        out[k] = s
        prev_ret = r
    return out


def mom_at_formation(tkr, rets, formation):
    """TSE convention: cum return Apr(formation)-12 .. Apr(formation)-2, >=8/12."""
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


def build():
    rets = load_returns()
    mcaps = load_mcaps()
    shares = load_shares()
    vols = load_volumes()
    fin = load_financials()
    print(f"financials: {len(fin)} tickers | returns: {len(rets)} | mcap: {len(mcaps)}")

    rows = []
    for tkr, fyears in sorted(fin.items()):
        if tkr not in rets:
            continue
        ann = annual_signals(tkr, fyears, fin[tkr], mcaps, shares)
        mon = monthly_chars(tkr, rets, mcaps, shares, vols)
        for (y, m), r in rets.get(tkr, {}).items():
            if r is None:
                continue
            formation = y if m >= 10 else y - 1
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

    with open(PK / "characteristics_panel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "year", "month", "ret_monthly"] + CHARS)
        w.writeheader()
        w.writerows(rows)

    # winsorize + z-score per month
    bym = defaultdict(list)
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
    with open(PK / "characteristics_z.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "year", "month", "ret_monthly"] + CHARS)
        w.writeheader()
        for row in out_rows:
            out = dict(row)
            for c in CHARS:
                v = out.get(c)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    out[c] = ""
            w.writerow(out)
    print(f"saved characteristics_panel.csv + characteristics_z.csv ({len(out_rows)} rows)")


if __name__ == "__main__":
    build()
