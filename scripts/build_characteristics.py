#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_characteristics.py — DLAP-TSE Phase 1
============================================
Builds the 20-characteristic monthly panel for the CPZ (2024) SDF replication.

Outputs (in dlap-tse/data/):
  characteristics_panel.csv  — raw panel: ticker, year, month, ret_monthly, 20 chars
  characteristics_z.csv      — same, but each char winsorized (1%/99%) and
                               cross-sectionally z-scored within each month

Conventions (inherited from fama-five):
  - Annual characteristics are joined to returns via formation_year:
      month (y, m) with m >= 7  -> formation year y
      month (y, m) with m <  7  -> formation year y-1
    (i.e. characteristics known ~July of formation year, held 12 months)
  - Financial firms excluded (sector contains بانک / بیمه / موسسات اعتباری / نهادهای مالی)
  - Tickers are Persian; all files UTF-8
"""
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMA = Path(os.environ.get("FAMA_ROOT", str(Path.home() / "research/fama-five/data")))
OUT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse"))) / "data"
OUT.mkdir(parents=True, exist_ok=True)

# ── 20 characteristics: (name, kind, source) ────────────────────────────
# kind: 'annual' = formation-year aligned; 'monthly' = computed at month t
CHARS = [
    # monthly
    "size",        # ln(market cap)
    "st_rev",      # 1-month reversal (lagged return)
    "turnover",    # monthly traded volume / shares outstanding
    "vol",         # trailing 12-month realized vol of monthly returns
    # annual (anomaly_signals.csv)
    "bm", "mom", "roe", "ag", "ac", "noa", "nsi", "gp",
    "cei", "ita", "ig", "dist", "oscore",
    # annual (cbop_panel.csv)
    "investment",  # I/A (q-factor style)
    "cbop",        # cash-based operating profitability
    # annual (dps_panel.csv, announcement-aligned)
    "dy",          # dividend yield: last announced dps / month-end price
]
# order in the npz/CSV
ANNUAL_FROM_SIGNALS = ["bm", "mom", "roe", "ag", "ac", "noa", "nsi", "gp",
                       "cei", "ita", "ig", "dist", "oscore"]
ANNUAL_FROM_CBOP = ["investment", "cbop"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_financial_tickers():
    fin = set()
    for row in read_csv(FAMA / "stock_universe.csv"):
        sector = row.get("sector_name", "") or ""
        if any(w in sector for w in ["بانک", "بیمه", "موسسات اعتباری", "نهادهای مالی"]):
            fin.add(row["ticker"])
    return fin


def load_signals():
    """formation_year -> ticker -> {char: value}"""
    out = defaultdict(dict)
    for row in read_csv(FAMA / "mispricing" / "anomaly_signals.csv"):
        fy = int(row["formation_year"]) if row["formation_year"] else None
        if not fy:
            continue
        d = {}
        for c in ANNUAL_FROM_SIGNALS:
            col = "bm_proxy" if c == "bm" else c
            v = row.get(col, "")
            d[c] = float(v) if v not in ("", "None") else math.nan
        out[fy][row["ticker"]] = d
    return out


def load_cbop():
    """gregorian_year -> ticker -> {investment, cbop}  (formation = greg+1)"""
    out = defaultdict(dict)
    for row in read_csv(FAMA / "processed" / "cbop_panel.csv"):
        gy = row.get("gregorian_year", "")
        if not gy:
            continue
        gy = int(gy)
        d = {}
        for c in ["investment", "cbop_bs"]:
            v = row.get(c, "")
            d[c] = float(v) if v not in ("", "None") else math.nan
        out[gy + 1][row["ticker"]] = d  # formation_year = gregorian + 1
    return out


def load_dps():
    """ticker -> sorted [(announcement_date_str, dps)]"""
    out = defaultdict(list)
    for row in read_csv(FAMA / "processed" / "dps_panel.csv"):
        ad = row.get("announcement_date", "")
        dps = row.get("dps", "")
        if not ad or not dps:
            continue
        try:
            out[row["ticker"]].append((ad, float(dps)))
        except ValueError:
            continue
    for t in out:
        out[t].sort()
    return out


def load_book_equity():
    """ticker -> {persian fiscal_year: book_equity}  (ff5_accounting.csv)"""
    out = defaultdict(dict)
    for row in read_csv(FAMA / "processed" / "ff5_accounting.csv"):
        py = row.get("fiscal_year", "")
        be = row.get("book_equity", "")
        if not py or be in ("", "None"):
            continue
        try:
            out[row["ticker"]][int(py)] = float(be)
        except ValueError:
            continue
    return out


def month_end(y, m):
    if m == 12:
        return f"{y + 1}-01-01"
    return f"{y}-{m + 1:02d}-01"


def load_monthly_prices():
    """ticker -> {(y,m): (sum_volume, shares_at_end)}"""
    out = defaultdict(lambda: defaultdict(lambda: [0.0, None]))
    for row in read_csv(FAMA / "processed" / "prices_tsetmc_adjusted.csv"):
        date = row["date"]
        try:
            y, m = int(date[:4]), int(date[5:7])
        except ValueError:
            continue
        vol = row.get("volume", "")
        sh = row.get("shares_at_date", "")
        rec = out[row["ticker"]][(y, m)]
        if vol not in ("", "None"):
            rec[0] += float(vol)
        if sh not in ("", "None") and float(sh) > 0:
            rec[1] = float(sh)
    return {t: {k: tuple(v) for k, v in mdict.items()} for t, mdict in out.items()}


def main():
    print("Loading inputs ...")
    fin = load_financial_tickers()
    signals = load_signals()
    cbop = load_cbop()
    dps = load_dps()
    prices = load_monthly_prices()
    book_equity = load_book_equity()

    # returns + mcap in one pass
    returns = defaultdict(dict)   # ticker -> {(y,m): ret}
    mcap = defaultdict(dict)      # ticker -> {(y,m): cap}
    with open(FAMA / "processed" / "monthly_returns.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = row["ticker"]
            try:
                y, m = int(row["year"]), int(row["month"])
                r = float(row["ret_monthly"])
            except (ValueError, KeyError):
                continue
            returns[t][(y, m)] = r
    with open(FAMA / "processed" / "market_cap_monthly.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = row["ticker"]
            try:
                y, m = int(row["year"]), int(row["month"])
                c = float(row["market_cap"])
            except (ValueError, KeyError):
                continue
            mcap[t][(y, m)] = c

    # universe of months: all (y,m) present in returns
    all_months = sorted({k for d in returns.values() for k in d})
    print(f"  {len(all_months)} stock-months universe, {len(returns)} tickers, "
          f"{len(fin)} financial firms excluded")

    rows = []
    n_ann_joined = 0
    for ticker, rmap in returns.items():
        if ticker in fin:
            continue
        # precompute vol (trailing 12m) once per ticker
        rets_by_month = rmap
        months_sorted = sorted(rmap.keys())
        for i, (y, m) in enumerate(months_sorted):
            r = rmap[(y, m)]
            if not math.isfinite(r):
                continue
            # --- monthly chars ---
            cap = mcap.get(ticker, {}).get((y, m))
            size = math.log(cap) if cap and cap > 0 else math.nan
            # st_rev: previous month return
            if m == 1:
                prev = rmap.get((y - 1, 12))
            else:
                prev = rmap.get((y, m - 1))
            st_rev = prev if prev is not None and math.isfinite(prev) else math.nan
            # volatility: trailing 12m std
            window = [rmap[(yy, mm)] for (yy, mm) in months_sorted[max(0, i - 11):i]
                      if math.isfinite(rmap[(yy, mm)])]
            vol = float(np.std(window)) if len(window) >= 6 else math.nan
            # turnover
            pv = prices.get(ticker, {}).get((y, m), (0.0, None))
            turnover = (pv[0] / pv[1]) if (pv[1] and pv[1] > 0) else math.nan
            # dividend yield: last announced dps before month end / close price
            dy = math.nan
            if ticker in dps:
                cutoff = month_end(y, m)
                last = None
                for ad, dval in dps[ticker]:
                    if ad < cutoff:
                        last = dval
                    else:
                        break
                if last is not None and cap and cap > 0:
                    # close price per share = mcap / shares? use mcap price proxy:
                    # dps is per-share; price = mcap / shares_outstanding
                    sh = prices.get(ticker, {}).get((y, m), (0.0, None))[1]
                    if sh and sh > 0:
                        dy = last / (cap / sh)
            # --- annual chars (formation-year aligned) ---
            fy = y if m >= 7 else y - 1
            row = {"ticker": ticker, "year": y, "month": m,
                   "ret_monthly": r, "size": size, "st_rev": st_rev,
                   "turnover": turnover, "vol": vol, "dy": dy}
            ok = False
            sig = signals.get(fy, {}).get(ticker)
            if sig:
                for c in ANNUAL_FROM_SIGNALS:
                    if c == "bm":
                        continue  # bm now from BE/ME below (not TE/TA proxy)
                    row[c] = sig[c]
                ok = True
            # --- book-to-market: BE/ME (FF definition) ---
            # BE = book equity of Persian fiscal year (fy - 622), announced
            # before July of fy (verified mapping: gregorian = shamsi + 621,
            # formation = gregorian + 1, 9779/9779 rows consistent).
            # Units: ff5_accounting book_equity is in MILLION Rials (Rahavard
            # convention); market_cap_monthly is in Rials -> scale BE by 1e6.
            py = fy - 622
            be = book_equity.get(ticker, {}).get(py)
            if be is not None and math.isfinite(be) and cap and cap > 0:
                row["bm"] = be * 1e6 / cap
            cbp = cbop.get(fy, {}).get(ticker)
            if cbp:
                row["investment"] = cbp["investment"]
                row["cbop"] = cbp["cbop_bs"]
                ok = True
            if ok:
                n_ann_joined += 1
            # fill missing annual with nan
            for c in CHARS:
                row.setdefault(c, math.nan)
            rows.append(row)
        if len(rows) % 20000 < 2:
            pass

    print(f"  {len(rows)} stock-month rows with a return")

    # ── winsorize returns per month at 1%/99% ─────────────────────────
    # TSE capital-increase months produce adjustment artifacts (returns up to
    # +1700%); standard GKX/CPZ practice is monthly cross-sectional
    # winsorization. Applied to ret_monthly in BOTH output CSVs (documented).
    by_month_ret = defaultdict(list)
    for r in rows:
        by_month_ret[(r["year"], r["month"])].append(r)
    n_clipped = 0
    for (y, m), group in by_month_ret.items():
        vals = np.array([r["ret_monthly"] for r in group], dtype=float)
        good = np.isfinite(vals)
        if good.sum() < 10:
            continue
        lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
        for r, x in zip(group, vals):
            if np.isfinite(x) and (x < lo or x > hi):
                r["ret_monthly"] = float(np.clip(x, lo, hi))
                n_clipped += 1
    print(f"  winsorized {n_clipped} return outliers ({n_clipped / len(rows) * 100:.2f}%)")

    # ── write raw panel ────────────────────────────────────────────────
    fieldnames = ["ticker", "year", "month", "ret_monthly"] + CHARS

    def sanitize(r):
        out = dict(r)
        for c in fieldnames:
            v = out.get(c)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                out[c] = ""
        return out

    with open(OUT / "characteristics_panel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(sanitize(r))

    # ── z-scored panel: winsorize 1%/99% then z-score per month ────────
    by_month = defaultdict(list)
    for r in rows:
        by_month[(r["year"], r["month"])].append(r)
    zrows = []
    for (y, m), group in sorted(by_month.items()):
        for c in CHARS:
            vals = np.array([r[c] for r in group], dtype=float)
            good = ~np.isnan(vals)
            if good.sum() < 5:
                for r in group:
                    r[c] = math.nan
                continue
            lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
            v = np.clip(vals, lo, hi)
            mu, sd = v[good].mean(), v[good].std()
            if sd == 0 or not math.isfinite(sd):
                for r in group:
                    r[c] = math.nan
                continue
            for r, x in zip(group, v):
                r[c] = (x - mu) / sd if math.isfinite(x) else math.nan
            # guard: tiny-sd months can produce extreme z; clip to [-10, 10]
            for r in group:
                if isinstance(r[c], (float, np.float64)) and math.isfinite(r[c]):
                    r[c] = max(-10.0, min(10.0, r[c]))
        zrows.extend(group)
    with open(OUT / "characteristics_z.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in zrows:
            w.writerow(sanitize(r))

    # ── report ─────────────────────────────────────────────────────────
    print("\nCoverage per characteristic (% of rows non-missing):")
    for c in CHARS:
        n = sum(1 for r in rows if r.get(c) is not None and math.isfinite(r[c]))
        print(f"  {c:<12} {n / len(rows) * 100:6.1f}%")
    print(f"\nSaved {len(rows):,} rows -> {OUT / 'characteristics_panel.csv'}")
    print(f"Saved z-scored       -> {OUT / 'characteristics_z.csv'}")


if __name__ == "__main__":
    main()
