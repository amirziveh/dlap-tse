#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_macro_panel.py — DLAP-TSE Phase 1
=========================================
Builds the monthly macro-state panel for the CPZ (2024) SDF replication.

Series (all monthly, 2001-01 .. 2026-12):
  cbirate      CBI policy rate (annual %)          — local fama-five risk_free_rate.csv
  cpi          CPI (2010=100)                      — World Bank FP.CPI.TOTL (annual → within-year constant)
  usd_official Official USD/IRR                    — World Bank PA.NUS.FCRF (annual → within-year constant)
  brent        Brent crude USD/bbl                 — FRED DCOILBRENTEU (daily → monthly mean)
  gold_coin    Emami gold coin (IRR)               — tgju.org profile/sekee (daily last → monthly last)
  usd_market   Market USD/IRR                      — tgju.org profile/price_dollar_rl (daily last → monthly last)

Coverage limits (documented in PHASE0_FINDINGS.md):
  gold_coin from 2010-04, usd_market from 2011-11, usd_official ends 2023 (WB lag),
  cpi ends 2025 (WB lag). NaN = missing.

Outputs:
  data/macro_panel.csv          — combined monthly panel
  data/macro_raw/*.csv          — raw daily/annual intermediates (reproducibility)
"""
import csv
import os
import datetime as dt
import json
import re
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path

FAMA = Path(os.environ.get("FAMA_ROOT", str(Path.home() / "research/fama-five/data")))
OUT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse"))) / "data"
RAW = OUT / "macro_raw"
RAW.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (compatible; DLAP-TSE/1.0)"}


def months_range(y0=2001, m0=1, y1=2026, m1=12):
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


MONTHS = months_range()


def fetch_wb(indicator):
    """World Bank annual series -> {year: value}"""
    url = f"https://api.worldbank.org/v2/country/IRN/indicator/{indicator}?format=json&per_page=1000"
    req = urllib.request.Request(url, headers=UA)
    d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8-sig"))
    out = {}
    for r in d[1]:
        if r.get("value") is not None:
            out[int(r["date"])] = float(r["value"])
    return out


def fetch_fred_daily(series_id):
    """FRED csv (no key needed) -> {date_str: value}. Uses curl (urllib times out on this link)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = None
    for attempt in range(3):
        r = subprocess.run(["curl", "-sL", "--max-time", "40", "-A", UA["User-Agent"], url],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout:
            text = r.stdout
            break
    if text is None:
        return None  # caller falls back to Yahoo
    out = {}
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        date, val = line.split(",")
        try:
            out[date] = float(val)
        except ValueError:
            continue
    return out


def fetch_yahoo_monthly(symbol):
    """Yahoo chart API -> {month 'YYYY-MM': month-end close} (no key needed)."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?range=max&interval=1mo")
    req = urllib.request.Request(url, headers=UA)
    d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
    r = d["chart"]["result"][0]
    out = {}
    for t, c in zip(r["timestamp"], r["indicators"]["quote"][0]["close"]):
        if c is None:
            continue
        m = dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m")
        out[m] = float(c)
    return out


def fetch_tgju_daily(key):
    """tgju charts page -> {date_str: last price of day}"""
    url = f"https://www.tgju.org/profile/{key}/charts"
    req = urllib.request.Request(url, headers=UA)
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    best = None
    best_span = -1
    i = html.find("chartData: ")
    while i >= 0:
        s = i + len("chartData: ")
        depth, j = 0, s
        while j < len(html):
            c = html[j]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        raw = html[s:j + 1]
        try:
            data = json.loads(raw)
            if (isinstance(data, list) and data and isinstance(data[0], list)
                    and len(data[0]) >= 2 and len(data) > 100):
                span = max(p[0] for p in data) - min(p[0] for p in data)
                if span > best_span:
                    best, best_span = data, span
        except Exception:
            pass
        i = html.find("chartData: ", j)
    if best is None:
        raise RuntimeError(f"no chart data for {key}")
    daily = {}
    for pt in best:
        ts, price = int(pt[0]), float(pt[1])
        d = dt.datetime.fromtimestamp(ts / 1000, dt.UTC).strftime("%Y-%m-%d")
        daily[d] = price  # last of day wins (points are chronological)
    return daily


def daily_to_monthly_last(daily):
    out = {}
    for d, v in daily.items():
        out[d[:7]] = v  # last of month wins (chronological order)
    return out


def daily_to_monthly_mean(daily):
    agg = defaultdict(list)
    for d, v in daily.items():
        agg[d[:7]].append(v)
    return {m: sum(vs) / len(vs) for m, vs in agg.items()}


def annual_to_monthly(annual):
    """{year: value} -> {month 'YYYY-MM': value} constant within calendar year"""
    out = {}
    for y, v in annual.items():
        for m in range(1, 13):
            out[f"{y}-{m:02d}"] = v
    return out


def save_raw(name, mapping, header):
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in sorted(mapping):
            w.writerow([k, mapping[k]])


def main():
    panel = {m: {} for m in MONTHS}

    # 1. CBI policy rate (local)
    cbi = {}
    with open(FAMA / "risk_free_rate.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            month = row["month"][:7]
            if month in panel:
                cbi[month] = float(row["annual_rate_pct"])
    save_raw("cbirate.csv", cbi, ["month", "annual_rate_pct"])
    for m, v in cbi.items():
        panel[m]["cbirate"] = v
    print(f"  cbirate      {len(cbi)} months  {min(cbi)}..{max(cbi)}")

    # 2. CPI (World Bank, annual -> monthly constant)
    cpi_annual = fetch_wb("FP.CPI.TOTL")
    cpi = annual_to_monthly(cpi_annual)
    save_raw("wb_cpi_annual.csv", cpi_annual, ["year", "cpi"])
    for m, v in cpi.items():
        if m in panel:
            panel[m]["cpi"] = v
    print(f"  cpi          annual {min(cpi_annual)}..{max(cpi_annual)}")

    # 3. USD/IRR official (World Bank)
    usd_annual = fetch_wb("PA.NUS.FCRF")
    usd = annual_to_monthly(usd_annual)
    save_raw("wb_usd_official_annual.csv", usd_annual, ["year", "usd_irr"])
    for m, v in usd.items():
        if m in panel:
            panel[m]["usd_official"] = v
    print(f"  usd_official annual {min(usd_annual)}..{max(usd_annual)}")

    # 4. Brent (FRED daily -> monthly mean; fallback Yahoo BZ=F month-end close)
    brent_daily = fetch_fred_daily("DCOILBRENTEU")
    if brent_daily:
        brent = daily_to_monthly_mean(brent_daily)
        save_raw("brent_daily.csv", brent_daily, ["date", "usd"])
        print(f"  brent        {len(brent)} months (FRED)  {min(brent)}..{max(brent)}")
    else:
        brent = fetch_yahoo_monthly("BZ=F")
        save_raw("brent_monthly_yahoo.csv", brent, ["month", "usd"])
        print(f"  brent        {len(brent)} months (Yahoo BZ=F)  {min(brent)}..{max(brent)}")
    for m, v in brent.items():
        if m in panel:
            panel[m]["brent"] = v

    # 5. Gold coin Emami (tgju)
    gold_daily = fetch_tgju_daily("sekee")
    gold = daily_to_monthly_last(gold_daily)
    save_raw("gold_coin_daily.csv", gold_daily, ["date", "irr"])
    for m, v in gold.items():
        if m in panel:
            panel[m]["gold_coin"] = v
    print(f"  gold_coin    {len(gold)} months  {min(gold)}..{max(gold)}")

    # 6. USD market (tgju)
    usd_daily = fetch_tgju_daily("price_dollar_rl")
    usdm = daily_to_monthly_last(usd_daily)
    save_raw("usd_market_daily.csv", usd_daily, ["date", "irr"])
    for m, v in usdm.items():
        if m in panel:
            panel[m]["usd_market"] = v
    print(f"  usd_market   {len(usdm)} months  {min(usdm)}..{max(usdm)}")

    # write panel
    fields = ["month", "cbirate", "cpi", "usd_official", "brent", "gold_coin", "usd_market"]
    with open(OUT / "macro_panel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in MONTHS:
            row = {"month": m}
            row.update(panel[m])
            w.writerow(row)
    print(f"\nSaved {len(MONTHS)} months -> {OUT / 'macro_panel.csv'}")


if __name__ == "__main__":
    main()
