#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_build_panels.py — DLAP-TSE Pakistan (PSX) Phase 1
=====================================================
Build monthly panels from the daily mkt_summary text files.

Input : data_pk/raw/{YYYY-MM-DD}.txt   (pipe-delimited from pk_download_prices.py)
Format: DATE|SYMBOL|CODE|NAME|OPEN|HIGH|LOW|CLOSE|VOLUME|AVG|...

Outputs (data_pk/processed/):
  monthly_returns.csv   ticker,year,month,ret_monthly,n_days,first_close,last_close
  volume_monthly.csv    ticker,year,month,volume_shares (sum)
  coverage_summary.csv  per-ticker: first/last date, n_days, n_months
"""
import csv
import os
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
RAW = ROOT / "data_pk" / "raw"
OUT = ROOT / "data_pk" / "processed"
OUT.mkdir(exist_ok=True, parents=True)

MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def parse_date(s):
    """02MAR2015 -> (2015, 3, 2)"""
    m = re.match(r"(\d{2})([A-Z]{3})(\d{4})", s.strip())
    if not m:
        return None
    return int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))


def load_universe():
    """Load /symbols list (local cache preferred); return equity symbol set."""
    import json
    import urllib.request
    cache = ROOT / "data_pk" / "symbols.json"
    data = None
    if cache.exists():
        try:
            data = json.load(open(cache, encoding="utf-8"))
        except Exception:
            data = None
    if data is None:
        try:
            req = urllib.request.Request(
                "https://dps.psx.com.pk/symbols",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            cache.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            print(f"universe fetch failed ({e}); falling back to pattern filter",
                  flush=True)
            return None
    eq = {s["symbol"] for s in data
          if not s.get("isDebt") and not s.get("isETF")}
    print(f"universe: {len(data)} symbols, {len(eq)} equities", flush=True)
    return eq


def is_equity(sym, universe):
    import re
    # always exclude obvious non-equities by pattern (rights, pref, futures,
    # corporate-action variants, TFC/ETF/REIT suffixes)
    if re.search(r"-(AUG|SEP|OCT|JAN|FEB|MAR|APR|MAY|JUN|JUL|NOV|DEC|CAUG|XB|XD)$", sym):
        return False
    if re.search(r"(TFC|ETF|REIT|SC\d?)$", sym):
        return False
    if re.search(r"-R$|R\d*$", sym):
        return False
    if re.search(r"(PS|PREFS?|PRF)$", sym):
        return False
    if universe is not None:
        return sym in universe
    return True


def main():
    universe = load_universe()
    files = sorted(RAW.glob("*.txt"))
    print(f"{len(files)} daily files", flush=True)
    # ticker -> sorted list of (ym, close, volume) using LAST close of month
    monthly = defaultdict(dict)     # (ticker, ym) -> (close, vol_sum, n_days)
    first_last = defaultdict(lambda: [None, None])  # ticker -> [first_date, last_date]
    n_bad = 0
    for p in files:
        d = None
        try:
            dd = date.fromisoformat(p.stem)  # filename is ISO YYYY-MM-DD
            d = (dd.year, dd.month, dd.day)
        except ValueError:
            n_bad += 1
            continue
        ym = (d[0], d[1]) if d else None
        for line in p.open(encoding="utf-8", errors="ignore"):
            parts = line.rstrip("\r\n").split("|")
            if len(parts) < 9:
                continue
            sym = parts[1].strip()
            if not sym or sym == "SYMBOL":
                continue
            if not is_equity(sym, universe):
                continue
            try:
                close = float(parts[7])
                vol = float(parts[8])
            except ValueError:
                continue
            key = (sym, ym)
            if key not in monthly:
                monthly[key] = [close, vol, 1]
            else:
                rec = monthly[key]
                rec[0] = close          # last close of month (files are date-ordered)
                rec[1] += vol
                rec[2] += 1
            if d:
                fl = first_last[sym]
                if fl[0] is None or d < fl[0]:
                    fl[0] = d
                if fl[1] is None or d > fl[1]:
                    fl[1] = d
        n_bad += 0

    # build monthly returns
    rows = []
    by_ticker = defaultdict(list)
    for (sym, ym), (close, vol, nd) in monthly.items():
        by_ticker[sym].append((ym, close, vol, nd))
    for sym in sorted(by_ticker):
        series = sorted(by_ticker[sym])
        for i, (ym, close, vol, nd) in enumerate(series):
            ret = None
            if i > 0 and series[i - 1][1] and close:
                ret = close / series[i - 1][1] - 1.0
            rows.append((sym, ym[0], ym[1], ret, nd, series[i - 1][1] if i else None, close, vol))

    with open(OUT / "monthly_returns.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "year", "month", "ret_monthly", "n_days",
                    "first_close", "last_close"])
        for r in rows:
            w.writerow(r)
    with open(OUT / "volume_monthly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "year", "month", "volume_shares"])
        for r in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
            w.writerow([r[0], r[1], r[2], r[7]])
    with open(OUT / "coverage_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "first_date", "last_date", "n_days", "n_months"])
        for sym in sorted(first_last):
            fl = first_last[sym]
            w.writerow([sym, fl[0], fl[1], sum(1 for r in rows if r[0] == sym),
                        len(by_ticker[sym])])

    n_syms = len(by_ticker)
    yrs = sorted(set(r[1] for r in rows))
    valid = sum(1 for r in rows if r[3] is not None)
    print(f"rows: {len(rows)}, tickers: {n_syms}, years: {yrs[0]}-{yrs[-1]}", flush=True)
    print(f"valid monthly returns: {valid}", flush=True)
    print(f"saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
