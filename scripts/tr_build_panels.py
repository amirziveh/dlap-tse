#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tr_build_panels.py — DLAP-TSE Turkey (BIST) Phase 1
=====================================================
Build monthly panels from the raw İş Yatırım daily JSON files.

Inputs : data_tr/raw/{ticker}_prices.json   (from tr_download_prices.py)
Outputs:
  data_tr/processed/monthly_returns.csv   ticker,year,month,ret_monthly,n_days,first_close,last_close
  data_tr/processed/market_cap_monthly.csv ticker,year,month,market_cap (last PD of month)
  data_tr/processed/shares_panel.csv       ticker,year,month,capital (last SERMAYE)
  data_tr/processed/volume_monthly.csv     ticker,year,month,volume_tl (sum HGDG_HACIM)
  data_tr/processed/coverage_summary.csv   per-ticker: first/last date, n_days, n_months
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
RAW = ROOT / "data_tr" / "raw"
OUT = ROOT / "data_tr" / "processed"
OUT.mkdir(exist_ok=True, parents=True)


def parse_date(s):
    """dd-mm-yyyy -> (year, month, day)"""
    d, m, y = s.split("-")
    return int(y), int(m), int(d)


def main():
    files = sorted(RAW.glob("*_prices.json"))
    print(f"{len(files)} raw files")
    monthly = []        # (ticker, y, m, ret, n_days, first_close, last_close)
    mcaps = []          # (ticker, y, m, pd)
    shares = []         # (ticker, y, m, capital)
    volumes = []        # (ticker, y, m, vol_tl)
    coverage = []       # summary rows
    n_empty = 0
    for p in files:
        tkr = p.name.replace("_prices.json", "")
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            coverage.append((tkr, "PARSE_ERROR", 0, 0, 0, 0))
            continue
        rows = d.get("value") or []
        if not rows:
            n_empty += 1
            coverage.append((tkr, "EMPTY", 0, 0, 0, 0))
            continue
        # per-month last values
        bym = defaultdict(list)   # (y,m) -> list of (day, adj_close, pd, capital)
        for r in rows:
            try:
                dt = r.get("HGDG_TARIH")
                if not dt:
                    continue
                y, m, day = parse_date(dt)
                adj = r.get("HG_KAPANIS")
                raw_close = r.get("HGDG_KAPANIS")
                pd = r.get("PD")
                cap = r.get("SERMAYE")
                vol = r.get("HGDG_HACIM")
                bym[(y, m)].append((day, adj, raw_close, pd, cap, vol))
            except Exception:
                continue
        if not bym:
            coverage.append((tkr, "NO_VALID_ROWS", 0, 0, 0, 0))
            continue
        keys = sorted(bym)
        prev_last_adj = None
        for k in keys:
            y, m = k
            recs = sorted(bym[k])
            last_day = recs[-1]
            adj_last = last_day[1]
            raw_last = last_day[2]
            pd_last = last_day[3]
            cap_last = last_day[4]
            vol_sum = sum((r[5] or 0) for r in recs)
            n_days = len(recs)
            if prev_last_adj and adj_last and prev_last_adj:
                ret = adj_last / prev_last_adj - 1.0
            else:
                ret = None
            monthly.append((tkr, y, m, ret, n_days,
                            recs[0][2] if recs[0][2] else None, raw_last))
            if pd_last:
                mcaps.append((tkr, y, m, pd_last))
            if cap_last:
                shares.append((tkr, y, m, cap_last))
            volumes.append((tkr, y, m, vol_sum))
            prev_last_adj = adj_last if adj_last else prev_last_adj
        dates = [parse_date(r.get("HGDG_TARIH")) for r in rows if r.get("HGDG_TARIH")]
        if dates:
            coverage.append((tkr, "OK", len(rows), min(dates), max(dates), len(keys)))

    def wcsv(name, header, rows):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    wcsv("monthly_returns.csv",
         ["ticker", "year", "month", "ret_monthly", "n_days", "first_close", "last_close"],
         [[t, y, m, f"{r:.6f}" if r is not None else "", n, f"{fc:.4f}" if fc else "",
           f"{lc:.4f}" if lc else ""] for t, y, m, r, n, fc, lc in monthly])
    wcsv("market_cap_monthly.csv", ["ticker", "year", "month", "market_cap"], mcaps)
    wcsv("shares_panel.csv", ["ticker", "year", "month", "capital"], shares)
    wcsv("volume_monthly.csv", ["ticker", "year", "month", "volume_tl"], volumes)
    wcsv("coverage_summary.csv",
         ["ticker", "status", "n_days", "first_date", "last_date", "n_months"], coverage)
    ok = sum(1 for c in coverage if c[1] == "OK")
    print(f"OK {ok}, empty {n_empty}, total {len(files)}")
    print(f"monthly rows: {len(monthly)}, market cap rows: {len(mcaps)}, "
          f"shares rows: {len(shares)}")
    # quick stats
    months = sorted({(y, m) for _, y, m, *_ in monthly})
    print(f"month range: {months[0]} .. {months[-1]} ({len(months)} months)")
    rets = [r for _, _, _, r, *_ in monthly if r is not None]
    if rets:
        import statistics
        print(f"valid monthly returns: {len(rets)}; mean {statistics.mean(rets):.4f}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
