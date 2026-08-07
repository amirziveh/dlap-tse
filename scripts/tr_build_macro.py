#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tr_build_macro.py — DLAP-TSE Turkey (BIST) Phase 3
=====================================================
Macro panel + risk-free rate for Turkey, from FRED (no API key).

Outputs (data_tr/):
  macro_panel.csv  month, cbrate, cpi, usd_try, tbill3m, brent
  risk_free_rate.csv  month, monthly_rate_pct  (3M t-bill / 12, decimal)

FRED series (all monthly):
  IRSTCI01TRM156N  CBRT policy rate (%)
  CPALTT01TRM657N  CPI inflation, monthly (%)
  DEXTAUS          USD/TRY
  INTGSTTRM193N    3-month T-bill rate (%)
  DCOILBRENTEU     Brent USD/bbl
"""
import csv
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("DLAP_ROOT", str(Path.home() / "research/dlap-tse")))
OUT = ROOT / "data_tr"
OUT.mkdir(exist_ok=True)

SERIES = {
    "cbrate": "IRSTCI01TRM156N",
    "cpi": "CPALTT01TRM657N",
    "usd_try": "DEXTAUS",
    "tbill3m": "INTGSTTRM193N",
    "brent": "DCOILBRENTEU",
}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def fetch_fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    # NOTE: a browser UA triggers bot detection (empty body); plain curl UA works
    for attempt in range(3):
        r = subprocess.run(["curl", "-sL", "--max-time", "40", url],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout:
            out = {}
            for line in r.stdout.splitlines()[1:]:
                if not line.strip():
                    continue
                date, val = line.split(",")
                try:
                    out[date[:7]] = float(val)
                except ValueError:
                    continue
            return out
    return {}


def months_range(y0=2008, m0=1, y1=2026, m1=12):
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def main():
    data = {k: fetch_fred(v) for k, v in SERIES.items()}
    for k, d in data.items():
        print(f"{k}: {len(d)} months ({min(d) if d else '-'} .. {max(d) if d else '-'})")
    months = months_range()
    rows = []
    for m in months:
        row = {"month": m}
        for k in SERIES:
            row[k] = data[k].get(m, "")
        rows.append(row)
    with open(OUT / "macro_panel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["month"] + list(SERIES))
        w.writeheader()
        w.writerows(rows)
    # risk-free: CBRT policy rate / 12 -> monthly decimal (3M t-bill ends 2008)
    with open(OUT / "risk_free_rate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month", "monthly_rate_pct"])
        for m in months:
            v = data["cbrate"].get(m)
            if v is not None:
                w.writerow([m, f"{v / 12 / 100:.8f}"])
    print("saved macro_panel.csv and risk_free_rate.csv (RF = CBRT policy/12)")


if __name__ == "__main__":
    main()
