#!/usr/bin/env python3
"""Build PK shares-outstanding table + market cap series (no PDFs).

Sources (all already cached locally):
  - dps per-year financials (2022-2025): shares = PAT_after_tax / EPS
  - dps current profile: eq.shares (2026+)
  - backfill 2013-2021 with the 2022 share count (documented approximation)

Outputs:
  data_pk/shares_annual.csv       ticker, year, shares (fiscal-year, stepped)
  data_pk/market_cap_monthly.csv  ticker, year, month, date, close, shares, mktcap
"""
import json, csv, glob, os
from pathlib import Path

ROOT = Path('/home/ubuntu/research/dlap-tse')
PK = ROOT / 'data_pk'
dps = json.load(open(PK / 'unit_norm' / 'pk_dps_financials.json'))

# 1) per-year shares from PAT/EPS (2022-2025)
shares = {}   # {ticker: {year: shares}}
for sym, d in dps.items():
    fin = d.get('fin', {})
    for yr in sorted(fin, key=int):
        pat = fin[yr].get('Profit after Taxation')
        eps = fin[yr].get('EPS')
        if pat and eps and eps > 0 and pat > 0:
            shares.setdefault(sym, {})[int(yr)] = round(pat * 1000.0 / eps)  # PAT in thousands
    eq = d.get('eq', {})
    if eq.get('shares'):
        shares.setdefault(sym, {})[2026] = int(eq['shares'])

# 2) backfill 2013-2021 with earliest known (2022) count
rows = []
for sym, years in sorted(shares.items()):
    known = sorted(years)
    anchor = years.get(2022) or years.get(2026) or years.get(2023)
    if anchor is None:
        continue
    for y in range(2013, 2027):
        if y in years:
            v = years[y]
        else:
            v = anchor if y < 2022 else (years.get(2026) or anchor)
        rows.append({'ticker': sym, 'year': y, 'shares': v})

with open(PK / 'shares_annual.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['ticker', 'year', 'shares'])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f'shares_annual.csv: {len(rows)} rows, {len(shares)} tickers')

# sanity: ABOT
abot = [r for r in rows if r['ticker'] == 'ABOT']
print('ABOT shares:', [(r['year'], f"{r['shares']:,}") for r in abot[:3]], '...', [(r['year'], f"{r['shares']:,}") for r in abot[-2:]])
