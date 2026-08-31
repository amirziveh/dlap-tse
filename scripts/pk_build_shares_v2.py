#!/usr/bin/env python3
"""Build PK shares-outstanding table + market cap series (no PDFs) — v2, 2026-08-31.

Sources:
  - pkf_seed.json VLM extraction (355 companies x FY2013-2026): issued/subscribed/paid-up
    capital per year (audited validation: implied 2022-25 must match the PAT/EPS-implied
    counts within 15% per company, unit flips normalized)
  - dps per-year financials (2022-2025): shares = PAT_after_tax / EPS  [anchor years]
  - dps current profile: eq.shares (2026+)

The pre-2022 flat backfill of v1 (which made nsi an extraction artifact, 2026-08-31 audit)
is REPLACED by the extracted issued-capital series where validated; companies without a
validated extraction keep the v1 series (unchanged).

Outputs:
  data_pk/shares_annual.csv       ticker, year, shares (fiscal-year, stepped)
  data_pk/market_cap_monthly.csv  ticker, year, month, date, close, shares, mktcap
"""
import json, csv, os
from pathlib import Path

ROOT = Path('/home/ubuntu/research/dlap-tse')
PK = ROOT / 'data_pk'
dps = json.load(open(PK / 'unit_norm' / 'pk_dps_financials.json'))
V2 = json.load(open('/tmp/pk_shares_v6.json'))  # validated extraction (per-company dict year->shares)

# 1) anchor years 2022-2025 from PAT/EPS (unchanged from v1)
shares = {}
for sym, d in dps.items():
    fin = d.get('fin', {})
    for yr in sorted(fin, key=int):
        pat = fin[yr].get('Profit after Taxation')
        eps = fin[yr].get('EPS')
        if pat and eps and eps > 0 and pat > 0:
            shares.setdefault(sym, {})[int(yr)] = round(pat * 1000.0 / eps)
    eq = d.get('eq', {})
    if eq.get('shares'):
        shares.setdefault(sym, {})[2026] = int(eq['shares'])

# 2) merge the validated extraction: pre-2022 (and any missing year) from v2
replaced = kept = 0
for sym, ys in V2.items():
    tgt = shares.setdefault(sym, {})
    for ystr, v in ys.items():
        y = int(ystr)
        if y < 2022 and y >= 2013 and v > 0:
            if tgt.get(y) != v:
                replaced += 1
            tgt[y] = round(v)
        elif y not in tgt and v > 0:
            tgt[y] = round(v)

rows = []
for sym, years in sorted(shares.items()):
    for y in range(2013, 2027):
        if y in years:
            v = years[y]
        elif 2026 in years:
            v = years[2026]
        else:
            continue
        rows.append({'ticker': sym, 'year': y, 'shares': v})

with open(PK / 'shares_annual.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['ticker', 'year', 'shares'])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f'shares_annual.csv: {len(rows)} rows, {len(shares)} tickers, {replaced} pre-2022 cells replaced by audited extraction')

# sanity: issuers that must show real variation
for probe in ['ADAMS', 'AGTL', 'ACPL']:
    ser = [(r['year'], r['shares']) for r in rows if r['ticker'] == probe]
    print(probe, ser[:6])
