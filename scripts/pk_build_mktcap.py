#!/usr/bin/env python3
"""Build PK market-cap monthly series: shares (annual, stepped) x month-end close.
Reads data_pk/processed/monthly_returns.csv (last_close) + shares_annual.csv.
"""
import csv
from collections import defaultdict
from pathlib import Path

PK = Path('/home/ubuntu/research/dlap-tse/data_pk')

sh = defaultdict(dict)
with open(PK / 'shares_annual.csv') as f:
    for r in csv.DictReader(f):
        sh[r['ticker']][int(r['year'])] = int(r['shares'])

def shares_at(sym, year):
    y = sh.get(sym, {})
    if not y:
        return None
    yr = min(max(year, min(y)), max(y))
    return y.get(yr)

out = []
with open(PK / 'processed' / 'monthly_returns.csv') as f:
    for r in csv.DictReader(f):
        try:
            close = float(r['last_close'])
            year = int(r['year'])
        except (ValueError, TypeError):
            continue
        s = shares_at(r['ticker'], year)
        if s and close > 0:
            out.append({
                'ticker': r['ticker'], 'year': year, 'month': int(r['month']),
                'date': r.get('date', ''), 'close': round(close, 2),
                'shares': s, 'mktcap': round(s * close, 2),
            })

with open(PK / 'market_cap_monthly.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['ticker', 'year', 'month', 'date', 'close', 'shares', 'mktcap'])
    w.writeheader()
    for r in sorted(out, key=lambda x: (x['ticker'], x['year'], x['month'])):
        w.writerow(r)

print(f'market_cap_monthly.csv: {len(out):,} rows')
# sanity: ABOT 2025
ab = [r for r in out if r['ticker'] == 'ABOT' and r['year'] == 2025]
if ab:
    print('ABOT 2025-12:', ab[-1], '| mktcap =', f"{ab[-1]['mktcap']/1e6:.1f}M")
