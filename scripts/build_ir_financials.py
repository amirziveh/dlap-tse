#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_ir_financials.py — complete Iran financials to the 16-field schema.

Base 12 fields: replicated EXACTLY from accounting_panel.csv (verified 100%
against combined_financials_v2.csv IR section: 9,888 rows / 85,537 values).

New fields (all from Rahavard365, same dedup rule as the panel builder —
annual report, per fiscal year keep latest report_date):
  Cash      <- «وجوه نقد و موجودی‌های نزد بانک»           (bs)
  Inv       <- «موجودی مواد و کالا»                       (bs)
  PPE       <- «اموال ماشین آلات و تجهیزات»               (bs)
  dividends <- rahavard_dps: payout_ratio × net_income (fallback dps×capital/1e6)

Backfill: Sales/COGS empty in panel <- unified is-lines
  («درآمد حاصل از خدمات و فروش» / «بهای تمام شده کالای فروش رفته»)

Output: data_ir/financials_annual.csv (16 fields, میلیون ریال)
"""
import csv, collections, sys
from datetime import date
import jdatetime

BASE = '/home/ubuntu/research'
PANEL = f'{BASE}/fama-five/data/processed/accounting_panel.csv'
UNIFIED = f'{BASE}/fama-five/data/rahavard_unified.csv'
DPS = f'{BASE}/fama-five/data/rahavard_dps.csv'
OUT = f'{BASE}/dlap-tse/data_ir/financials_annual.csv'

# ── 1) base 12 fields from panel (v2-verified mapping) ──────────────────────
M12 = {'total_assets': 'TA', 'total_liabilities': 'TL', 'total_equity': 'Eq',
       'current_assets': 'CA', 'current_liabilities': 'CL', 'revenue': 'Sales',
       'cogs': 'COGS', 'gross_profit': 'GP', 'net_income': 'PAT'}

base = {}   # (ticker, gregorian_year) -> {field: value, 'shamsi': fy}
with open(PANEL, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        li = M12.get(r['line_item'])
        if not li:
            continue
        k = (r['ticker'], r['gregorian_year'])
        d = base.setdefault(k, {'shamsi': r['fiscal_year_shamsi']})
        d[li] = r['value']

# ── 2) unified bs lines: cash / inv / ppe, dedup latest report_date ─────────
BS_LINES = {'وجوه نقد و موجودی‌های نزد بانک': 'Cash',
            'موجودی مواد و کالا': 'Inv',
            'اموال ماشین آلات و تجهیزات': 'PPE'}

best = {}   # (ticker, shamsi_year, field) -> value  (latest report_date wins)
def unify(lines, key):
    with open(UNIFIED, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['line_item'] not in key or r['statement'] not in ('bs', 'is'):
                continue
            k = (r['ticker'], r['fiscal_year'][:4], key[r['line_item']])
            rd = r['report_date']
            if k not in lines or rd > lines[k][0]:
                lines[k] = (rd, r['value'])

bs_map = {}
unify(bs_map, BS_LINES)

# ── 3) backfill Sales/COGS from unified is-lines ────────────────────────────
IS_LINES = {'درآمد حاصل از خدمات و فروش': 'Sales',
            'بهای تمام شده کالای فروش رفته': 'COGS'}
is_map = {}
unify(is_map, IS_LINES)

# ── 4) dividends from dps ───────────────────────────────────────────────────
dps = collections.defaultdict(list)
with open(DPS, encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        fy = date.fromisoformat(r['fiscal_year'][:10])
        s_year = str(jdatetime.date.fromgregorian(date=fy).year)
        dps[(r['ticker'], s_year)].append(r)
# dedup (2 cases): keep latest announcement_date
for k in dps:
    dps[k].sort(key=lambda r: r['announcement_date'])
dps_final = {k: v[-1] for k, v in dps.items()}

# ── 5) assemble rows ────────────────────────────────────────────────────────
FIELDS = ['TA', 'TL', 'Eq', 'CA', 'CL', 'Sales', 'COGS', 'GP', 'PAT',
          'Cash', 'Inv', 'PPE', 'dividends']
rows = []
for (ticker, gy), d in sorted(base.items()):
    sy = d['shamsi']
    row = {'country': 'IR', 'symbol': ticker, 'year': gy}
    for f in FIELDS:
        row[f] = d.get(f, '')
    # new bs fields
    for field in ('Cash', 'Inv', 'PPE'):
        hit = bs_map.get((ticker, sy, field))
        if hit:
            row[field] = hit[1]
    # backfill Sales/COGS
    for field in ('Sales', 'COGS'):
        if not row[field]:
            hit = is_map.get((ticker, sy, field))
            if hit:
                row[field] = hit[1]
    # dividends: declared amount = pure_dps × capital / 1e6 (میلیون ریال)
    # (ground truth — survives restatements; payout×PAT only for QA)
    dp = dps_final.get((ticker, sy))
    if dp:
        try:
            row['dividends'] = str(int(round(float(dp['pure_dps']) * float(dp['capital']) / 1e9)))
        except (ValueError, KeyError):
            pass
    rows.append(row)

# ── 6) write ────────────────────────────────────────────────────────────────
with open(OUT, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['country', 'symbol', 'year'] + FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f'written {len(rows)} rows -> {OUT}')

# ── 7) QA ───────────────────────────────────────────────────────────────────
n = len(rows)
print(f'\n=== coverage after (n={n}) ===')
for f in FIELDS:
    filled = sum(1 for r in rows if r[f])
    print(f'  {f:9s} {100*filled/n:6.1f}%  ({filled})')

# accounting identity checks
def num(r, f):
    try:
        return float(r[f])
    except (ValueError, TypeError):
        return None

bad_eq = 0
for r in rows:
    ta, tl, eq = num(r, 'TA'), num(r, 'TL'), num(r, 'Eq')
    if ta and tl and eq and abs(ta - tl - eq) > max(1, 0.02 * ta):
        bad_eq += 1
print(f'\nEq vs TA-TL mismatch (>2%): {bad_eq}')
bad_gp = 0
for r in rows:
    s, c, g = num(r, 'Sales'), num(r, 'COGS'), num(r, 'GP')
    if s and c and g and abs(s - c - g) > max(1, 0.02 * s):
        bad_gp += 1
print(f'GP vs Sales-COGS mismatch (>2%): {bad_gp}')

# par-share sanity: implied PAT (eps × capital/par=1000) vs panel PAT
import random
random.seed(42)
agree = 0; tested = 0; samples = []
dps_by_key = {k: v for k, v in dps_final.items()}
for r in random.sample(rows, 400):
    if not (r['dividends'] and r['PAT']):
        continue
    # rebuild key from base map
    k = (r['symbol'], r['year'])
    if k not in base: continue
    sy = base[k]['shamsi']
    dp = dps_by_key.get((r['symbol'], sy))
    if not dp: continue
    try:
        implied = float(dp['pure_eps']) * float(dp['capital']) / 1e9
        pat = float(r['PAT'])
        tested += 1
        if implied > 0 and abs(implied - pat) <= 0.05 * max(implied, pat):
            agree += 1
        elif implied <= 0 and pat <= 0:
            agree += 1
    except (ValueError, KeyError):
        pass
print(f'\npar=1000 check (implied PAT vs panel PAT, 5% tol): {agree}/{tested} agree')

# spot-checks
for probe in [('آباد', '2019'), ('فولاد', '2022'), ('خودرو', '2018')]:
    hits = [r for r in rows if r['symbol'].startswith(probe[0]) and r['year'] == probe[1]]
    for h in hits[:1]:
        print(f'\nspot {probe}: {h}')
