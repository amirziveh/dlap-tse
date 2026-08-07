#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_tr_financials.py — rebuild Turkey financials, 16-field schema.

Source: data_tr/financials/<TICKER>/<YEAR>.json (KAP 110-chart, raw TL).

KAP item codes (verified against v2 for TA/CA/CL/Eq/Sales/COGS/GP/PAT/Cash/PPE):
  TA=1BL  CA=1A  CL=2A  TL=2A+2B (no direct TL code)  Eq=2N
  Cash=1AA  Inv=1AF (Stoklar/Inventories)  PPE=1BG
  Sales=3C  COGS=3CA  GP=3D  PAT=3L  dividends=4CBB

Fixes vs combined v2:
  * Inv: v2 used a wrong source (values not present in raw data, garbage for
    REITs/insurers) -> 1AF Stoklar (correct, ~99% coverage)
  * COGS: KAP reports negative (deduction); v2 kept sign -> abs()
  * dividends: KAP 4CBB is negative cash outflow; v2 kept sign -> abs();
    absent 4CBB (old years 2013-15 + financial-sector cash-flow format)
    stays empty, documented.
Output: data_tr/financials_annual.csv (country=TR, TL units)
"""
import json, os, csv, glob

FIN = '/home/ubuntu/research/dlap-tse/data_tr/financials'
OUT = '/home/ubuntu/research/dlap-tse/data_tr/financials_annual.csv'
V2 = '/tmp/combined_financials_v2.csv'

def code_val(j, code):
    for i in (j.get('value') or []):
        if i.get('itemCode') == code:
            return i.get('value1') or i.get('value2') or ''
    return ''

def build():
    rows = []
    for path in sorted(glob.glob(FIN + '/*/*.json')):
        tkr = os.path.basename(os.path.dirname(path))
        yr = os.path.basename(path)[:4]
        try:
            j = json.load(open(path))
        except Exception:
            continue
        def g(code):
            return code_val(j, code)
        ta, ca, cl, eq = g('1BL'), g('1A'), g('2A'), g('2N')
        tl, sales, cogs, gp = g('2ODB'), g('3C'), g('3CA'), g('3D')
        pat, cash, inv, ppe = g('3L'), g('1AA'), g('1AF'), g('1BG')
        div = g('4CBB')
        # core gate: need at least TA+Eq (or TL) and Sales+PAT to be usable
        if not (ta and eq and sales and pat):
            continue
        row = {'country': 'TR', 'symbol': tkr, 'year': yr,
               'TA': ta,
               'TL': (str(float(tl) - float(eq)) if tl and eq else ''),
               'Eq': eq, 'CA': ca, 'CL': cl,
               'Sales': sales,
               'COGS': str(abs(float(cogs))) if cogs else '',
               'GP': gp, 'PAT': pat,
               'Cash': cash, 'Inv': inv, 'PPE': ppe,
               'dividends': str(abs(float(div))) if div else ''}
        rows.append(row)
    return rows

rows = build()
# dedup (shouldn't be needed but safe)
seen = set(); uniq = []
for r in rows:
    k = (r['symbol'], r['year'])
    if k in seen: continue
    seen.add(k); uniq.append(r)
rows = uniq
rows.sort(key=lambda r: (r['symbol'], r['year']))

with open(OUT, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f'written {len(rows)} rows -> {OUT}')

# ── QA vs v2 ────────────────────────────────────────────────────────────────
v2 = {(r['symbol'], r['year']): r for r in csv.DictReader(open(V2)) if r['country'] == 'TR'}
new = {(r['symbol'], r['year']): r for r in rows}
only_v2 = set(v2) - set(new)
only_new = set(new) - set(v2)
print(f'v2 rows: {len(v2)} | new rows: {len(new)} | only in v2: {len(only_v2)} | only in new: {len(only_new)}')
if only_v2:
    for k in sorted(only_v2)[:8]:
        print('  v2-only:', k, 'TA=', v2[k]['TA'], 'Sales=', v2[k]['Sales'])
if only_new:
    for k in sorted(only_new)[:8]:
        print('  new-only:', k, 'TA=', new[k]['TA'], 'Sales=', new[k]['Sales'])

M = ['TA', 'TL', 'Eq', 'CA', 'CL', 'Sales', 'COGS', 'GP', 'PAT', 'Cash', 'Inv', 'PPE', 'dividends']
diffs = {f: 0 for f in M}
ex = {}
for k in set(v2) & set(new):
    for f in M:
        if (v2[k][f] or '').strip() != (new[k][f] or '').strip():
            diffs[f] += 1
            ex.setdefault(f, (k, v2[k][f], new[k][f]))
print('\nvalue diffs vs v2 (per field):', diffs)
for f, e in ex.items():
    print('  %s: %s  v2=%s -> new=%s' % (f, e[0], str(e[1])[:25], str(e[2])[:25]))

# identities
def num(r, f):
    try: return float(r[f])
    except: return None
bad_eq = bad_gp = 0
for r in rows:
    ta, tl, eq = num(r, 'TA'), num(r, 'TL'), num(r, 'Eq')
    if ta and tl and eq and abs(ta - tl - eq) > max(1, 0.02 * ta): bad_eq += 1
    s, c, gp_ = num(r, 'Sales'), num(r, 'COGS'), num(r, 'GP')
    if s and c and gp_ and abs(s - c - gp_) > max(1, 0.02 * s): bad_gp += 1
print(f'\nEq=TA-TL mismatch >2%: {bad_eq} | GP=Sales-COGS mismatch >2%: {bad_gp}')

# coverage
n = len(rows)
print('\ncoverage:')
for f in M:
    filled = sum(1 for r in rows if r[f])
    print('  %-9s %6.1f%% (%d)' % (f, 100 * filled / n, filled))
