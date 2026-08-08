#!/usr/bin/env python3
"""pk_gap_matrix.py — coverage matrix for PK data (Phase 0 baseline / reusable after v3).

Outputs:
  data_pk/GAP_MATRIX_<date>.csv   — field x year coverage, per-layer
  stdout summary                  — one-line numbers per layer

Run: python3 scripts/pk_gap_matrix.py
"""
import csv, json, os, sys
from collections import Counter, defaultdict
from datetime import date

PK = '/home/ubuntu/research/dlap-tse/data_pk'
EMPTY = ('', 'None', 'nan', 'NaN', 'NA', 'N/A', '-')

def is_empty(v):
    return v is None or str(v).strip() in EMPTY

FIN_FIELDS = ['TA','TL','Eq','CA','CL','Sales','COGS','GP','PAT','Cash','Inv','PPE',
              'lt_investments','dividends','dividends_paid','cfo']
REAL_FIELDS = [f for f in FIN_FIELDS if f not in ('dividends','lt_investments','dividends_paid')]

def load(path, key_col=None):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))

def main():
    out_rows = []
    def add(layer, dim, name, filled, total):
        out_rows.append({'layer': layer, 'dim': dim, 'name': name,
                         'filled': filled, 'total': total,
                         'coverage_pct': round(100.0*filled/total, 2) if total else None})

    # ---- financials ----
    fin = load(f'{PK}/financials_annual.csv')
    n = len(fin)
    print(f'financials: {n} rows, {len(set(r["symbol"] for r in fin))} symbols')
    per_year = defaultdict(int)
    for r in fin:
        per_year[r['year']] += 1
    for y in sorted(per_year):
        add('financials', 'year', y, per_year[y], n)
    for f in FIN_FIELDS:
        filled = sum(1 for r in fin if not is_empty(r.get(f,'')))
        add('financials', 'field', f, filled, n)
        print(f'  {f}: {filled} ({100.0*filled/n:.1f}%)')

    # ---- characteristics ----
    chars = load(f'{PK}/characteristics_panel.csv')
    if chars:
        nc = len(chars)
        char_cols = [k for k in chars[0].keys() if k not in ('ticker','year','month')]
        print(f'\ncharacteristics: {nc} rows, cols={len(char_cols)}')
        per_year_c = defaultdict(int)
        for r in chars:
            per_year_c[r.get('year','')] += 1
        for y in sorted(per_year_c):
            add('characteristics', 'year', y, per_year_c[y], nc)
        for c in char_cols:
            filled = sum(1 for r in chars if not is_empty(r.get(c,'')))
            add('characteristics', 'field', c, filled, nc)
            print(f'  {c}: {filled} ({100.0*filled/nc:.1f}%)')

    # ---- macro ----
    macro = load(f'{PK}/macro_panel.csv')
    if macro:
        nm = len(macro)
        mcols = [k for k in macro[0].keys() if k != 'month']
        print(f'\nmacro: {nm} rows, series={mcols}')
        for c in mcols:
            filled = sum(1 for r in macro if not is_empty(r.get(c,'')))
            add('macro', 'series', c, filled, nm)
        if macro:
            add('macro', 'span', 'first', 1, 1)
            add('macro', 'span', 'last', 1, 1)

    # ---- returns ----
    rets = load(f'{PK}/processed/monthly_returns.csv')
    if rets:
        nr = len(rets)
        print(f'\nmonthly_returns: {nr} rows, {len(set(r["ticker"] for r in rets))} tickers')
        add('returns', 'layer', 'symbol-months', nr, nr)

    # ---- market cap / shares ----
    mc = load(f'{PK}/market_cap_monthly.csv')
    if mc:
        key = 'symbol' if 'symbol' in mc[0] else 'ticker'
        add('marketcap', 'layer', 'rows', len(mc), len(mc))
        add('marketcap', 'layer', 'symbols', len(set(r[key] for r in mc)), len(set(r[key] for r in mc)))
    sh = load(f'{PK}/shares_annual.csv')
    if sh:
        key = 'symbol' if 'symbol' in sh[0] else 'ticker'
        add('shares', 'layer', 'rows', len(sh), len(sh))
        add('shares', 'layer', 'symbols', len(set(r[key] for r in sh)), len(set(r[key] for r in sh)))

    # ---- gaps report summary ----
    gaps = load(f'{PK}/data_gaps_report.csv')
    if gaps:
        print(f'\ndata_gaps_report: {len(gaps)} rows')
        for lvl, cnt in Counter(r['level'] for r in gaps).most_common():
            add('gaps', 'level', lvl, 0, cnt)
            print(f'  {lvl}: {cnt}')

    # ---- verification ----
    vr = load(f'{PK}/verification_report.csv')
    if vr:
        print(f'\nverification_report: {len(vr)} rows')
        for lvl, cnt in Counter(r.get('level','') for r in vr).most_common():
            add('verification', 'level', lvl, 0, cnt)
            print(f'  {lvl}: {cnt}')

    # ---- write output ----
    today = date.today().isoformat()
    out = f'{PK}/GAP_MATRIX_{today}.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f'\nwritten: {out} ({len(out_rows)} rows)')

if __name__ == '__main__':
    main()
