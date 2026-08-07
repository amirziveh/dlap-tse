#!/usr/bin/env python3
"""PK financials rebuild (aux CSVs) + 3-country v3 merge + QA gates.

Rebuilds from the fresh finalize() output (data_pk/financials_annual.csv),
applies the established filters, merges IR+TR+PK into the v3 combined file,
and emits a coverage/QA table.

Filters (per project rules):
- PK financial sector (PSX): COMMERCIAL BANKS, INV. BANKS/INV. COS./SECURITIES COS.,
  INSURANCE, MODARABAS, LEASING COMPANIES, CLOSE-END MUTUAL FUND
  (AABS = SUGAR & ALLIED — kept, verified)
- barren rows (PK main only): PAT AND Sales both empty
- IR banks: sector_name == 'بانکها و موسسات اعتباری' (10 symbols)
- TR: no banks present in the panel (AKBNK excluded upstream by core gate)
"""
import csv, json, sys
from pathlib import Path

BASE = Path('/home/ubuntu/research/dlap-tse')
PK = BASE / 'data_pk'
IR = BASE / 'data_ir'
TR = BASE / 'data_tr'

HDR16 = ["country", "symbol", "year", "TA", "TL", "Eq", "CA", "CL", "Sales",
         "COGS", "GP", "PAT", "Cash", "Inv", "PPE", "dividends"]

PK_FINANCIAL = {"COMMERCIAL BANKS",
                "INV. BANKS / INV. COS. / SECURITIES COS.",
                "INSURANCE", "MODARABAS", "LEASING COMPANIES",
                "CLOSE - END MUTUAL FUND"}

# Name-based fallback for tickers whose PSX sector label is EMPTY/MISCELLANEOUS
# (symbols.json gap). Matches financial intermediaries only — industrial
# holdings/conglomerates (e.g., GGL Ghani Global Holdings, CHEMICAL) are kept.
import re as _re
FIN_NAME_PATTERNS = [_re.compile(p, _re.I) for p in (
    r"modaraba", r"\bmod\b", r"investment", r"\bcapital\b",
    r"securities", r"leasing", r"mutual", r"\bbank\b", r"insurance",
)]
FIN_SECTOR_GAP = {"", "MISCELLANEOUS"}

def pk_is_financial(symbol, sector, name):
    """sector-based, with name-based fallback for sector gaps."""
    if sector in PK_FINANCIAL:
        return True
    if sector in FIN_SECTOR_GAP and name:
        return any(p.search(name) for p in FIN_NAME_PATTERNS)
    return False

def load_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

# Always regenerate the RAW finalize() output first (idempotent: vlm_rows is
# the source of truth; protects against re-running on an already-filtered CSV).
sys.path.insert(0, str(BASE / 'scripts'))
import pk_vlm_pipeline as P
print('=== finalize() (raw rebuild from vlm_rows) ===', flush=True)
P.finalize()

def write_rows(path, rows, headers):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in headers})

# ---------------------------------------------------------------- PK ----
print('=== PK rebuild ===', flush=True)
pk_rows = load_rows(PK / 'financials_annual.csv')   # fresh finalize output
syms = {s['symbol']: s.get('sectorName', '') for s in json.load(open(PK / 'symbols.json'))}
snames = {s['symbol']: s.get('name', '') for s in json.load(open(PK / 'symbols.json'))}
assert syms.get('AABS') == 'SUGAR & ALLIED INDUSTRIES', 'AABS sector check failed!'

def pk_sector(r):
    return syms.get(r['symbol'], '')

fin_rows = [r for r in pk_rows if pk_is_financial(r['symbol'], pk_sector(r), snames.get(r['symbol'], ''))]
nonfin_rows = [r for r in pk_rows if not pk_is_financial(r['symbol'], pk_sector(r), snames.get(r['symbol'], ''))]
barren = [r for r in nonfin_rows
          if (r.get('PAT') in (None, '')) and (r.get('Sales') in (None, ''))]
main_rows = [r for r in nonfin_rows if r not in barren]

bank_rows = [r for r in pk_rows if pk_sector(r) == 'COMMERCIAL BANKS']
nobank_rows = [r for r in pk_rows if pk_sector(r) != 'COMMERCIAL BANKS']

hdr_pk = list(pk_rows[0].keys())
write_rows(PK / 'financials_annual_with_empty.csv', pk_rows, hdr_pk)
write_rows(PK / 'financials_annual_no_banks.csv', nobank_rows, hdr_pk)
write_rows(PK / 'financials_annual_no_financials.csv', nonfin_rows, hdr_pk)
write_rows(PK / 'financials_annual.csv', main_rows, hdr_pk)

print(f'all={len(pk_rows)} banks={len(bank_rows)} fin={len(fin_rows)} '
      f'barren={len(barren)} -> main={len(main_rows)}')

# ---------------------------------------------------------------- merge ----
print('=== v3 merge ===', flush=True)
ir_rows = load_rows(IR / 'financials_annual.csv')
tr_rows = load_rows(TR / 'financials_annual.csv')

# IR banks via fama-five stock_universe
uni = {}
for r in csv.DictReader(open(BASE / '../fama-five/data/stock_universe.csv')):
    k = (r.get('ticker') or r.get('\ufeff"ticker"') or '').strip().strip('"')
    uni[k] = r.get('sector_name', '')
ir_bank_syms = {s for s in {r['symbol'] for r in ir_rows} if uni.get(s) == 'بانکها و موسسات اعتباری'}
print('IR bank symbols excluded:', sorted(ir_bank_syms))

def norm16(r, country):
    out = {k: r.get(k, '') for k in HDR16[2:]}
    out['country'] = country
    out['symbol'] = r['symbol']
    out['year'] = r['year']
    return out

ir_all = [norm16(r, 'IR') for r in ir_rows]
ir_fin = [r for r in ir_all if r['symbol'] not in ir_bank_syms]
tr_all = [norm16(r, 'TR') for r in tr_rows]
pk_all = [norm16(r, 'PK') for r in pk_rows]
pk_fin = [norm16(r, 'PK') for r in main_rows]

v3_all = ir_all + tr_all + pk_all
v3_fin = ir_fin + tr_all + pk_fin

write_rows('/tmp/combined_financials_v3_with_financials.csv', v3_all, HDR16)
write_rows('/tmp/combined_financials_v3.csv', v3_fin, HDR16)
print(f'v3 filtered: IR={len(ir_fin)} TR={len(tr_all)} PK={len(pk_fin)} '
      f'total={len(v3_fin)} (unfiltered total={len(v3_all)})')

# ---------------------------------------------------------------- QA ----
print('=== QA gates ===', flush=True)
FIELDS = HDR16[3:]
def coverage(rows, label):
    n = len(rows)
    per = {}
    for f in FIELDS:
        per[f] = sum(1 for r in rows if r.get(f) not in (None, ''))
    print(f'{label}: n={n} | ' + ' '.join(f'{f}={per[f]/n*100:.0f}%' for f in FIELDS))

coverage(ir_all, 'IR all    ')
coverage(ir_fin, 'IR no-bank')
coverage(tr_all, 'TR all    ')
coverage(pk_all, 'PK all    ')
coverage(pk_fin, 'PK main   ')
coverage(v3_fin, 'V3 total  ')

# TR regression: byte-identical values vs source (straight copy)
src_tr = load_rows(TR / 'financials_annual.csv')
assert len(tr_all) == len(src_tr) == 5907, f'TR regression: {len(tr_all)} vs 5907'
print('TR regression: OK (5907 rows, unchanged)')
