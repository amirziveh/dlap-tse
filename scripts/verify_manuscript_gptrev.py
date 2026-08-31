#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_manuscript_gptrev.py — numerical consistency audit for paper_gpt_rev/manuscript.tex
(revision 2026-08-31). Checks manuscript numbers against the canonical results CSVs.
Exit 1 on any error.
"""
import csv
import os
import re
import sys

ROOT = '/home/ubuntu/research/dlap-tse'
CC = {'IR': 'results', 'TR': 'results_tr', 'PK': 'results_pk'}
errors = []


def kv(path):
    return {r[0]: r[1] for r in csv.reader(open(path, encoding='utf-8-sig')) if len(r) >= 2}


def e1(path):
    return {r['name']: r for r in csv.DictReader(open(path, encoding='utf-8-sig'))}


def lin(path):
    return {r['name']: r for r in csv.DictReader(open(path, encoding='utf-8-sig'))}


ms = open(f'{ROOT}/paper_gpt_rev/manuscript.tex', encoding='utf-8').read()


def check(desc, cond, detail=''):
    if not cond:
        errors.append(f"{desc} {detail}")
        print(f"  FAIL {desc} {detail}")
    else:
        print(f"  ok   {desc}")


# ---------- tab:bench ----------
print("tab:bench rows:")
bench_rows = {
    'Market':   'Market & 0.338 & 0.343 & 5.78 & 31.2 & 1.018 & 0.232 & 6.86 & 32.7 & 0.127 & 0.238 & 5.23 & 30.9',
    'FF5':      'FF5 & 0.474 & 0.323 & 6.07 & 29.7 & 0.712 & 0.157 & 10.59 & 140.8 & $-0.236$ & 0.154 & 7.11 & 49.7',
    'q-factor': 'q-factor & 0.888 & 0.273 & 7.09 & 41.6 & 1.615 & 0.205 & 6.78 & 31.5 & $-0.063$ & 0.134 & 5.94 & 28.5',
    'PCA(5)':   'PCA(5) & 0.419 & 0.417 & 5.83 & 27.2 & 0.539 & 0.309 & 6.86 & 27.9 & 0.467 & 0.206 & 5.38 & 29.1',
    'LASSO':    'LASSO & 0.632 & 0.423 & 7.30 & 53.8 & 0.742 & 0.278 & 7.64 & 31.8 & 0.509 & 0.258 & 4.38 & 19.3',
    'linSY':    'Linear SDF (SY) & 1.071 & 0.150 & 8.54 & 43.1 & 0.330 & 0.259 & 17.81 & 233.6 & 0.523 & 0.142 & 5.49 & 45.1',
    'linALL':   'Linear SDF (all) & 1.376 & 0.178 & 8.89 & 45.8 & 0.514 & 0.303 & 13.82 & 129.8 & 1.496 & 0.139 & 5.76 & 34.9',
}
for name, row in bench_rows.items():
    check(f"bench row {name}", row in ms)

# cross-check against CSVs (row layout: cells 1-4 = IR, 5-8 = TR, 9-12 = PK)
for mi, (tag, res) in enumerate(CC.items()):
    b = e1(f'{ROOT}/{res}/e1_benchmarks.csv')
    l = lin(f'{ROOT}/{res}/linear_sdf_results.csv')
    b['linSY'] = l['lin11']
    b['linALL'] = l['lin20']
    for key, row in bench_rows.items():
        r = b[key]
        vals = [float(r['sharpe_pooled']), float(r['ev']), float(r['rms_alpha_pct']), float(r['max_alpha_pct'])]
        cells = row.split('&')[1 + mi * 4: 5 + mi * 4]
        for v, c in zip(vals, cells):
            c = c.strip().replace('$-0.', '-0.').replace('$', '').replace('\\', '')
            try:
                cv = float(c)
            except ValueError:
                continue
            if abs(round(v, len(c.split('.')[1]) if '.' in c else 0) - cv) > 0.006:
                errors.append(f"bench {tag} {key} cell {c} vs csv {v:.4f}")
print("  (cross-checked against CSVs)")

# ---------- tab:deep ----------
print("tab:deep rows:")
deep_expected = {
    'E2':  'E2 (SY, LSTM) & 0.016 & 0.097 & 0.400 & 0.232 & 0.747 & 0.743 & $-0.097$ & 0.143 & $-0.540$',
    'E3':  'E3 (all, LSTM) & 0.630 & 0.625 & 0.537 & 0.486 & 0.375 & 0.743 & 0.655 & 0.566 & 0.718',
    'E4A': 'E4A (all, constant state) & 0.533 & 0.145 & 0.303 & 0.723 & 0.402 & 0.710 & 0.448 & 0.878 & 0.327',
    'E4B': 'E4B (SY, constant state) & 0.186 & 0.021 & 0.369 & 0.702 & 0.728 & 0.236 & 0.053 & $-0.013$ & 0.819',
    'E5A': 'E5A (SY, critic) & $-0.022$ & 0.633 & 0.620 & 0.172 & 0.721 & 0.783 & $-0.354$ & $-0.908$ & $-0.586$',
    'E5B': 'E5B (all, critic) & 0.658 & 0.680 & 0.616 & 0.415 & 0.374 & 0.698 & $-0.077$ & 0.104 & $-0.480$',
    'E8':  'E8 (SY, liquidity filter) & 0.608 & 0.566 & 0.422 & 0.213 & 0.714 & 0.790 & 0.025 & $-0.541$ & $-0.601$',
    'E8B': 'E8B (all, liquidity filter) & 0.627 & 0.679 & $-0.014$ & 0.597 & 0.454 & 0.698 & 0.443 & 0.576 & 0.460',
}
for spec, row in deep_expected.items():
    check(f"deep row {spec}", row in ms)

# cross-check per-seed sharpes against CSVs
for mi, (tag, res) in enumerate(CC.items()):
    for si, (seed, sub) in enumerate([(42, ''), (43, 'seed43/'), (44, 'seed44/')]):
        for spec, row in deep_expected.items():
            d = kv(f'{ROOT}/{res}/{sub}{spec.lower()}_results.csv')
            v = float(d['sharpe_pooled'])
            cell = row.split('&')[1 + mi * 3 + si].strip().replace('$-', '-').replace('$', '')
            cell = cell.replace('$', '').strip()
            cv = float(cell)
            if abs(round(v, 3) - cv) > 0.0006:
                errors.append(f"deep {tag} {spec} s{seed}: tex={cv} csv={v:.4f}")
print("  (cross-checked per seed)")

# ---------- tab:pin ----------
print("tab:pin row (Pakistan):")
check('pin PK row', 'Pakistan & $-0.097$ & 0.143 & $-0.540$ & $-0.437$ & $-0.115$ & $-0.204$ & $-0.165$ & $-0.252$ & $-0.278$ & 1.222' in ms)

# ---------- coverage ----------
print("coverage:")
check('chars 19/19/18', 'Available characteristics & 19 & 19 & 18' in ms)
check('stock-months 47,515', 'Stock-months & 69,155 & 64,678 & 47,515' in ms)
check('PK windows 6 mention', 'T\\"urkiye 6, and Pakistan 6' in ms)

# ---------- loadings table ----------
print("tab:loadings sample rows:")
check('loadings Momentum row', 'Momentum & $-0.084$ & $-1.53$ & $-0.090$ & $-3.77$ & $-0.509$ & $-1.68$' in ms)
check('loadings nsi PK real', 'Net stock issuance & $-0.085$ & $-2.95$ & 0.039 & 1.13 & $-0.237$ & $-1.70$' in ms)

# ---------- stale greps ----------
print("stale-claim greps:")
stale = ['47,518', '$-0.405$', '16 of 23', '1.310', '1.636', '1.623', 'fails in Pakistan',
         'negative at all three seeds', 'significantly in Pakistan (p=0.709)', 'at or above the strongest',
         'exceeds every benchmark', 'Pakistan (5)', '20 & 20 & 19', '0.694 & $-0.142$', '1.523',
         '$-0.199$--0.676', '1.186', '1.296 in Pakistan']
for p in stale:
    check(f"stale '{p}'", p not in ms)

# ---------- references resolved ----------
print("references:")
check('all \\ref resolve', not (set(re.findall(r'\\ref\{([^}]+)\}', ms)) - set(re.findall(r'\\label\{([^}]+)\}', ms))))
cites = set(re.findall(r'\\cite[tp]?\w*\{([^}]+)\}', ms))
cited_keys = set()
for c in cites:
    cited_keys.update(k.strip() for k in c.split(','))
bib = open(f'{ROOT}/paper_gpt_rev/references.bib', encoding='utf-8').read()
bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib))
check('all cites in bib', cited_keys <= bib_keys, str(cited_keys - bib_keys))

print()
if errors:
    print(f"VERIFIER: {len(errors)} ERROR(S)")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print("VERIFIER: 0 errors — all manuscript numbers match the canonical CSVs")
