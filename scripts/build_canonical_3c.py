#!/usr/bin/env python3
"""build_canonical_3c.py — canonical JSON for render_manuscript_3c.py.

Single source of truth: every number is read from the results CSVs in the
repo. Run this whenever any results file changes, then render_manuscript_3c.py.
Output: /tmp/dlap_canonical.json (renderer input) — schema documented inline.
"""
import csv
import json

import numpy as np

ROOT = '/home/ubuntu/research/dlap-tse'
CC = {'IR': 'results', 'TR': 'results_tr', 'PK': 'results_pk'}
SPEC_FILES = {'E2': 'e2', 'E3': 'e3', 'E4A': 'e4a', 'E4B': 'e4b',
              'E5A': 'e5a', 'E5B': 'e5b', 'E8': 'e8', 'E8B': 'e8b'}
# HJ bounds (avg max-Sharpe of test assets, train cov) are properties of the
# RETURN panel: IR = bank-free re-estimation (commit 4831af91), TR/PK panels
# unchanged since their E1 runs. Not recomputed here; e1_summary.md is stale.
HJ = {'IR': 2.845, 'TR': 2.605, 'PK': 3.366}


def rd(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


def kv(path):
    return {r[0]: r[1] for r in csv.reader(open(path, encoding='utf-8-sig'))
            if len(r) >= 2}


def series(path):
    """pooled series: single-column CSVs (deep specs) or model,value (e1)."""
    rows = list(csv.reader(open(path, encoding='utf-8-sig')))
    if rows[0] and rows[0][0] == 'model':
        out = {}
        for r in rows[1:]:
            if len(r) >= 2:
                out.setdefault(r[0], []).append(float(r[1]))
        return out
    return [float(r[0]) for r in rows[1:] if r and r[0]]


def sign_norm(vals, nwin):
    v = np.array(vals)
    w = len(v) // nwin
    segs = [v[i * w:(i + 1) * w] if v[i * w:(i + 1) * w].mean() >= 0
            else -v[i * w:(i + 1) * w] for i in range(nwin)]
    cat = np.concatenate(segs)
    return float(cat.mean() / cat.std(ddof=0) * np.sqrt(12))


out = {'hj': HJ}
for tag, base in CC.items():
    # dynamic window count: E2's pooled OOS months / window length (60/12
    # protocol -> 12-month test windows); reflects disclosed skips (PK w0)
    _e2m = len(series(f'{ROOT}/{base}/e2_pooled_series.csv'))
    _nw = _e2m // 12
    assert _nw * 12 == _e2m, f'{tag}: e2 pooled months {_e2m} not a multiple of 12'
    d = {'n_windows': _nw}
    # master: seed-42 results of every model (deep + benchmarks + linear)
    master = {}
    for spec, f in SPEC_FILES.items():
        k = kv(f'{ROOT}/{base}/{f}_results.csv')
        master[spec] = {'sharpe': k['sharpe_pooled'], 'ev': k['ev'],
                        'rms_alpha_pct': k['rms_alpha_pct'],
                        'max_alpha_pct': k['max_alpha_pct']}
    for b in rd(f'{ROOT}/{base}/e1_benchmarks.csv'):
        master[b['name']] = {'sharpe': b['sharpe_pooled'], 'ev': b['ev'],
                             'rms_alpha_pct': b['rms_alpha_pct'],
                             'max_alpha_pct': b['max_alpha_pct']}
    for r in rd(f'{ROOT}/{base}/master_results.csv'):
        if r['model'].startswith('Linear SDF'):
            ch = '11ch' if '11ch' in r['model'] else '20ch'
            master[f'Linear SDF ({ch})'] = {'sharpe': r['sharpe'],
                                            'ev': r['ev'],
                                            'rms_alpha_pct': r['rms_alpha_pct'],
                                            'max_alpha_pct': r['max_alpha_pct']}
        elif r['model'] in ('E2-CS', 'E3-CS'):  # charscore robustness
            master[r['model']] = {'sharpe': r['sharpe'], 'ev': r['ev'],
                                  'rms_alpha_pct': r['rms_alpha_pct'],
                                  'max_alpha_pct': r['max_alpha_pct']}
    d['master'] = master
    # per-seed table (keep all rows; renderer filters to the 8 deep specs)
    d['seed'] = rd(f'{ROOT}/{base}/seed_sensitivity.csv')
    # sign-normalized pooled Sharpe (seed 42, per-window flip convention)
    d['sign_norm'] = {
        'E2': sign_norm(series(f'{ROOT}/{base}/e2_pooled_series.csv'), _nw),
        'E8': sign_norm(series(f'{ROOT}/{base}/e8_pooled_series.csv'), _nw)}
    # pooled monthly series (sign-para + fig_sign_windows): e1 = {model: [vals]},
    # deep specs keyed lowercase with {'S': [...]}
    e1 = series(f'{ROOT}/{base}/e1_pooled_series.csv')
    pooled = {'e1': {k: v for k, v in e1.items()}}
    for spec, f in SPEC_FILES.items():
        pooled[spec.lower()] = {'S': series(f'{ROOT}/{base}/{f}_pooled_series.csv')}
    d['pooled'] = pooled
    # sign-symmetry diagnostic: per-window L(+w), L(-w), rel gap (seed 42).
    # Gap = |L+ - L-| / min(L+, L-): 0 = perfect mirror symmetry; large =
    # the converged orientation prices strictly better than its mirror.
    sym = rd(f'{ROOT}/{base}/e2_sign_symmetry.csv')
    gaps = [float(r['rel_gap']) for r in sym
            if r.get('rel_gap') not in ('nan', '')]
    d['sym_gap'] = {
        'n': len(gaps),
        'median': float(np.median(gaps)) if gaps else None,
        'max': float(np.max(gaps)) if gaps else None}
    # bootstrap / spa / leverage / e2lag
    d['boot'] = {'e2': rd(f'{ROOT}/{base}/sharp_diff_bootstrap_e2.csv'),
                 'e8': rd(f'{ROOT}/{base}/sharp_diff_bootstrap_e8.csv')}
    d['spa'] = rd(f'{ROOT}/{base}/spa_test.csv')
    d['leverage'] = rd(f'{ROOT}/{base}/bench_leverage.csv')
    rows = list(csv.reader(open(f'{ROOT}/{base}/e2lag_results.csv', encoding='utf-8-sig')))
    k = dict(zip(rows[0], rows[1]))  # transposed layout: header row + value row
    d['e2lag'] = dict(zip(rows[0], rows[1]))  # keep full row incl. note
    # placebo summary
    d['placebo'] = rd(f'{ROOT}/{base}/placebo_summary.csv')
    # loadings bootstrap
    d['loadings'] = {'sy': rd(f'{ROOT}/{base}/e6_loadings_boot_sy.csv'),
                     'all': rd(f'{ROOT}/{base}/e6_loadings_boot_all.csv')}
    # linear SDF detail (replication-package rows)
    d['lin'] = [dict(r, name=r['name']) for r in rd(f'{ROOT}/{base}/linear_sdf_results.csv')]
    out[tag] = d

with open('/tmp/dlap_canonical.json', 'w') as f:
    json.dump(out, f)
print('canonical JSON written: /tmp/dlap_canonical.json')
for tag in CC:
    sn = out[tag]['sign_norm']
    print(f"  {tag}: nwin={out[tag]['n_windows']} sign_norm E2={sn['E2']:.3f} "
          f"E8={sn['E8']:.3f} E2(42)={out[tag]['master']['E2']['sharpe']}")
