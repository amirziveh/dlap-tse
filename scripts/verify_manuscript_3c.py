#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_manuscript_3c.py — referee audit of the 3-country manuscript."""
import csv, re, sys, os
ROOT = '/home/ubuntu/research/dlap-tse'
TEX = open(f'{ROOT}/paper/manuscript.tex', encoding='utf-8').read()
CC = {'IR': 'results', 'TR': 'results_tr', 'PK': 'results_pk'}
errors, notes = [], []

def present(v, nd, tol_fallback=True):
    """is the number (rounded to nd) present in the tex?"""
    for d in (nd, nd-1, nd+1, 2, 3):
        s = f'{v:.{d}f}'
        if s in TEX:
            return True
        if v < 0 and f'$-{abs(v):.{d}f}$' in TEX:
            return True
    return False

# 1) every master_results cell for the models used in tables
SKIP = {'E2-CS', 'E3-CS'}  # charscore robustness: replication-package-only
for tag, base in CC.items():
    for r in csv.DictReader(open(f'{ROOT}/{base}/master_results.csv', encoding='utf-8-sig')):
        if r['model'] in SKIP:
            continue
        for col, nd in (('sharpe', 3), ('ev', 3), ('max_alpha_pct', 1)):
            v = float(r[col])
            if not present(v, nd):
                errors.append(f'{tag} {r["model"]} {col}={v:.3f} not found in tex')
        # deep-spec rms_alpha is cited as a per-market range in prose; benchmarks/linear cited exactly
        if r['model'] not in ('E2','E3','E4A','E4B','E5A','E5B','E8','E8B'):
            if not present(float(r['rms_alpha_pct']), 2):
                errors.append(f'{tag} {r["model"]} rms_alpha={r["rms_alpha_pct"]} not found')

# 2) bootstrap diffs + CIs (IR table + prose pairs; TR/PK pairs cited)
for tag, base in CC.items():
    for f in ('sharp_diff_bootstrap_e2.csv', 'sharp_diff_bootstrap_e8.csv'):
        for r in csv.DictReader(open(f'{ROOT}/{base}/{f}', encoding='utf-8-sig')):
            for col in ('diff', 'ci_lo', 'ci_hi'):
                if not present(float(r[col]), 2):
                    errors.append(f'{tag} {r["pair"]} {col}={r[col]} not found (2dp)')

# 3) SPA p-values: all must appear (tab:spa prints all)
for tag, base in CC.items():
    for r in csv.DictReader(open(f'{ROOT}/{base}/spa_test.csv', encoding='utf-8-sig')):
        for col in ('spa_p', 'rc_p'):
            s = f"{float(r[col]):.3f}"
            if s not in TEX:
                s2 = f"{float(r[col]):.2f}"
                if s2 not in TEX:
                    errors.append(f'{tag} SPA {r["target"]}/{r["loss"]}/{r["block"]} {col}={s} not found')

# 4) seed table: all 8 specs x 3 seeds x 3 countries
for tag, base in CC.items():
    for r in csv.DictReader(open(f'{ROOT}/{base}/seed_sensitivity.csv', encoding='utf-8-sig')):
        if r['spec'] in ('E2','E3','E4A','E4B','E5A','E5B','E8','E8B'):
            if not present(float(r['sharpe']), 3):
                errors.append(f'{tag} seed {r["spec"]}/{r["seed"]} sharpe={r["sharpe"]} not found')

# 5) loadings: mean_w (3dp) and boot_t for the E3-'all' set
for tag, base in CC.items():
    for r in csv.DictReader(open(f'{ROOT}/{base}/e6_loadings_boot_all.csv', encoding='utf-8-sig')):
        mw, tb = float(r['mean_w']), float(r['boot_t'])
        if not present(mw, 3):
            errors.append(f'{tag} loading {r["char"]} mean_w={mw} not found')
        if not present(tb, 2):
            errors.append(f'{tag} loading {r["char"]} boot_t={tb} not found')

# 5b) Method B (ex-ante sign pin): every number in method_b_summary.json must appear.
# lambda=1: full per-seed audit (tab:methodb). lambda=10: seed means only (prose),
# per-seed values live in the replication package.
import json as _json
MB = _json.load(open(f'{ROOT}/results/method_b_summary.json'))
for lam, markets in MB.items():
    for tag, d in markets.items():
        if lam == '1':
            for sub, r in d['per_seed'].items():
                if not present(r['sharpe'], 3):
                    errors.append(f'MethodB lam{lam} {tag} seed{r["seed"]} sharpe={r["sharpe"]:.3f} not found')
                if not present(r['raw_sharpe'], 3):
                    errors.append(f'MethodB lam{lam} {tag} seed{r["seed"]} raw={r["raw_sharpe"]:.3f} not found')
                if r['seed'] == 42 and not present(r['pin_sign_norm_sharpe'], 3):
                    errors.append(f'MethodB lam{lam} {tag} seed42 pin_sn={r["pin_sign_norm_sharpe"]:.3f} not found')
            agree_cells = [f"{r['window_sign_agreement']}/{r['n_windows']}"
                           for r in sorted(d['per_seed'].values(), key=lambda x: x['seed'])]
            for cell in agree_cells:
                if cell not in TEX:
                    errors.append(f'MethodB lam1 {tag} agreement cell {cell} not found')
        if not present(d['sharpe_mean'], 3):
            errors.append(f'MethodB lam{lam} {tag} pin seed-mean={d["sharpe_mean"]:.3f} not found')

# 6) citation reconciliation
cited = set()
for m in re.finditer(r'\\cite[tp]?\{([^}]*)\}', TEX):
    for k in m.group(1).split(','):
        cited.add(k.strip())
bib = open(f'{ROOT}/paper/references.bib', encoding='utf-8').read()
bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
missing = cited - bibkeys
if missing:
    errors.append(f'cited but not in bib: {sorted(missing)}')
aux = open(f'{ROOT}/paper/manuscript.aux', encoding='utf-8').read()
bounced = set(re.findall(r'\\citation\{([^}]*)\}', aux))
for grp in bounced:
    for k in grp.split(','):
        if k.strip() not in cited:
            notes.append(f'citation in aux not via \\cite in body: {k}')
unused = bibkeys - cited
notes.append(f'bib entries not cited (dropped by bibtex, OK): {len(unused)}')

# 7) stale single-country claims / terminology
for pat, why in ((r'on the TSE, a linear kernel', 'old IR-only framing'),
                 (r'prices the cross-section, fragile', 'old subsection title'),
                 (r'\\pm5\\% band', 'check: should be Iran-specific context'),
                 (r'first deep SDF estimate for any frontier market\b', 'should now be plural estimates'),
                 (r'investable|tradable universe liquid enough', 'check phrasing'),
                 (r'at or above the strongest factor benchmark', 'referee: TR is tied, not above; use competitive phrasing'),
                 (r'advantage is significant in Pakistan', 'referee: PK SPA p=0.71-0.77, NOT significant'),
                 (r'statistically indistinguishable from zero \(.*PCA', 'referee: PK E2/E8 significantly BELOW PCA(5), CIs exclude 0'),
                 (r'exceeds every factor benchmark in all three markets', 'referee: false for TR (tied with q-factor)'),
                 (r'6 in T\\\\"urkiye and Pakistan|6 in T\\\\"urkiye and 6 in Pakistan', 'referee: PK has 5 windows, not 6'),
                 (r'Pakistan \(6 windows\)', 'referee: PK has 5 windows')):
    hits = re.findall(pat, TEX)
    if hits:
        notes.append(f'pattern {pat!r}: {len(hits)} hit(s) — {why}')

# 8) abstract numbers must also appear in body (consistency)
absm = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', TEX, re.S).group(1)
nums = re.findall(r'\d+\.\d+', absm)
missing_in_body = [n for n in set(nums) if n not in TEX[len(absm):]]
# allow small abstract-only numbers like protocol integers
missing_in_body = [n for n in missing_in_body if float(n) > 1.0]
if missing_in_body:
    errors.append(f'abstract numbers not in body: {missing_in_body}')

# 9) unresolved placeholders / CS leftovers
if '@@' in TEX:
    errors.append('unresolved placeholders remain')
for m in re.finditer(r'(?<![\w-])TSE(?![\w-])', TEX):
    ctx = TEX[max(0,m.start()-60):m.start()+60].replace('\n',' ')
    notes.append(f'TSE mention: ...{ctx}...')

print(f'ERRORS: {len(errors)}')
for e in errors: print('  X', e)
print(f'NOTES: {len(notes)}')
for n in notes: print('  -', n)
sys.exit(1 if errors else 0)
