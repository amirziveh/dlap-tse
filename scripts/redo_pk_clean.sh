#!/usr/bin/env bash
# PK clean-subsample + PK common-calendar, done correctly:
# mask the verified git-restored PK NPZ in place (preserves z-scores), run, restore.
set -uo pipefail
cd ~/research/dlap-tse
PY=/home/ubuntu/venvs/dlap-tse/bin/python

$PY - <<'EOF'
import numpy as np, csv

# ---------- 1. PK clean subsample (mask in place) ----------
d = np.load('data_pk/Char_all.npz', allow_pickle=True)
X = d['data']; dates = list(d['date']); tickers = list(d['ticker'])
orig = X.copy()
np.save('/tmp/pk_orig_snapshot.npy', orig)

# coverage per (stock, month) across the 19 characteristics
with np.errstate(invalid='ignore'):
    n_obs = np.sum(np.isfinite(X), axis=2)   # (T, N)
mask = n_obs >= 15
masked = np.where(mask[:, :, None], X, np.nan)
kept = int(mask.sum()); total = mask.size
print(f"PK clean mask: keep {kept}/{total} stock-months ({100*kept/total:.1f}%)")
np.savez_compressed('data_pk/Char_all.npz', data=masked, date=d['date'],
                    ticker=d['ticker'], variable=d['variable'])
print("Masked NPZ in place.")
EOF
echo "### running PK clean E2 (expect to overwrite results_pk root temporarily)"
DLAP_COUNTRY=PK $PY scripts/train_e2.py --charset sy --states lstm --seed 42 2>&1 | grep pooled
mkdir -p results_pk/clean_subsample
cp results_pk/e2_results.csv results_pk/clean_subsample/e2_results.csv
echo "saved -> results_pk/clean_subsample/e2_results.csv"
echo "### running PK clean E8"
DLAP_COUNTRY=PK $PY scripts/train_e2.py --charset sy --states lstm --seed 42 --liq-filter 2>&1 | grep pooled
cp results_pk/e8_results.csv results_pk/clean_subsample/e8_results.csv

# restore original PK panel + verify hash
$PY - <<'EOF'
import numpy as np, hashlib
orig = np.load('/tmp/pk_orig_snapshot.npy')
d = np.load('data_pk/Char_all.npz', allow_pickle=True)
np.savez_compressed('data_pk/Char_all.npz', data=orig, date=d['date'],
                    ticker=d['ticker'], variable=d['variable'])
import subprocess
EOF
md5sum -c /tmp/panels_before.md5 2>/dev/null | grep pk || true
md5sum data_pk/Char_all.npz
grep pk /tmp/panels_before.md5
echo "### CLEAN DONE"
