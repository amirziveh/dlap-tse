#!/usr/bin/env bash
# Verification reruns: reproduce committed baselines + spot-check batch seeds
set -uo pipefail
cd ~/research/dlap-tse
PY=/home/ubuntu/venvs/dlap-tse/bin/python

# Hash panels BEFORE (must equal AFTER)
md5sum data/Char_all.npz data_tr/Char_all.npz data_pk/Char_all.npz > /tmp/panels_before.md5

echo "### V1: IR seed42 E2 (expect sharpe_pooled=0.0160, n_win=12)"
DLAP_COUNTRY=IR $PY scripts/train_e2.py --charset sy --states lstm --seed 42 2>&1 | grep "pooled"
echo "IR e2 file: $(grep sharpe_pooled results/seed42_check/e2_results.csv 2>/dev/null || true)"
echo "### V2: TR seed42 E2 (expect 0.2323, n_win=6)"
DLAP_COUNTRY=TR $PY scripts/train_e2.py --charset sy --states lstm --seed 42 2>&1 | grep "pooled"
echo "### V3: PK seed42 E2 (expect -0.0972, n_win=6)"
DLAP_COUNTRY=PK $PY scripts/train_e2.py --charset sy --states lstm --seed 42 2>&1 | grep "pooled"
echo "### V4: IR seed107 E8 spot-check (batch file says sharpe=0.1112)"
DLAP_COUNTRY=IR $PY scripts/train_e2.py --charset sy --states lstm --seed 107 --liq-filter 2>&1 | grep "pooled"
echo "IR s107 e8 file: $(grep sharpe_pooled results/seed107/e8_results.csv | tr -d '\r')"
echo "### V5: TR seed103 E2 spot-check (batch file says sharpe=0.2224)"
DLAP_COUNTRY=TR $PY scripts/train_e2.py --charset sy --states lstm --seed 103 2>&1 | grep "pooled"
echo "TR s103 e2 file: $(grep sharpe_pooled results_tr/seed103/e2_results.csv | tr -d '\r')"

md5sum -c /tmp/panels_before.md5 && echo "PANELS UNCHANGED" || echo "PANEL CHANGED — INVESTIGATE"
echo "### VERIFY DONE"
