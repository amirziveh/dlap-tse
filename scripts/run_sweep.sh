#!/usr/bin/env bash
# 10-item sweep for one country (DLAP_COUNTRY set by caller)
set -e
cd /home/ubuntu/research/dlap-tse
V=/home/ubuntu/venvs/dlap-tse/bin/python
echo "########## COUNTRY=$DLAP_COUNTRY ##########"

echo "--- [1] sharp-diff bootstrap E2/E8 (block 6) ---"
$V scripts/sharp_diff_bootstrap.py --spec e2 2>&1 | tail -2
$V scripts/sharp_diff_bootstrap.py --spec e8 2>&1 | tail -2

echo "--- [2] SPA test (E2 vs benchmark set) ---"
$V scripts/spa_test.py 2>&1 | tail -4

echo "--- [3] lasso lag check ---"
$V scripts/lasso_lag_check.py 2>&1 | tail -3

echo "--- [4] leverage-normalized benchmarks ---"
$V scripts/bench_leverage_check.py 2>&1 | tail -3

echo "--- [5] loadings bootstrap (sy + all) ---"
$V scripts/loadings_bootstrap.py --charset sy 2>&1 | tail -2
$V scripts/loadings_bootstrap.py --charset all 2>&1 | tail -2

echo "--- [6] e2 lag ---"
$V scripts/train_e2_lag.py 2>&1 | tail -3

echo "--- [7] placebo ---"
$V scripts/placebo_summary.py 2>&1 | tail -3

echo "SWEEP_DONE"
