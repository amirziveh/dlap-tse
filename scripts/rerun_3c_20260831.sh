#!/usr/bin/env bash
# rerun_3c_20260831.sh — full 3-country battery on revised panels
# (investment removed everywhere; PK nsi removed as extraction artifact)
set -u
cd /home/ubuntu/research/dlap-tse
V=/home/ubuntu/venvs/dlap-tse/bin/python
LOG=/tmp/rerun_3c_20260831.log
: > $LOG

run_market() {  # run_market <COUNTRY>
  local C=$1
  echo "########## MARKET $C — $(date +%H:%M:%S) ##########" | tee -a $LOG
  export DLAP_COUNTRY=$C

  echo "--- E1 benchmarks ---" | tee -a $LOG
  $V scripts/run_e1.py >> $LOG 2>&1 && echo "E1_${C}_DONE" || echo "E1_${C}_FAIL"

  echo "--- linear SDF sy/all ---" | tee -a $LOG
  $V scripts/linear_sdf_benchmark.py --charset sy  >> $LOG 2>&1 && echo "LINSY_${C}_DONE" || echo "LINSY_${C}_FAIL"
  $V scripts/linear_sdf_benchmark.py --charset all >> $LOG 2>&1 && echo "LINALL_${C}_DONE" || echo "LINALL_${C}_FAIL"

  echo "--- deep battery 8 specs x 3 seeds ---" | tee -a $LOG
  for seed in 42 43 44; do
    S=""; [ "$seed" != "42" ] && S="--seed $seed"
    $V scripts/train_e2.py --charset sy $S                 >> $LOG 2>&1 && echo "E2_s${seed}_${C}_DONE"  || echo "E2_s${seed}_${C}_FAIL"
    $V scripts/train_e2.py --charset all $S                >> $LOG 2>&1 && echo "E3_s${seed}_${C}_DONE"  || echo "E3_s${seed}_${C}_FAIL"
    $V scripts/train_e2.py --charset all --states const $S >> $LOG 2>&1 && echo "E4A_s${seed}_${C}_DONE" || echo "E4A_s${seed}_${C}_FAIL"
    $V scripts/train_e2.py --charset sy --states const $S  >> $LOG 2>&1 && echo "E4B_s${seed}_${C}_DONE" || echo "E4B_s${seed}_${C}_FAIL"
    $V scripts/train_e2.py --charset sy --critic $S        >> $LOG 2>&1 && echo "E5A_s${seed}_${C}_DONE" || echo "E5A_s${seed}_${C}_FAIL"
    $V scripts/train_e2.py --charset all --critic $S       >> $LOG 2>&1 && echo "E5B_s${seed}_${C}_DONE" || echo "E5B_s${seed}_${C}_FAIL"
    $V scripts/train_e2.py --charset sy --liq-filter $S    >> $LOG 2>&1 && echo "E8_s${seed}_${C}_DONE"  || echo "E8_s${seed}_${C}_FAIL"
    $V scripts/train_e2.py --charset all --liq-filter $S   >> $LOG 2>&1 && echo "E8B_s${seed}_${C}_DONE" || echo "E8B_s${seed}_${C}_FAIL"
    echo "##### SEED $seed / $C COMPLETE — $(date +%H:%M:%S)" | tee -a $LOG
  done

  echo "--- post: master, loadings, bootstrap, spa ---" | tee -a $LOG
  $V scripts/build_master_results.py           >> $LOG 2>&1 && echo "MASTER_${C}_DONE" || echo "MASTER_${C}_FAIL"
  $V scripts/e6_loadings.py --charset all      >> $LOG 2>&1 || true
  $V scripts/e6_loadings.py --charset sy       >> $LOG 2>&1 || true
  $V scripts/loadings_bootstrap.py --charset all >> $LOG 2>&1 || true
  $V scripts/loadings_bootstrap.py --charset sy  >> $LOG 2>&1 || true
  $V scripts/sharp_diff_bootstrap.py --spec e2 >> $LOG 2>&1 && echo "BOOT_E2_${C}_DONE" || echo "BOOT_E2_${C}_FAIL"
  $V scripts/sharp_diff_bootstrap.py --spec e8 >> $LOG 2>&1 && echo "BOOT_E8_${C}_DONE" || echo "BOOT_E8_${C}_FAIL"
  $V scripts/spa_test.py                       >> $LOG 2>&1 && echo "SPA_${C}_DONE" || echo "SPA_${C}_FAIL"
  echo "########## MARKET $C COMPLETE — $(date +%H:%M:%S) ##########" | tee -a $LOG
}

run_market IR
run_market TR
run_market PK

echo "ALL_MARKETS_DONE — $(date +%H:%M:%S)" | tee -a $LOG
