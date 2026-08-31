#!/usr/bin/env bash
# rerun_wave2_20260831.sh — second-wave analyses on revised panels
# (runs AFTER rerun_3c_20260831.sh completes; same corrected inputs)
set -u
cd /home/ubuntu/research/dlap-tse
V=/home/ubuntu/venvs/dlap-tse/bin/python
LOG=/tmp/rerun_wave2.log
: > $LOG

for C in IR TR PK; do
  echo "########## WAVE2 $C — $(date +%H:%M:%S) ##########" | tee -a $LOG
  export DLAP_COUNTRY=$C

  # Method B pins (E2 sy): lambda 1/10 x seeds 42/43/44
  for seed in 42 43 44; do
    S=""; [ "$seed" != "42" ] && S="--seed $seed"
    $V scripts/train_e2.py --charset sy --pin-lambda 1 $S  >> $LOG 2>&1 && echo "PIN1_s${seed}_${C}_DONE" || echo "PIN1_s${seed}_${C}_FAIL"
    $V scripts/train_e2.py --charset sy --pin-lambda 10 $S >> $LOG 2>&1 && echo "PIN10_s${seed}_${C}_DONE" || echo "PIN10_s${seed}_${C}_FAIL"
  done

  # placebos (sy): pnoisy/prandom x 3 seeds
  for seed in 42 43 44; do
    S=""; [ "$seed" != "42" ] && S="--seed $seed"
    $V scripts/train_e2.py --charset sy --drop-noisy $S  >> $LOG 2>&1 && echo "PNOISY_s${seed}_${C}_DONE" || echo "PNOISY_s${seed}_${C}_FAIL"
    $V scripts/train_e2.py --charset sy --drop-random $S >> $LOG 2>&1 && echo "PRANDOM_s${seed}_${C}_DONE" || echo "PRANDOM_s${seed}_${C}_FAIL"
  done

  # loadings for seeds 43/44 (E3 all) — seed42 already done in wave 1
  for seed in 43 44; do
    $V scripts/e6_loadings.py --charset all --seed $seed >> $LOG 2>&1 && echo "E6ALL_s${seed}_${C}_DONE" || echo "E6ALL_s${seed}_${C}_FAIL"
  done
done

# IR-only aux (all-char effects + diagnostics that appear in the paper)
echo "########## WAVE2 IR-AUX ##########" | tee -a $LOG
export DLAP_COUNTRY=IR
$V scripts/train_e2_lag.py                     >> $LOG 2>&1 && echo "LAG_IR_DONE" || echo "LAG_IR_FAIL"
$V scripts/e7_subperiod.py                     >> $LOG 2>&1 && echo "E7_IR_DONE" || echo "E7_IR_FAIL"
$V scripts/method_b_summary.py                 >> $LOG 2>&1 && echo "MB_DONE"     || echo "MB_FAIL"
$V scripts/placebo_summary.py                  >> $LOG 2>&1 && echo "PLAC_DONE"   || echo "PLAC_FAIL"
$V scripts/seed_inference_summary.py           >> $LOG 2>&1 && echo "SEEDINF_DONE" || echo "SEEDINF_FAIL"
for C in IR TR PK; do
  DLAP_COUNTRY=$C $V scripts/rms_window_bootstrap.py >> $LOG 2>&1 && echo "RMSBOOT_${C}_DONE" || echo "RMSBOOT_${C}_FAIL"
done
DLAP_COUNTRY=IR $V scripts/seed_inference_summary.py >> $LOG 2>&1 && echo "SEEDINF2_DONE" || echo "SEEDINF2_FAIL"

echo "WAVE2_ALL_DONE — $(date +%H:%M:%S)" | tee -a $LOG
