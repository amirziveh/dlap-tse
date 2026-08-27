#!/usr/bin/env bash
# placebo runs for one country (pnoisy + prandom, seeds 42/43/44, liq-filter spec as in IR)
set -e
cd /home/ubuntu/research/dlap-tse
V=/home/ubuntu/venvs/dlap-tse/bin/python
for seed in 42 43 44; do
  S=""; if [ "$seed" != "42" ]; then S="--seed $seed"; fi
  echo "== pnoisy s$seed =="
  $V scripts/train_e2.py --charset sy --drop-noisy $S 2>&1 | tail -1
  echo "== prandom s$seed =="
  $V scripts/train_e2.py --charset sy --drop-random $S 2>&1 | tail -1
done
echo PLACEBO_RUNS_DONE
