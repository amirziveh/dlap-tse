#!/usr/bin/env bash
# PK battery re-run on enriched data: e2/e3/e4a/e4b/e5a/e5b/e8/e8b x seeds 42/43/44
set -e
cd /home/ubuntu/research/dlap-tse
export DLAP_COUNTRY=PK
V=/home/ubuntu/venvs/dlap-tse/bin/python

run() {  # run <spec-args> <name>
  echo "=== $2 ==="
  $V scripts/train_e2.py $1 2>&1 | tail -1
}

for seed in 42 43 44; do
  echo "########## SEED $seed ##########"
  S=""; if [ "$seed" != "42" ]; then S="--seed $seed"; fi
  run "--charset sy $S"                 "E2 s$seed"
  run "--charset all $S"                "E3 s$seed"
  run "--charset all --states const $S" "E4a s$seed"
  run "--charset sy --states const $S"  "E4b s$seed"
  run "--charset sy --critic $S"        "E5a s$seed"
  run "--charset all --critic $S"       "E5b s$seed"
  run "--charset sy --liq-filter $S"    "E8 s$seed"
  run "--charset all --liq-filter $S"   "E8b s$seed"
done
$V scripts/seed_sensitivity.py 2>&1 | tail -32
echo BATTERY_DONE
