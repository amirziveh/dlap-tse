#!/usr/bin/env bash
# Batch seed expansion: seeds 100-109, E2+E8, IR/TR/PK
# Output: results{,_pk,_tr}/seed{100..109}/e{2,8}_*.csv
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=$(seq 100 109)
COUNTRIES=("IR" "TR" "PK")
SPECS=("e2" "e8")

total=0
for country in "${COUNTRIES[@]}"; do
  for seed in $SEEDS; do
    for spec in "${SPECS[@]}"; do
      total=$((total + 1))
    done
  done
done

echo "=== Seed expansion: $total runs ==="
count=0
for country in "${COUNTRIES[@]}"; do
  for seed in $SEEDS; do
    for spec in "${SPECS[@]}"; do
      count=$((count + 1))
      extra=""
      if [ "$spec" = "e8" ]; then extra="--liq-filter"; fi
      echo "[$count/$total] $country seed=$seed $spec"
      DLAP_COUNTRY="$country" /home/ubuntu/venvs/dlap-tse/bin/python \
        scripts/train_e2.py --charset sy --states lstm --seed "$seed" $extra \
        2>&1 | grep -E "pooled|window [0-9].*\[" | head -20
    done
  done
done
echo "=== ALL DONE ==="
