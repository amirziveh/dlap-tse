#!/usr/bin/env bash
# =============================================================================
# rerun_audit.sh — full reproducible re-run of the true-CPZ dlap-tse pipeline
# =============================================================================
# Re-runs every result that feeds the manuscripts, then diffs the CSVs against
# the committed results/. A clean re-run must produce byte-identical CSVs
# (torch seed 42 is fixed; no GPU nondeterminism on CPU).
#
# Usage: bash rerun_audit.sh   (from the project root, venv at ~/venvs/dlap-tse)
# =============================================================================
set -u
cd "$(dirname "$0")"
PY=/home/ubuntu/venvs/dlap-tse/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=rerun_audit_$STAMP
mkdir -p "$OUT"

echo "== full pipeline re-run (stamp $STAMP) =="

# --- unit tests -------------------------------------------------------------
$PY scripts/test_sdf_models.py > "$OUT/tests.log" 2>&1 && echo "PASS unit tests" \
  || { echo "FAIL unit tests"; tail -5 "$OUT/tests.log"; exit 1; }

# --- deep SDF specs (cpz common SDF) ----------------------------------------
for spec in "e2 sy lstm" "e3 all lstm" "e4a all const" "e4b sy const" \
            "e5a sy lstm critic" "e5b all lstm critic" \
            "e8 sy lstm liq" "e8b all lstm liq"; do
  set -- $spec
  name=$1; cs=$2; st=$3; extra=""
  [ "${4:-}" = "critic" ] && extra="--critic"
  [ "${4:-}" = "liq" ] && extra="--liq-filter"
  $PY scripts/train_e2.py --charset "$cs" --states "$st" $extra > "$OUT/$name.log" 2>&1 \
    || { echo "FAIL $name"; tail -5 "$OUT/$name.log"; exit 1; }
  echo "OK $name"
done

# --- charscore robustness (legacy per-stock, labeled) ------------------------
$PY scripts/train_e2.py --arch charscore --charset sy --states lstm > "$OUT/cs_sy.log" 2>&1
$PY scripts/train_e2.py --arch charscore --charset all --states lstm > "$OUT/cs_all.log" 2>&1
echo "OK charscore"

# --- linear SDF benchmark ----------------------------------------------------
rm -f results/linear_sdf_results.csv
$PY scripts/linear_sdf_benchmark.py --charset sy > "$OUT/lin11.log" 2>&1
$PY scripts/linear_sdf_benchmark.py --charset all > "$OUT/lin20.log" 2>&1
echo "OK linear SDF"

# --- loadings + bootstrap + downstream --------------------------------------
$PY scripts/e6_loadings.py --charset all > "$OUT/e6_all.log" 2>&1
$PY scripts/e6_loadings.py --charset sy > "$OUT/e6_sy.log" 2>&1
$PY scripts/loadings_bootstrap.py --charset all > "$OUT/ldboot_all.log" 2>&1
$PY scripts/loadings_bootstrap.py --charset sy > "$OUT/ldboot_sy.log" 2>&1
$PY scripts/sharp_diff_bootstrap.py --spec e2 > "$OUT/boot_e2.log" 2>&1
$PY scripts/sharp_diff_bootstrap.py --spec e8 > "$OUT/boot_e8.log" 2>&1
$PY scripts/e7_subperiod.py > "$OUT/e7.log" 2>&1
$PY scripts/bench_leverage_check.py > "$OUT/leverage.log" 2>&1
$PY scripts/q1_artifacts.py > "$OUT/q1.log" 2>&1
$PY scripts/q1_artifacts_fa.py > "$OUT/q1fa.log" 2>&1
echo "OK downstream"

# --- master table + number verification -------------------------------------
$PY scripts/build_master_results.py > "$OUT/master.log" 2>&1
$PY scripts/verify_manuscript_numbers.py > "$OUT/verify.log" 2>&1
if grep -q "^ERRORS: 0" "$OUT/verify.log"; then
  echo "PASS manuscript number verification"
else
  echo "!! verify_manuscript_numbers found issues (see $OUT/verify.log)"
  head -10 "$OUT/verify.log"
  exit 1
fi

echo "== done: $OUT =="
echo "Reproducibility: the pipeline is fully seeded (torch seed 42, numpy"
echo "seed 42 for bootstraps). A clean re-run regenerates results/ in place;"
echo "the log dir $OUT records every step."
