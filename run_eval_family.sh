#!/usr/bin/env bash
#
# run_eval_family.sh -- evaluate every EE-velocity variant under one root.
#
# The family comes from configs.VARIANT_TABLE, the same way run_vel_family.sh
# derives it, so the two sweeps cannot disagree about what "the EE velocity
# family" means.
#
#   ./run_eval_family.sh --train-from from_20261016      the noised runs
#   ./run_eval_family.sh --train-from from_20260916      the clean baseline
#   ./run_eval_family.sh --train-from from_20261016 vel_manip_seq    just one
#   ./run_eval_family.sh --list
#   ./run_eval_family.sh -n --train-from from_20261016
#
# --train-from is REQUIRED. Every velocity variant now has a row under both
# roots, and evaluate.py deliberately refuses to pick between them -- an
# evaluation filed under the wrong root is indistinguishable from a real result.
#
# Env knobs:
#   PYTHON=python3.11 ./run_eval_family.sh ...
#   CUDA_VISIBLE_DEVICES=1 ./run_eval_family.sh ...
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
LOG_DIR="results/logs"
DRY_RUN=0
GRIPPER_ENV=""
TRAIN_FROM_ENV=""

clean_python() {
    env -u PHYPUSH_INPUT_VARIANT -u PHYPUSH_GRIPPER_CLOSED -u VEL_NOISE \
        -u TRAIN_FROM "$PYTHON" "$@"
}

# Marked rows only: importing configs PRINTS its [CONFIG] block, and parsing
# raw stdout would turn those lines into variants.
read_table() {
    clean_python - <<'PY' | awk -F'\t' '$1=="__VARIANT__" {
        printf "%s\t%s\n", $2, $3 }'
import sys
try:
    from configs import VARIANT_TABLE, VEL_PREFIX
except Exception as e:
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
for name, (chans, cond) in VARIANT_TABLE.items():
    if VEL_PREFIX not in chans:
        continue
    kind = "cond" if cond is not None else "seq"
    print(f"__VARIANT__\t{name}\t{kind}")
PY
}

TABLE="$(read_table)" || {
    echo "ERROR: could not import configs.py." >&2
    echo "       Run this from the directory holding configs.py and evaluate.py." >&2
    exit 1
}

FAMILY=()
while IFS=$'\t' read -r name _kind; do
    [ -n "$name" ] && FAMILY+=("$name")
done <<< "$TABLE"
[ ${#FAMILY[@]} -eq 0 ] && { echo "ERROR: no velocity variants found." >&2; exit 1; }

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'; }

REQUESTED=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        --list)           printf '%s\n' "${FAMILY[@]}"; exit 0 ;;
        -n|--dry-run)     DRY_RUN=1 ;;
        --all)            REQUESTED+=("${FAMILY[@]}") ;;
        --train-from)     shift; TRAIN_FROM_ENV="${1:-}" ;;
        --train-from=*)   TRAIN_FROM_ENV="${1#*=}" ;;
        --closed-gripper) GRIPPER_ENV=1 ;;
        --open-gripper)   GRIPPER_ENV=0 ;;
        -*)               echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)                REQUESTED+=("$1") ;;
    esac
    shift
done
[ ${#REQUESTED[@]} -eq 0 ] && REQUESTED=("${FAMILY[@]}")

if [ -z "$TRAIN_FROM_ENV" ]; then
    echo "ERROR: --train-from is required." >&2
    echo >&2
    echo "Every velocity variant has a row under more than one root, and" >&2
    echo "evaluate.py refuses to choose. Name the one you mean:" >&2
    echo "    ./run_eval_family.sh --train-from from_20261016   # noised" >&2
    echo "    ./run_eval_family.sh --train-from from_20260916   # clean" >&2
    exit 2
fi

BAD=()
for v in "${REQUESTED[@]}"; do
    found=0
    for a in "${FAMILY[@]}"; do [ "$v" = "$a" ] && { found=1; break; }; done
    [ $found -eq 1 ] || BAD+=("$v")
done
if [ ${#BAD[@]} -gt 0 ]; then
    echo "Not in the EE velocity family: ${BAD[*]}" >&2
    echo "Family: ${FAMILY[*]}" >&2
    exit 2
fi

VARIANTS=()
for v in "${REQUESTED[@]}"; do
    seen=0
    if [ ${#VARIANTS[@]} -gt 0 ]; then
        for s in "${VARIANTS[@]}"; do [ "$v" = "$s" ] && { seen=1; break; }; done
    fi
    [ $seen -eq 1 ] || VARIANTS+=("$v")
done

ROOT_PY="from configs import GRIPPER_CLOSED, ARM_STATS_CSV_BASE, CSV_PATH
print('__INFO__\t' + ('gripper_closed' if GRIPPER_CLOSED else 'gripper_opened')
      + '\t' + CSV_PATH.split('/')[-1] + '\t' + ARM_STATS_CSV_BASE.split('/')[-1])"
PRE_ENV=(TRAIN_FROM="$TRAIN_FROM_ENV")
[ -n "$GRIPPER_ENV" ] && PRE_ENV+=(PHYPUSH_GRIPPER_CLOSED="$GRIPPER_ENV")
INFO="$(env -u PHYPUSH_INPUT_VARIANT -u VEL_NOISE "${PRE_ENV[@]}" \
        "$PYTHON" -c "$ROOT_PY" 2>&1 | awk -F'\t' '$1=="__INFO__"{print}')" || true
if [ -z "$INFO" ]; then
    echo "ERROR: configs.py rejected TRAIN_FROM='$TRAIN_FROM_ENV'." >&2
    env -u PHYPUSH_INPUT_VARIANT -u VEL_NOISE "${PRE_ENV[@]}" \
        "$PYTHON" -c "$ROOT_PY" 2>&1 | tail -3 >&2
    exit 1
fi
GRIPPER_FOLDER="$(printf '%s' "$INFO" | cut -f2)"
EVAL_CSV="$(printf '%s' "$INFO" | cut -f3)"
EVAL_STATS="$(printf '%s' "$INFO" | cut -f4)"

mkdir -p "$LOG_DIR"

echo "=============================================================="
echo " EE VELOCITY FAMILY EVALUATION"
echo "=============================================================="
echo "   variants   : ${#VARIANTS[@]}   ${VARIANTS[*]}"
echo "   train_from : $TRAIN_FROM_ENV"
echo "   gripper    : $GRIPPER_FOLDER${GRIPPER_ENV:+  (forced)}"
echo "   data       : $EVAL_CSV"
echo "   stats      : $EVAL_STATS"
echo "   logs       : $LOG_DIR/"
[ "$DRY_RUN" -eq 1 ] && echo "   DRY RUN -- nothing will be evaluated"
echo "=============================================================="

OK=(); FAILED=()
START=$SECONDS
n=0
for v in "${VARIANTS[@]}"; do
    n=$(( n + 1 ))
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="$LOG_DIR/eval_${stamp}_${v}_${TRAIN_FROM_ENV}.log"

    echo
    echo "--------------------------------------------------------------"
    echo ">>> [$n/${#VARIANTS[@]}] $v   ($(date '+%F %T'))"
    echo "    log: $log"
    echo "--------------------------------------------------------------"

    ENV_ARGS=(PHYPUSH_INPUT_VARIANT="$v" TRAIN_FROM="$TRAIN_FROM_ENV")
    [ -n "$GRIPPER_ENV" ] && ENV_ARGS+=(PHYPUSH_GRIPPER_CLOSED="$GRIPPER_ENV")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    would run: env ${ENV_ARGS[*]} $PYTHON evaluate.py"
        OK+=("$v")
        continue
    fi

    env "${ENV_ARGS[@]}" "$PYTHON" evaluate.py 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        OK+=("$v")
        # Confirm which checkpoint it actually read, from evaluate.py's own
        # output -- the whole risk here is evaluating the wrong root's model.
        used=$(grep -m1 '^\[RUN\] ' "$log" | sed 's/^\[RUN\] //')
        [ -n "$used" ] && echo "    $used"
        grep -m1 'model dir carries a run tag' "$log" | sed 's/^/    /'
    else
        FAILED+=("$v(exit $rc)")
        echo "<<< $v FAILED with exit code $rc -- see $log"
    fi
done

echo
echo "=============================================================="
echo " EVAL SUMMARY   ($(( (SECONDS - START) / 60 ))m)   root $TRAIN_FROM_ENV"
echo "=============================================================="
echo "  succeeded : ${#OK[@]}/${#VARIANTS[@]}"
[ ${#OK[@]} -gt 0 ] && echo "              ${OK[*]}"
echo "  failed    : ${#FAILED[@]}"
[ ${#FAILED[@]} -gt 0 ] && echo "              ${FAILED[*]}"
echo "=============================================================="

[ ${#FAILED[@]} -eq 0 ]