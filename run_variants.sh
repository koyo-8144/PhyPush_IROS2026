#!/usr/bin/env bash
#
# run_variants.sh -- train one or more INPUT_VARIANTs, one after another.
#
# The variant reaches configs.py through PHYPUSH_INPUT_VARIANT rather than by
# rewriting configs.py, so a sweep leaves your source untouched and two runs can
# never fight over the file.
#
#   ./run_variants.sh eff_only                   one variant
#   ./run_variants.sh osim_only osim_manip       several, in the order given
#   ./run_variants.sh --all                      every variant in VARIANT_TABLE
#   ./run_variants.sh --velocity-free            the six that drop velocity
#   ./run_variants.sh --list                     show the names and exit
#   ./run_variants.sh -n --all                   print what would run, run nothing
#   ./run_variants.sh --open-gripper vel_only    train against the opened-gripper CSV
#
# Each run's stdout+stderr goes to results/logs/<timestamp>_<variant>.log as well
# as to the terminal. A failed run does NOT stop the sweep: the summary lists what
# failed and the script exits non-zero.
#
# Env knobs:
#   PYTHON=python3.11 ./run_variants.sh ...       interpreter to use
#   CUDA_VISIBLE_DEVICES=1 ./run_variants.sh ...  passed straight through
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
LOG_DIR="results/logs"
DRY_RUN=0
GRIPPER_ENV=""          # "", "0" or "1"

# Run python with our own overrides cleared, so a stale exported value in the
# shell cannot change what the script reports.
clean_python() {
    env -u PHYPUSH_INPUT_VARIANT -u PHYPUSH_GRIPPER_CLOSED "$PYTHON" "$@"
}

# -----------------------------------------------------------------------------
# The valid names come from configs.VARIANT_TABLE, never from a list pasted into
# this script -- a paste would drift the first time a variant is added.
# -----------------------------------------------------------------------------
read_table() {
    clean_python - <<'PY'
import sys
try:
    from configs import VARIANT_TABLE, VEL_PREFIX
except Exception as e:                       # surfaced to the user below
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
for name, (chans, cond) in VARIANT_TABLE.items():
    extra = [c for c in chans if c != VEL_PREFIX]
    needs = extra + ([cond] if cond is not None else [])
    kind = "cond" if cond is not None else ("vel" if VEL_PREFIX in chans else "novel")
    print(f"{name}\t{kind}\t{len(chans)}\t{','.join(needs) or '-'}")
PY
}

TABLE="$(read_table)" || {
    echo "ERROR: could not import configs.py." >&2
    echo "       Run this from the directory holding configs.py and train.py," >&2
    echo "       with the right conda/venv environment activated." >&2
    exit 1
}

ALL_VARIANTS=()
while IFS=$'\t' read -r name _kind _dim _needs; do
    [ -n "$name" ] && ALL_VARIANTS+=("$name")
done <<< "$TABLE"

if [ ${#ALL_VARIANTS[@]} -eq 0 ]; then
    echo "ERROR: configs.VARIANT_TABLE came back empty." >&2
    exit 1
fi

print_list() {
    printf '%-20s %-7s %-7s %s\n' "variant" "kind" "in_dim" "channels beside velocity"
    printf '%s\n' "---------------------------------------------------------------------"
    while IFS=$'\t' read -r name kind dim needs; do
        [ -n "$name" ] || continue
        printf '%-20s %-7s %-7s %s\n' "$name" "$kind" "$dim" "$needs"
    done <<< "$TABLE"
    echo
    echo "kind:  vel   = velocity + extra channels"
    echo "       cond  = velocity + a static conditioning scalar"
    echo "       novel = NO velocity input (the arm's own signature alone)"
}

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'; }

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
REQUESTED=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        --list)           print_list; exit 0 ;;
        -n|--dry-run)     DRY_RUN=1 ;;
        --all)            REQUESTED+=("${ALL_VARIANTS[@]}") ;;
        --velocity-free)
            while IFS=$'\t' read -r name kind _dim _needs; do
                [ "$kind" = "novel" ] && REQUESTED+=("$name")
            done <<< "$TABLE" ;;
        --closed-gripper) GRIPPER_ENV=1 ;;
        --open-gripper)   GRIPPER_ENV=0 ;;
        -*)               echo "Unknown option: $1" >&2; echo >&2; usage >&2; exit 2 ;;
        *)                REQUESTED+=("$1") ;;
    esac
    shift
done

if [ ${#REQUESTED[@]} -eq 0 ]; then
    echo "No variant given." >&2; echo >&2
    print_list >&2; echo >&2
    usage >&2
    exit 2
fi

# Validate every name BEFORE starting, so a typo in position 7 of a sweep does
# not surface six hours in.
BAD=()
for v in "${REQUESTED[@]}"; do
    found=0
    for a in "${ALL_VARIANTS[@]}"; do [ "$v" = "$a" ] && { found=1; break; }; done
    [ $found -eq 1 ] || BAD+=("$v")
done
if [ ${#BAD[@]} -gt 0 ]; then
    echo "Unknown variant(s): ${BAD[*]}" >&2; echo >&2
    print_list >&2
    exit 2
fi

# De-duplicate, preserving the order given.
VARIANTS=()
for v in "${REQUESTED[@]}"; do
    seen=0
    if [ ${#VARIANTS[@]} -gt 0 ]; then
        for s in "${VARIANTS[@]}"; do [ "$v" = "$s" ] && { seen=1; break; }; done
    fi
    [ $seen -eq 1 ] || VARIANTS+=("$v")
done

# Which results/checkpoints/<...>/<gripper>/ folder these runs will land in --
# asked of configs.py under the same override the runs will use, rather than
# assumed, so the summary cannot name the wrong one.
GRIPPER_PY="from configs import GRIPPER_CLOSED
print('gripper_closed' if GRIPPER_CLOSED else 'gripper_opened')"
if [ -n "$GRIPPER_ENV" ]; then
    GRIPPER_FOLDER="$(env -u PHYPUSH_INPUT_VARIANT \
        PHYPUSH_GRIPPER_CLOSED="$GRIPPER_ENV" "$PYTHON" -c "$GRIPPER_PY" 2>/dev/null)"
else
    GRIPPER_FOLDER="$(clean_python -c "$GRIPPER_PY" 2>/dev/null)"
fi
GRIPPER_FOLDER="${GRIPPER_FOLDER:-gripper_closed}"

mkdir -p "$LOG_DIR"

echo "=============================================================="
echo " TRAINING SWEEP: ${#VARIANTS[@]} variant(s)"
echo "   ${VARIANTS[*]}"
echo "   python  : $PYTHON"
echo "   gripper : $GRIPPER_FOLDER${GRIPPER_ENV:+  (forced via PHYPUSH_GRIPPER_CLOSED=$GRIPPER_ENV)}"
echo "   logs    : $LOG_DIR/"
[ "$DRY_RUN" -eq 1 ] && echo "   DRY RUN -- nothing will be trained"
echo "=============================================================="

OK=(); FAILED=(); CKPTS=()
SWEEP_START=$SECONDS
n=0

for v in "${VARIANTS[@]}"; do
    n=$(( n + 1 ))
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="$LOG_DIR/${stamp}_${v}.log"

    echo
    echo "--------------------------------------------------------------"
    echo ">>> [$n/${#VARIANTS[@]}] $v   ($(date '+%F %T'))"
    echo "    log: $log"
    echo "--------------------------------------------------------------"

    ENV_ARGS=(PHYPUSH_INPUT_VARIANT="$v")
    [ -n "$GRIPPER_ENV" ] && ENV_ARGS+=(PHYPUSH_GRIPPER_CLOSED="$GRIPPER_ENV")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    would run: env ${ENV_ARGS[*]} $PYTHON train.py"
        OK+=("$v")
        continue
    fi

    start=$SECONDS
    # PIPESTATUS, not $?, because $? would be tee's exit code.
    env "${ENV_ARGS[@]}" "$PYTHON" train.py 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    mins=$(( (SECONDS - start) / 60 ))

    if [ "$rc" -eq 0 ]; then
        OK+=("$v")
        echo "<<< $v finished in ${mins}m"
        # train.py prints the checkpoint dir; lift the run timestamp out of it so
        # it can go straight into evaluate.py's RUN_TIMESTAMPS.
        ck=$(grep -m1 'Configuration saved to:' "$log" \
             | sed 's/.*saved to: *//; s#/config\.json##')
        [ -n "$ck" ] && CKPTS+=("$v|$(basename "$(dirname "$ck")")|$ck")
    else
        FAILED+=("$v(exit $rc)")
        echo "<<< $v FAILED after ${mins}m with exit code $rc -- see $log"
    fi
done

# -----------------------------------------------------------------------------
echo
echo "=============================================================="
echo " SWEEP SUMMARY   ($(( (SECONDS - SWEEP_START) / 60 ))m total)"
echo "=============================================================="
echo "  succeeded : ${#OK[@]}"
[ ${#OK[@]} -gt 0 ] && echo "              ${OK[*]}"
echo "  failed    : ${#FAILED[@]}"
[ ${#FAILED[@]} -gt 0 ] && echo "              ${FAILED[*]}"

if [ ${#CKPTS[@]} -gt 0 ]; then
    echo
    echo "  Paste into RUN_TIMESTAMPS in evaluate.py:"
    for row in "${CKPTS[@]}"; do
        IFS='|' read -r rv ts _dir <<< "$row"
        printf '    ("%s", "%s"): "%s",\n' "$GRIPPER_FOLDER" "$rv" "$ts"
    done
    echo
    echo "  Full checkpoint directories:"
    for row in "${CKPTS[@]}"; do
        IFS='|' read -r rv _ts dir <<< "$row"
        printf '    %-20s %s\n' "$rv" "$dir"
    done
fi
echo "=============================================================="

[ ${#FAILED[@]} -eq 0 ]