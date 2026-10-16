#!/usr/bin/env bash
#
# run_vel_family.sh -- train every EE-VELOCITY variant, with or without the
# fitted observation noise.
#
# The family is every entry in configs.VARIANT_TABLE whose channel list
# contains VEL_PREFIX, read from configs.py rather than pasted here, so adding
# a variant to the table adds it to this sweep automatically. The velocity-free
# variants (osim_*, eff_*) are excluded by construction: they never feed
# velocity to the model, so noise injection is a no-op for them and a "noised"
# run of one would be byte-identical to a clean one.
#
#   ./run_vel_family.sh                      the family, noise ON
#   ./run_vel_family.sh --noise off          the family, clean
#   ./run_vel_family.sh --both               clean AND noised, paired
#   ./run_vel_family.sh --noise 0.0018       a sigma by hand, in m/s
#   ./run_vel_family.sh vel_manip_seq vel_only    a subset of the family
#   ./run_vel_family.sh --list               show the family and exit
#   ./run_vel_family.sh -n --both            print what would run, run nothing
#   ./run_vel_family.sh --train-from from_20260916
#   ./run_vel_family.sh --open-gripper
#
# Each run's stdout+stderr goes to results/logs/<timestamp>_<variant>_<tag>.log
# as well as to the terminal. A failed run does NOT stop the sweep: the summary
# lists what failed and the script exits non-zero.
#
# Env knobs:
#   PYTHON=python3.11 ./run_vel_family.sh ...       interpreter to use
#   CUDA_VISIBLE_DEVICES=1 ./run_vel_family.sh ...  passed straight through
#   VEL_NOISE_IN_VAL=0 ./run_vel_family.sh ...      leave validation clean
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
LOG_DIR="results/logs"
DRY_RUN=0
GRIPPER_ENV=""          # "", "0" or "1"
TRAIN_FROM_ENV=""
NOISE_MODE="on"         # on | off | <sigma in m/s>
BOTH=0

clean_python() {
    env -u PHYPUSH_INPUT_VARIANT -u PHYPUSH_GRIPPER_CLOSED -u VEL_NOISE \
        -u TRAIN_FROM "$PYTHON" "$@"
}

# -----------------------------------------------------------------------------
# The family comes from configs.VARIANT_TABLE, never from a list in this script.
# -----------------------------------------------------------------------------
# Rows carry a marker and the shell keeps only marked lines. Importing configs
# PRINTS -- the [CONFIG] block whenever TRAIN_FROM resolves a ROOT_SPEC entry --
# and parsing raw stdout turned each of those lines into a "variant", which then
# became four doomed runs at the top of the sweep. Anything configs chooses to
# print from now on is ignored instead of becoming data.
read_table() {
    clean_python - <<'PY' | awk -F'\t' '$1=="__VARIANT__" {
        printf "%s\t%s\t%s\t%s\n", $2, $3, $4, $5 }'
import sys
try:
    from configs import VARIANT_TABLE, VEL_PREFIX
except Exception as e:                       # surfaced to the user below
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
for name, (chans, cond) in VARIANT_TABLE.items():
    if VEL_PREFIX not in chans:
        continue                             # velocity-free: not this family
    extra = [c for c in chans if c != VEL_PREFIX]
    needs = extra + ([cond] if cond is not None else [])
    kind = "cond" if cond is not None else "seq"
    print(f"__VARIANT__\t{name}\t{kind}\t{len(chans)}\t{','.join(needs) or '-'}")
PY
}

TABLE="$(read_table)" || {
    echo "ERROR: could not import configs.py." >&2
    echo "       Run this from the directory holding configs.py and train.py," >&2
    echo "       with the right conda/venv environment activated." >&2
    exit 1
}

FAMILY=()
while IFS=$'\t' read -r name _kind _dim _needs; do
    [ -n "$name" ] && FAMILY+=("$name")
done <<< "$TABLE"

if [ ${#FAMILY[@]} -eq 0 ]; then
    echo "ERROR: no velocity-based variants found in configs.VARIANT_TABLE." >&2
    exit 1
fi

print_list() {
    printf '%-22s %-6s %-7s %s\n' "variant" "kind" "in_dim" "channels beside velocity"
    printf '%s\n' "----------------------------------------------------------------------"
    while IFS=$'\t' read -r name kind dim needs; do
        [ -n "$name" ] || continue
        printf '%-22s %-6s %-7s %s\n' "$name" "$kind" "$dim" "$needs"
    done <<< "$TABLE"
    echo
    echo "kind:  seq  = velocity + extra per-step channel(s)"
    echo "       cond = velocity + a static conditioning scalar"
    echo
    echo "Velocity-free variants (osim_*, eff_*) are NOT in this family: they do"
    echo "not feed velocity to the model, so the noise cannot reach them."
}

usage() { sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'; }

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
REQUESTED=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        --list)           print_list; exit 0 ;;
        -n|--dry-run)     DRY_RUN=1 ;;
        --all)            REQUESTED+=("${FAMILY[@]}") ;;
        --noise)          shift; NOISE_MODE="${1:-on}" ;;
        --noise=*)        NOISE_MODE="${1#*=}" ;;
        --both)           BOTH=1 ;;
        --train-from)     shift; TRAIN_FROM_ENV="${1:-}" ;;
        --train-from=*)   TRAIN_FROM_ENV="${1#*=}" ;;
        --closed-gripper) GRIPPER_ENV=1 ;;
        --open-gripper)   GRIPPER_ENV=0 ;;
        -*)               echo "Unknown option: $1" >&2; echo >&2; usage >&2; exit 2 ;;
        *)                REQUESTED+=("$1") ;;
    esac
    shift
done

# No names given means the whole family -- that is what this script is for.
[ ${#REQUESTED[@]} -eq 0 ] && REQUESTED=("${FAMILY[@]}")

# Validate BEFORE starting, so a typo in position 5 of a sweep does not surface
# six hours in. A velocity-free name is rejected with its own message, because
# "unknown variant" would be wrong and misleading -- it exists, it just cannot
# carry this noise.
BAD=(); NOVEL=()
for v in "${REQUESTED[@]}"; do
    found=0
    for a in "${FAMILY[@]}"; do [ "$v" = "$a" ] && { found=1; break; }; done
    if [ $found -eq 0 ]; then
        if clean_python -c "
import sys
from configs import VARIANT_TABLE
sys.exit(0 if '$v' in VARIANT_TABLE else 1)" >/dev/null 2>&1; then
            NOVEL+=("$v")
        else
            BAD+=("$v")
        fi
    fi
done
if [ ${#NOVEL[@]} -gt 0 ]; then
    echo "These are velocity-FREE variants and are not in the EE velocity family:" >&2
    echo "    ${NOVEL[*]}" >&2
    echo >&2
    echo "They do not feed velocity to the model, so observation noise cannot" >&2
    echo "reach them and a 'noised' run would be identical to a clean one." >&2
    echo "Train them with run_variants.sh instead." >&2
    exit 2
fi
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

# -----------------------------------------------------------------------------
# Noise passes
#
# --both runs the family CLEAN first, then noised. Clean first on purpose: if
# the sweep is interrupted half way, the half that finished is the baseline,
# and a noised run with nothing to compare it against answers no question at
# all. The two passes write to different checkpoint names (train.py tags the
# run with the sigma), so neither overwrites the other.
# -----------------------------------------------------------------------------
PASSES=()
if [ "$BOTH" -eq 1 ]; then
    PASSES=("off" "$NOISE_MODE")
    if [ "$NOISE_MODE" = "off" ]; then
        echo "ERROR: --both with --noise off would run the same clean sweep twice." >&2
        exit 2
    fi
else
    PASSES=("$NOISE_MODE")
fi

# -----------------------------------------------------------------------------
# PREFLIGHT
#
# Resolve the noise model HERE, once, before anything trains. train.py would
# raise on a missing json anyway -- but only after load_dataset_csv has chewed
# through a 15 GB CSV, and then once per variant. Refusing up front costs a
# second and saves the obvious version of this mistake.
# -----------------------------------------------------------------------------
SIGMA_DESC=""
PRE_NOTES=""
for p in "${PASSES[@]}"; do
    case "$p" in
        off|0|false|"") continue ;;
        on|1|true)
            # from_json also prints its own notes about the fit (a rising or
            # coloured floor), so only the LAST line is the sigma. Keeping the
            # whole capture for the failure path, where those lines are the
            # explanation.
            PRE_OUT="$(clean_python - <<'PY' 2>&1
import os, sys
try:
    from vel_noise import VelNoise
except Exception as e:
    print(f"cannot import vel_noise.py: {type(e).__name__}: {e}"); sys.exit(1)
path = os.environ.get("VEL_NOISE_JSON", "vel_noise_model.json")
try:
    vn = VelNoise.from_json(path)
except Exception as e:
    print(f"{type(e).__name__}: {e}"); sys.exit(1)
print(f"{vn.sigma:.6f} m/s  (tag {vn.tag()}, from {path})")
PY
)" || {
                echo "ERROR: the noise model could not be loaded, so nothing was trained." >&2
                printf '       %s\n' "$PRE_OUT" >&2
                exit 1
            }
            SIGMA_DESC="$(printf '%s' "$PRE_OUT" | tail -1)"
            PRE_NOTES="$(printf '%s' "$PRE_OUT" | sed '$d')" ;;
        *)
            if ! clean_python -c "
import sys
try:
    s = float('$p')
except ValueError:
    sys.exit(1)
sys.exit(0 if s > 0 else 1)" >/dev/null 2>&1; then
                echo "ERROR: --noise expects 'on', 'off', or a positive sigma in m/s;" >&2
                echo "       got '$p'." >&2
                exit 2
            fi
            SIGMA_DESC="$p m/s  (given on the command line)" ;;
    esac
done

# Which results/checkpoints/<root>/<gripper>/ these land in -- asked of
# configs.py under the same overrides the runs will use, rather than assumed.
ROOT_PY="from configs import GRIPPER_CLOSED, TRAIN_FROM
print(('gripper_closed' if GRIPPER_CLOSED else 'gripper_opened') + '\t' + TRAIN_FROM)"
PRE_ENV=()
[ -n "$GRIPPER_ENV" ] && PRE_ENV+=(PHYPUSH_GRIPPER_CLOSED="$GRIPPER_ENV")
[ -n "$TRAIN_FROM_ENV" ] && PRE_ENV+=(TRAIN_FROM="$TRAIN_FROM_ENV")
if [ ${#PRE_ENV[@]} -gt 0 ]; then
    ROOT_INFO="$(env -u PHYPUSH_INPUT_VARIANT -u VEL_NOISE "${PRE_ENV[@]}" \
        "$PYTHON" -c "$ROOT_PY" 2>/dev/null | tail -1)"
else
    ROOT_INFO="$(clean_python -c "$ROOT_PY" 2>/dev/null | tail -1)"
fi
GRIPPER_FOLDER="$(printf '%s' "$ROOT_INFO" | cut -f1)"
CKPT_ROOT="$(printf '%s' "$ROOT_INFO" | cut -f2)"
GRIPPER_FOLDER="${GRIPPER_FOLDER:-gripper_closed}"
CKPT_ROOT="${CKPT_ROOT:-unknown}"

mkdir -p "$LOG_DIR"

TOTAL=$(( ${#VARIANTS[@]} * ${#PASSES[@]} ))

echo "=============================================================="
echo " EE VELOCITY FAMILY SWEEP"
echo "=============================================================="
echo "   variants : ${#VARIANTS[@]}   ${VARIANTS[*]}"
echo "   passes   : ${PASSES[*]}"
echo "   runs     : $TOTAL"
[ -n "$SIGMA_DESC" ] && echo "   sigma    : $SIGMA_DESC"
[ -n "$PRE_NOTES" ] && printf '   %s\n' "$PRE_NOTES"
echo "   python   : $PYTHON"
echo "   gripper  : $GRIPPER_FOLDER${GRIPPER_ENV:+  (forced)}"
echo "   ckpt root: results/checkpoints/$CKPT_ROOT/$GRIPPER_FOLDER/"
echo "   logs     : $LOG_DIR/"
[ "$DRY_RUN" -eq 1 ] && echo "   DRY RUN -- nothing will be trained"
echo "=============================================================="

OK=(); FAILED=(); CKPTS=()
SWEEP_START=$SECONDS
n=0

for pass in "${PASSES[@]}"; do
    case "$pass" in
        off|0|false|"") tag="clean" ;;
        on|1|true)      tag="noise" ;;
        *)              tag="noise" ;;
    esac

    if [ ${#PASSES[@]} -gt 1 ]; then
        echo
        echo "##############################################################"
        echo "## PASS: $tag   (VEL_NOISE=$pass)"
        echo "##############################################################"
    fi

    for v in "${VARIANTS[@]}"; do
        n=$(( n + 1 ))
        stamp="$(date +%Y%m%d_%H%M%S)"
        log="$LOG_DIR/${stamp}_${v}_${tag}.log"

        echo
        echo "--------------------------------------------------------------"
        echo ">>> [$n/$TOTAL] $v   [$tag]   ($(date '+%F %T'))"
        echo "    log: $log"
        echo "--------------------------------------------------------------"

        ENV_ARGS=(PHYPUSH_INPUT_VARIANT="$v" VEL_NOISE="$pass")
        [ -n "$GRIPPER_ENV" ] && ENV_ARGS+=(PHYPUSH_GRIPPER_CLOSED="$GRIPPER_ENV")
        [ -n "$TRAIN_FROM_ENV" ] && ENV_ARGS+=(TRAIN_FROM="$TRAIN_FROM_ENV")

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "    would run: env ${ENV_ARGS[*]} $PYTHON train.py"
            OK+=("$v[$tag]")
            continue
        fi

        start=$SECONDS
        # PIPESTATUS, not $?, because $? would be tee's exit code.
        env "${ENV_ARGS[@]}" "$PYTHON" train.py 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        mins=$(( (SECONDS - start) / 60 ))

        if [ "$rc" -eq 0 ]; then
            OK+=("$v[$tag]")
            echo "<<< $v [$tag] finished in ${mins}m"
            # train.py prints the checkpoint dir; lift the run timestamp out so
            # it can go straight into evaluate.py's RUN_TIMESTAMPS.
            ck=$(grep -m1 'Configuration saved to:' "$log" \
                 | sed 's/.*saved to: *//; s#/config\.json##')
            [ -n "$ck" ] && CKPTS+=("$tag|$v|$(basename "$(dirname "$ck")")|$ck")
            # The sigma train.py actually used, read back from its own output
            # rather than assumed -- this is where a no-op injection would show.
            used=$(grep -m1 '^\[vel_noise\] sigma' "$log" | awk '{print $3}')
            noop=$(grep -c 'NO-OP' "$log")
            if [ "$tag" = "noise" ]; then
                if [ "$noop" -gt 0 ]; then
                    echo "    [!] this run reported the injection as a NO-OP"
                elif [ -n "$used" ]; then
                    echo "    injected sigma: $used m/s"
                fi
            fi
        else
            FAILED+=("$v[$tag](exit $rc)")
            echo "<<< $v [$tag] FAILED after ${mins}m with exit code $rc -- see $log"
        fi
    done
done

# -----------------------------------------------------------------------------
echo
echo "=============================================================="
echo " SWEEP SUMMARY   ($(( (SECONDS - SWEEP_START) / 60 ))m total)"
echo "=============================================================="
echo "  succeeded : ${#OK[@]}/$TOTAL"
[ ${#OK[@]} -gt 0 ] && echo "              ${OK[*]}"
echo "  failed    : ${#FAILED[@]}"
[ ${#FAILED[@]} -gt 0 ] && echo "              ${FAILED[*]}"

if [ ${#CKPTS[@]} -gt 0 ]; then
    for pass_tag in clean noise; do
        rows=()
        for row in "${CKPTS[@]}"; do
            IFS='|' read -r rt _rv _ts _dir <<< "$row"
            [ "$rt" = "$pass_tag" ] && rows+=("$row")
        done
        [ ${#rows[@]} -eq 0 ] && continue
        echo
        echo "  RUN_TIMESTAMPS for evaluate.py -- $pass_tag pass:"
        for row in "${rows[@]}"; do
            IFS='|' read -r _rt rv ts _dir <<< "$row"
            printf '    ("%s", "%s"): "%s",\n' "$GRIPPER_FOLDER" "$rv" "$ts"
        done
    done
    if [ ${#PASSES[@]} -gt 1 ]; then
        echo
        echo "  NOTE: both passes produce a timestamp for the SAME"
        echo "  (gripper, variant) key, so the two blocks above cannot both sit"
        echo "  in RUN_TIMESTAMPS at once. Evaluate one pass, then the other."
        echo "  The checkpoint directory names differ -- the noised runs carry a"
        echo "  vn<sigma>u token -- so nothing was overwritten."
    fi
    echo
    echo "  Full checkpoint directories:"
    for row in "${CKPTS[@]}"; do
        IFS='|' read -r rt rv _ts dir <<< "$row"
        printf '    %-6s %-22s %s\n' "$rt" "$rv" "$dir"
    done
fi
echo "=============================================================="

[ ${#FAILED[@]} -eq 0 ]