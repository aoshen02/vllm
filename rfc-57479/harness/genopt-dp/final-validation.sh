#!/usr/bin/env bash
# FINAL VALIDATION chain for the production topology (DP8; EP8 is engine-side, the
# mock engine stays on CPU). Re-runnable unchanged on the final PR stacks.
#
# Usage:
#   final-validation.sh --impl rust|python --candidate PATH --out DIR
#                       [--baseline PATH] [--repeats N] [--cells LIST] [--job ID]
#                       [--rust-threads N] [--skip-baseline-cells LIST] [--dry-run]
#                       [--no-listeners] [--cand-env K=V]... [--base-env K=V]...
#   --candidate / --baseline: Rust frontend binary (observer build) for --impl rust,
#       node-local Python source tree (present on node02 and node03) for --impl python
#       (a PYTHONPATH, so TREE:TREE/plugins/X adds a source-tree plugin).
#   --no-listeners: Python candidate without --independent-listeners (shared accept
#       socket, as the baseline always runs); default: candidate with listeners.
#   --cand-env / --base-env: environment for that arm's dp-run.py (inherited by every
#       srun it starts), e.g. --cand-env VLLM_PLUGINS=rl_compact.
#   --cells: comma list from
#       cohort-rl-v3      (1) bs256 fixed 245760, RL-lean, client v3 official (8 cores)
#       cohort-rl-wide    (1) same, wide client (72 cores, 64 parse threads; measurement variant)
#       cohort-def-wide   (2) bs256 fixed 245760, default format, wide client
#       wf-paced          (3) bs256 lognormal RL workflow, paced engine, pause 10 s, sleep 5 s, resume, resubmit
#       n32-def, n32-rl   (4) n32 single-node deployable cells (frontend on node02 only)
#     default: all of the above.
#   --skip-baseline-cells: cells where the baseline cannot run (no compact format, known
#     memory failure); default: cohort-rl-v3,cohort-rl-wide,cohort-def-wide,wf-paced,n32-rl.
# Order: with a baseline A, repeats are interleaved A B B A A B ... (ABBA-style);
# without one, B is repeated. Every cell takes and releases the shared
# NODES-IN-USE.txt lock on its own, so other agents can interleave between cells.
# Rust runs as one process (the RustFrontendProcessManager layout) on node02;
# Python as 32 API servers per node on node02+node03 (bs256) or node02 (n32).
# All cells use real client_index output routing.
: "${GENOPT_ROOT:?set GENOPT_ROOT (see rfc-57479/harness/README.md)}"
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${GENOPT_PYTHON:-python3}
RUN=$HERE/dp-run.py
LOCK=${GENOPT_ROOT}/agent_run/results/generate-opt-20261004/NODES-IN-USE.txt
IMPL= CAND= BASE= OUT= REPEATS=3 JOB=${SLURM_HOLD_JOB:-} THREADS=64 DRY=0 LISTEN_B=1
CAND_ENV=() BASE_ENV=()
CELLS=cohort-rl-v3,cohort-rl-wide,cohort-def-wide,wf-paced,n32-def,n32-rl
SKIP_BASE=cohort-rl-v3,cohort-rl-wide,cohort-def-wide,wf-paced,n32-rl
while [ $# -gt 0 ]; do
  case $1 in
    --impl) IMPL=$2; shift 2;; --candidate) CAND=$2; shift 2;; --baseline) BASE=$2; shift 2;;
    --out) OUT=$2; shift 2;; --repeats) REPEATS=$2; shift 2;; --cells) CELLS=$2; shift 2;;
    --job) JOB=$2; shift 2;; --rust-threads) THREADS=$2; shift 2;;
    --skip-baseline-cells) SKIP_BASE=$2; shift 2;; --dry-run) DRY=1; shift;;
    --no-listeners) LISTEN_B=0; shift;;
    --cand-env) CAND_ENV+=("$2"); shift 2;; --base-env) BASE_ENV+=("$2"); shift 2;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$IMPL" ] && [ -n "$CAND" ] && [ -n "$OUT" ] || { echo "need --impl --candidate --out" >&2; exit 2; }
mkdir -p "$OUT"
LOG=$OUT/final-validation.log
echo "final-validation impl=$IMPL candidate=$CAND baseline=${BASE:-none} repeats=$REPEATS cells=$CELLS job=$JOB listeners_B=$LISTEN_B cand_env=${CAND_ENV[*]:-} base_env=${BASE_ENV[*]:-}" | tee -a "$LOG"

arm_args() { # arm path cell
  local arm=$1 path=$2 cell=$3
  # The Python baseline predates --independent-listeners support in the harness
  # runs (shared accept socket, as in all earlier baseline cells).
  local listeners="--listeners"; { [ "$arm" = A ] || [ $LISTEN_B = 0 ]; } && listeners=""
  if [ "$IMPL" = rust ]; then
    echo "--impl rust --binary $path --nodes node02 --workers-per-node 1 --rust-threads $THREADS"
  elif [ "${cell#n32}" != "$cell" ]; then
    echo "--impl python --source $path $listeners --nodes node02 --workers-per-node 32"
  else
    echo "--impl python --source $path $listeners --workers-per-node 32"
  fi
}
cell_args() {
  local RL="--logprobs-format compact --compact-no-sampled --compact-no-ranks"
  local WIDE="--client-cpus 64-135 --client-arg=--parse-threads --client-arg=64"
  case $1 in
    cohort-rl-v3)    echo "--dp 8 --requests 256 $RL --client v3 --validate-requests 32";;
    cohort-rl-wide)  echo "--dp 8 --requests 256 $RL --client v3 --validate-requests 32 $WIDE";;
    cohort-def-wide) echo "--dp 8 --requests 256 --client v3 --validate-requests 4 $WIDE";;
    wf-paced)        echo "--dp 8 --requests 256 $RL --client dp --client-cpus 64-135 --tokens-per-step 32 --step-ms 20 --client-arg=--workers --client-arg=32 --client-arg=--lengths --client-arg=lognormal --client-arg=--lognormal-median --client-arg=32768 --client-arg=--lognormal-sigma --client-arg=1.0 --client-arg=--length-min --client-arg=1024 --client-arg=--pause-at --client-arg=10 --client-arg=--resume --client-arg=--resubmit --client-arg=--sleep-s --client-arg=5";;
    n32-def)         echo "--dp 8 --requests 32 --client v3 --validate-requests 4";;
    n32-rl)          echo "--dp 8 --requests 32 $RL --client v3 --validate-requests 32";;
    *) echo "unknown cell $1" >&2; return 1;;
  esac
}
take_lock() {
  until ( set -o noclobber; echo "claude-rust-subagent final-validation $IMPL cell $1 on node02,node03,node04 (job $JOB); started $(date -u +%FT%TZ); remove when done" > "$LOCK" ) 2>/dev/null; do sleep 30; done
}
run_cell() { # cell arm rep path
  local cell=$1 arm=$2 rep=$3 path=$4 dir=$OUT/$1-$2-$3
  [ -e "$dir/run.json" ] && grep -q '"status": "PASS"' "$dir/run.json" && { echo "skip existing PASS $dir" | tee -a "$LOG"; return; }
  rm -rf "$dir"
  local args="--job $JOB $(arm_args "$arm" "$path" "$cell") $(cell_args "$cell")"
  local envs=("${CAND_ENV[@]}"); [ "$arm" = A ] && envs=("${BASE_ENV[@]}")
  if [ $DRY = 1 ]; then echo "DRY env ${envs[*]:-} $PY $RUN $args --label fv-$IMPL-$cell-$arm-$rep --run-dir $dir" | tee -a "$LOG"; return; fi
  take_lock "$cell-$arm-$rep"
  echo "$(date -u +%FT%TZ) start $cell $arm $rep" | tee -a "$LOG"
  /usr/bin/env "${envs[@]}" $PY "$RUN" $args --label "fv-$IMPL-$cell-$arm-$rep" --run-dir "$dir" > "$dir.out" 2>&1
  rm -f "$LOCK"
  echo "$(date -u +%FT%TZ) end $cell $arm $rep $(grep -o '"status": "[A-Z]*"' "$dir/run.json" 2>/dev/null)" | tee -a "$LOG"
}
for cell in ${CELLS//,/ }; do
  use_base=0
  if [ -n "$BASE" ] && [[ ",$SKIP_BASE," != *",$cell,"* ]]; then use_base=1; fi
  for ((rep = 1; rep <= REPEATS; rep++)); do
    if [ $use_base = 1 ]; then
      # A B B A A B ...: odd repeats A then B, even repeats B then A
      if [ $((rep % 2)) = 1 ]; then run_cell $cell A $rep "$BASE"; run_cell $cell B $rep "$CAND"
      else run_cell $cell B $rep "$CAND"; run_cell $cell A $rep "$BASE"; fi
    else
      run_cell $cell B $rep "$CAND"
    fi
  done
done
[ $DRY = 1 ] && exit 0
$PY "$HERE/final-summary.py" "$OUT" --impl "$IMPL" --candidate "$CAND" --baseline "${BASE:-}" | tee -a "$LOG"
echo FINAL-VALIDATION-DONE | tee -a "$LOG"
