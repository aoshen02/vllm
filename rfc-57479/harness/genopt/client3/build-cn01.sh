#!/bin/bash
# Build genopt-client3 on node01 (cores 128-143) with rustc 1.95.0, node-local target dir.
# Usage: build-node01.sh [cargo subcommand args...]   (default: build --release)
set -euo pipefail
cd "$(dirname "$0")"
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/tmp/claude-client3-target
exec srun --jobid=${SLURM_HOLD_JOB:?} --overlap --nodes=1 --ntasks=1 --nodelist=node01 --cpu-bind=none \
  taskset -c 128-143 cargo +1.95.0 "${@:-build}" --release --offline --locked
