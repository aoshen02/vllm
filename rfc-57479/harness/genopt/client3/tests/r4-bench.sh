#!/bin/bash
# Round-4 throughput: loopback replays of recorded full-size bodies (from smoke.py --dump-dir).
: "${GENOPT_ROOT:?set GENOPT_ROOT (see rfc-57479/harness/README.md)}"
D=${GENOPT_ROOT}/agent_run/scripts/genopt/client3
BIN=${BIN:?}
B=${BODIES:-/dev/shm/claude-c3-r4}; O=${OUTDIR:-/dev/shm/claude-c3-r4-replay}
rm -rf $O
CLIENT_EXTRA="--routed-experts-layers 4" $D/replay-bench.sh $BIN $B/py-compact-r3/body-0000.json compact 256 $O/py-compact-r3-256
$D/replay-bench.sh $BIN $B/rust-compact/body-0000.json compact 256 $O/rust-compact-256
$D/replay-bench.sh $BIN $B/rust-openai/body-0000.json openai 32 $O/rust-openai-32
