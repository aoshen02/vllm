#!/bin/bash
# Round-3 throughput: single-core bench of an R3 compact body + loopback replays (bodies from smoke.py --dump-dir).
: "${GENOPT_ROOT:?set GENOPT_ROOT (see rfc-57479/harness/README.md)}"
D=${GENOPT_ROOT}/agent_run/scripts/genopt/client3
BIN=${GENOPT_ROOT}/agent_run/results/generate-opt-20261004/src/client3/genopt-client3-ec5a6369
B=/dev/shm/claude-c3-r3; O=/dev/shm/claude-c3-r3-replay
POOL=${GENOPT_ROOT}/agent_run/results/frontend-mock-256k-20261004/frozen-r1/model-assets/token-pool-v2.json
rm -rf $O
ls -la $B/*/body-0000.json | awk "{print \$5, \$9}"
for v in 1 0; do idx=$((1-v)); echo "bench py-compact-r3 validated=$v"
  taskset -c 128 $BIN bench-body --body $B/py-compact-r3/body-000$idx.json --index $idx --output-tokens 245760 --token-pool $POOL --logprobs-format compact --routed-experts-layers 4 --validated $v --repeat 2 | grep -o "parse_validate_GBps[^,]*,.*per_core[^,]*\|status[^,]*"
done
CLIENT_EXTRA="--routed-experts-layers 4" $D/replay-bench.sh $BIN $B/py-compact-r3/body-0000.json compact 256 $O/py-compact-r3-256
$D/replay-bench.sh $BIN $B/rust-compact/body-0000.json compact 256 $O/rust-compact-256
$D/replay-bench.sh $BIN $B/rust-openai/body-0000.json openai 32 $O/rust-openai-32
