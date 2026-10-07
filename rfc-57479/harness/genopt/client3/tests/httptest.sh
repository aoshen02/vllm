#!/bin/bash
# HTTP framing / protocol tests for client v3 (copied from the Claude audit; BIN, EXTRA, URLHOST overridable) (node01, cores 136-143 server / 128-135 client).
: "${GENOPT_ROOT:?set GENOPT_ROOT (see rfc-57479/harness/README.md)}"
W=$(cd "$(dirname "$0")" && pwd)
R=${GENOPT_ROOT}/agent_run
BIN=${BIN:-$R/results/generate-opt-20261004/src/client3/genopt-client3-5bfdc72e}
POOL=$R/results/frontend-mock-256k-20261004/frozen-r1/model-assets/token-pool-v2.json
BODIES=$R/results/generate-opt-20261004/client3-smoke/bodies-rust-compact
PY=${GENOPT_PYTHON:-python3}
port=18700
for spec in "$@"; do
  mode=${spec%%:*}; extra=""; [[ "$spec" == *:* ]] && extra=${spec#*:}
  port=$((port+1))
  T=$(mktemp -d /dev/shm/c3audit.XXXX)
  taskset -c 136-143 $PY $W/fakesrv.py --port $port --bodies $BODIES --requests 2 --log $T/log --mode $mode $extra > $T/srv.log 2>&1 &
  srv=$!
  sleep 0.7
  start=$(date +%s.%N)
  timeout 60 taskset -c 128-135 $BIN --urls http://${URLHOST:-127.0.0.1}:$port --requests 2 --output-tokens 4096 \
     --logprobs-format compact --validate-requests 1 --token-pool $POOL --consumption-log $T/log \
     --result $T/result.json --workers 2 --max-reading 1 --barrier-timeout 20 --http-timeout ${HTTP_TIMEOUT:-20} $EXTRA > $T/out.txt 2> $T/err.txt
  rc=$?
  end=$(date +%s.%N)
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  status=$( [ -f $T/result.json ] && $PY -c "import json,sys;r=json.load(open('$T/result.json'));print(r['status'],r['semantically_validated_requests'],len(r['response_sha256']),r['response_bytes'])" )
  printf '%-28s rc=%-3s %5.1fs result=[%s] err=%s srv=%s\n' "$mode" "$rc" "$(echo "$end - $start" | bc)" "$status" "$(tr '\n' ' ' < $T/err.txt | cut -c1-220)" "$(grep -v listening $T/srv.log | tr '\n' ' ' | cut -c1-100)"
  rm -rf $T
done
