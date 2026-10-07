#!/bin/bash
# Client v3 end-to-end throughput on ONE node via loopback (debug/micro-measure only):
# replay-server (cores 136-143) serves a recorded body N times after the barrier+pause
# protocol; client v3 runs exactly as in the harness (cores 128-135, 8 workers).
# Usage (inside srun on node01): replay-bench.sh BIN BODY FORMAT N OUTDIR [max_reading] [validate]
: "${GENOPT_ROOT:?set GENOPT_ROOT (see rfc-57479/harness/README.md)}"
set -euo pipefail
BIN=$1; BODY=$2; FMT=$3; N=$4; OUT=$5; MR=${6:-4}; VAL=${7:-4}
POOL=${GENOPT_ROOT}/agent_run/results/frontend-mock-256k-20261004/frozen-r1/model-assets/token-pool-v2.json
PORT=$((8400 + RANDOM % 500))
mkdir -p "$OUT/observe"
taskset -c 136-143 "$BIN" replay-server --port $PORT --body "$BODY" --requests "$N" \
  --output-tokens 245760 --consumption-log "$OUT/observe" 2> "$OUT/replay.log" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 0.5
FMTARG=(); [ "$FMT" = compact ] && FMTARG=(--logprobs-format compact)
# CLIENT (optional env) = client command to measure instead of v3 (e.g. "python genopt-client2.py");
# the replay server always comes from BIN.
CLIENT=${CLIENT:-$BIN}
/usr/bin/time -v taskset -c 128-135 /usr/bin/env PYTHONPATH=${GENOPT_ROOT}/agent_run/results/frontend-mock-256k-20261004/frozen-r1/python-baseline $CLIENT --urls http://127.0.0.1:$PORT --requests "$N" \
  --output-tokens 245760 --barrier-timeout 600 --validate-requests "$VAL" --token-pool "$POOL" \
  --consumption-log "$OUT/observe" --result "$OUT/result.json" "${FMTARG[@]}" \
  --workers 8 --max-reading "$MR" ${CLIENT_EXTRA:-} > "$OUT/client.log" 2> "$OUT/client.time"
grep -E "User time|System time|Maximum resident|Percent of CPU|Exit status" "$OUT/client.time" | sed 's/^[[:space:]]*//' | paste -sd';'
python3 - "$OUT/result.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
gb = sum(r["response_bytes"]) / 1e9
print(json.dumps({"requests": r["requests"], "format": r["logprobs_format"], "GB": round(gb, 2),
  "all_headers_s": round(r["all_headers_s"], 3), "all_bodies_received_s": round(r["all_bodies_received_s"], 2),
  "all_parsed_s": round(r["all_parsed_s"], 2), "GBps_end_to_end": round(gb / r["all_parsed_s"], 2),
  "client": r["client"], "client_cpu_s": r.get("client_process_cpu_s"),
  "sha_s_sum": r.get("v3_sha256_s_sum", r.get("post_receive_sha256_s_sum")),
  "parse_s_sum": r.get("v3_parse_validate_s_sum", r.get("post_receive_parse_s_sum")),
  "validated": r["semantically_validated_requests"], "max_rss_gib": r.get("client_max_rss_gib"),
  "distinct_sha": len(set(r["response_sha256"])), "status": r["status"]}))
EOF
