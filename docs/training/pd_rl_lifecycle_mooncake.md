# PD disaggregation and RL pause/abort with MooncakeConnector

This note records why pause, sleep, abort and cache reset failed on a
prefill/decode (PD) deployment with `MooncakeConnector`, and what fixes them.
Measured on vLLM `9bc16ae`, one host, 1P1D on H200, Qwen3-8B, Mooncake over
RDMA, 32 lifecycle cases (pause/sleep/abort/reset/weight-update orders).

## How a Mooncake transfer works

- D pulls, P waits. D sends a pull to P; one of P's sender tasks waits until
  the prefill is ready, then writes the KV into D's blocks over RDMA.
- P has `num_workers * 2` sender tasks (20 by default). A pull that nobody
  answers holds a task for `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` (480 s);
  with all tasks held, every new request stalls.
- Requests in transfer are not in the running queue. On P the request has
  finished but keeps its blocks for D. On D it waits in
  `WAITING_FOR_REMOTE_KVS` and holds the blocks the KV is written to.

## Failures and root causes

| Symptom | Root cause |
|---|---|
| D's pause/sleep takes ~540 s after P aborts | P finishes a request before export but never answers its waiting pull |
| A few requests time out after any abort | After a request is aborted post-admission, the frontend (`notify_kv_transfer_request_rejected`) sends a release pull for KV that D already pulled; P no longer knows the transfer and waits on a placeholder |
| P keeps KV for D after everything is aborted | Nothing frees KV kept only for D before the lease expires |
| `reset_prefix_cache` returns 500 | It raises while transfers hold blocks, though its endpoint documents `success=false` |
| Keep pause with cache clear: D waits 480 s or fails | D's pending loads hold blocks the reset needs; a one-shot drop on P misses pulls that D sent just before its own pause |

## Fixes

1. **P answers every pull** (`_drop_transfers`). P tombstones each transfer it
   is done with: finished before export, sent, expired, or dropped. Waiting
   pulls fail now, later pulls fail on arrival, and running sends finish
   first. A connector hook `abort_exports()` drops every transfer that is not
   running; an abort of all requests (pause/sleep in abort mode,
   `/abort_requests {}`) calls it.
2. **Keep pause with cache clear.** The pause drops the transfers that are not
   running and aborts the waiting requests that hold blocks for them; the RL
   framework resubmits them, as in abort mode. D withdraws its pending pulls
   with the existing lease heartbeat (`withdraw` flag), and P drops them as
   above, so the aborted loads settle quickly and the reset gets all blocks.
   The reset returns False instead of raising while transfers hold blocks.

## Results (cases passing out of 32)

| main | fix 1 | fixes 1 and 2 |
|---|---|---|
| 1 | 23 | 30 |

Fix 1 was measured together with the change of `reset_prefix_cache` to
return False. "Fixes 1 and 2" was measured with an earlier version of fix 2
that kept and recomputed the requests in transfer instead of aborting them;
the abort version is being measured.
