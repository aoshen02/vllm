# [RFC]: AuxOutput Connector — block-keyed outputs and storage roadmap

Draft update of vllm-project/vllm#45221 (adds the section "Design comparison").

## Summary

AuxOutput Connector (formerly Artifact Connector) manages auxiliary model outputs for post-training workloads. Its first output is routed experts (R3): the expert IDs selected for each executed token of an MoE model.

The foundation is merged in #45635. The next steps are **faster value delivery** (#61086), **distributed storage and key-based delivery** (#57968), and **an extensible response schema** (#61291). This RFC tracks their shared architecture and follow-ups; implementation and validation belong to the linked PRs.

Status below was checked on October 15, 2026. Open proposals are not merged capabilities.

## Requirements and architecture

Two consumption patterns should remain available:

- **Return values:** simple integration for existing RL clients, without a separate remote reader.
- **Return references:** large deployments fetch data from shared storage, avoiding forwarding large payloads through API servers and a central RL controller.

Most initial outputs are token-aligned, so KV-compatible prefix hashes and block boundaries provide a useful basis for reuse. However, KV and auxiliary data need not have the same lifetime: a trainer may still need R3 after inference finishes or a GPU KV cache is reset.

A capture hook produces only the current forward's tensor. The connector handles accepted versus rejected rows, cached prefixes, block assembly, partial tails, and request completion. Storage owns bytes and retention; training coordination stays outside the connector.

```text
GPU Worker process
  model forward -> capture -> async GPU-to-CPU copy
                              |
                     AuxOutputWorkerConnector
                     accepted rows / complete blocks
                       |                    |
                 local SHM             Mooncake Store
                       |                    ^
                       v                    |
EngineCore process: progress, hashes, accepted boundaries/tails
                       |                    |
                       v                    v
API Server process:  R3 values         output references
                       |                    |
                       v                    v
                   RL client        trainer -> Store.get(keys)
```

Mooncake #57968 also has an EngineCore publisher for boundaries/tails finalized after stop and length handling. #61086 batches arrays at existing IPC boundaries on the value path. These are complementary workstreams; their combined runtime behavior still needs validation.

## Main workstreams

### 1. Batch incremental R3 transport — #61086, open

Pack many small per-request arrays into a compact batch array with request spans/row counts. Unpack at the receiver and accumulate into request-owned storage. This reduces Python object and serialization overhead while retaining incremental delivery and partial results on abort.

The original Nemotron/GB200 comparison reported approximately **19% higher output-token throughput** (3 baseline runs, 2 batching runs); this is workload-specific. Packing still copies data and is not a zero-copy or lower-memory claim. Mooncake worker-tail batching requires separate integration and measurements. [Reproduction and results](https://gist.github.com/aoshen02/377db5658721d30b934e9a227a3c19fa).

### 2. Mooncake backend — #57968, open

Store complete blocks and accepted partial tails remotely; return ordered keys so external consumers can fetch the values. Reuse the existing background writer and native Mooncake I/O. SHM remains the default.

The PR records cross-host RDMA retrieval, prefix reuse, preemption, speculative rejection and Mooncake PD handoff experiments. P/D sharing requires matching generation, model/R3 profile, hash configuration and block size. Removing a configuration restriction does not establish support for every KV connector/backend combination.

Retention remains deployment-managed: request completion does not delete remote objects, but the backend does not yet implement trainer-held references or guarantee protection until a delayed trainer reads them.

### 3. Structured HTTP output — RFC #61291 and [fork draft #132](https://github.com/aoshen02/vllm/pull/132)

Walter's current proposal builds on #57968 and replaces the R3-specific key-list contract with a named `aux_outputs` object:

```text
aux_outputs
  routed_experts
    token_start, num_rows
    metadata: dtype, num_layers, experts_per_token
    chunks: [{key, num_rows}, ...]
  <future output field>
    its own range, metadata and chunks
```

This describes how to decode and place each payload, allowing additional fields without requiring R3-only readers to understand them. Large top-k token logprobs for RL score centering motivate the extension; the schema draft itself does not implement their capture and storage.

The earlier stored-manifest experiment, [fork #131](https://github.com/aoshen02/vllm/pull/131), has been superseded in the RFC by this simpler HTTP structure. An additional remote manifest object is not required by the current proposal. Review still needs to settle migration from `aux_output_keys`, SHM/value-path compatibility, and preservation of stop/abort/empty-terminal behavior.

## Related implementation roadmap

| Area | Work | Current scope/status |
| --- | --- | --- |
| Additional outputs | #60789 | Draft: generated and prompt logprobs through AuxOutput. Selected-token/fixed top-k, raw-logprob mode; full-vocabulary output and online weight updates are excluded. Align its field/range contract with #61291. |
| Avoid repeated prefix delivery | #57966; RFC #56402 | Open: opt-in prefix omission with caller-owned route history and session identity. Does not itself add KV offloading. |
| Custom eviction policy | #58328 | Open: local-store `ExpertCachePolicy` hooks and selection from eligible unreferenced blocks. Worker configuration and Mooncake retention are outside this PR. |
| Pipeline parallelism | #61333 | Open: combine routing rows from all PP stages through existing stage transfers. |
| TP-sharded capture | #61388 | Open: restore the full local token axis when model-level SP shards rows at DP=1. |
| DBO ordering/offsets | #57184 | Related open prototype; does not establish general DBO+R3 correctness. |
| Multimodal token coordinates | #61097 | Open: account for model-expanded prompt ranges and P/D prompt cuts. |

Recent merged foundations/fixes:

- #61000: own SHM out-of-band buffers before releasing reusable queue slots; received arrays must remain valid after subsequent messages.
- #61643: retain all required routing rows during DeepSeek V4.1 decoder replay, using the existing keep-row mechanism. R3 can therefore require work that logits-only inference skips.
- #58150 / #58205: narrow KV-connector restrictions and scope the MRV2 requirement to applicable GPU platforms. These configuration changes do not certify every runtime combination.
- #59060: remove redundant Aux reset guards for the completed-pause workflow. It does not implement arbitrary reset during active inference or preservation across every GPU-only reset.

## Design comparison: where R3 rows of prefix-cached blocks live

Added October 28, 2026. This section addresses contract 3 ("KV hits also require matching auxiliary data"). It compares #57968 with a prototype that stores R3 as a KV cache layer. The prototype PR will follow; the numbers below come from our own experiments, not from the linked PRs.

### Two lifetimes

A row is produced by the forward pass that computes that position's KV. A prefix hit skips that forward pass, so R3 data has two lifetimes:

| Segment | Must live while | Lifetime decided by |
| --- | --- | --- |
| Cache segment | the block's KV can still be hit (GPU, CPU offload, KV store, P→D) | KV eviction: block manager and connectors |
| Delivery segment | after the request finishes, until the trainer fetches the rows | trainer consumption delay |

Outputs defined only at sampled positions (e.g. top-k logprobs, #60789) have no cache segment and are a pure delivery problem. Outputs defined at all positions (R3, indexer replay) need both segments. The two designs differ only in **who owns the cache segment**.

| | Cache segment | Delivery segment |
| --- | --- | --- |
| **A. #57968** | Mooncake objects keyed by block hash, shared across engines, first writer wins | Existing block keys for prompt blocks, plus an `r3-tail` object for decoded rows |
| **B. R3 as a KV layer** (prototype) | Host pinned memory registered as one extra layer of the full-attention KV group. Rows are written through `slot_mapping` and share block IDs with the KV, so they are hit, evicted, offloaded and transferred with it | At request end, all rows are written once to the delivery backend (SHM inline by default, or Mooncake keys) |

#58328 (eviction-policy hooks) applies only to a separate R3 store like A's; B has no separate store. #57966 (prefix omission) reduces what is delivered and works with either design; prefixes shared across requests still need a cache segment.

### Measured (Qwen3-30B-A3B, 384 B/token, single H200, Mooncake over RDMA)

**Mooncake eviction order.** A read gives an object a 10 s lease. Eviction runs at the 0.9 watermark, roughly in lease-expiry order, so **objects written but never read are evicted first**. Leased objects cannot be deleted either. A's cache segment consists of exactly such objects: prompt blocks are written at prefill and first read only when the trainer fetches, and an engine hitting a full block never reads its object.

**Loss under pool pressure.** Agent workload: 96 groups × 8 turns sharing a 2048-token system prompt, 64 GB CPU offload, 2304 requests per cell. Numbers are requests with missing rows; `d` = trainer fetch delay.

| Pool | A d0 / d5 / d15 | B, delivery deduplicated by block hash, d0 / d5 / d15 |
| --- | --- | --- |
| 192 MB | 1636 / 2304 / — | 4 / 62 / 122 |
| 256 MB | 4 of 9 rounds lost all 768 / 1 of 3 rounds lost all 768 / 3 of 3 rounds lost all 768 | 0 / 0 / 0 |
| 384 MB | 0 / 0 / 0 | 0 / 0 / 0 |

- A loses the cache segment, all or nothing. When a shared system-prompt block is evicted between prefill and the first fetch, every request referencing it misses all 128 system-prompt blocks. HTTP responses are all 200 and the engine stays healthy. The rows cannot be recovered, because the engine never had them.
- Above the threshold, A loses nothing; a GRPO workload that reads each object once also lost nothing at the watermark. The threshold depends on the whole KV hierarchy and the trainer's read pattern, and there is no provable bound.
- B's prototype delivers one variable-size object per request without dedup. That writes 1.4–6× more than A, fragments a full pool over time (1.2% put failures over 60 min at 512 MB), and a put failure exits EngineCore. All three are delivery-layer issues: dropping only the failed request, 64 KB chunking, and block-hash dedup each fixed theirs in these experiments. With dedup, B needs a smaller pool than A (0-loss threshold: B 192–224 MB, A 256–320 MB), and it loses a few rows per request instead of whole batches.

**Cost.**
- Pool writes: A ≈ 384 B × (unique prompt tokens + 2 × response tokens), since decoded rows are written once as `r3-tail` and once per completed block.
- B adds host pinned memory of ≈ KV token capacity × 384 B, i.e. ~0.4% of KV bytes: 109–298 MB per engine across four models. The offload tier also stores the R3 layer. GPU KV capacity changes by ≤ 0.04%.
- Worst workload above: A ≈ 384 MB; B ≈ 256 MB pool + 107 MB host layer + ~250 MB in the offload tier (estimated, tier full).

**Performance.**
- With Mooncake delivery, both are within ±5% of R3 off.
- Prefill with no hit: A 166 → 180 ms, B 165 → 166 ms.
- 16 GRPO siblings hitting an 8k prefix: A +18–35 ms, B +67 ms. B copies hit rows through the GPU at the last prompt step; reading them by slot in EngineCore would remove that, but this is not implemented.
- Trainer fetch: A 2.5–2.8 GB/s, B 3–28 GB/s.
- Stop strings cost 31–37% on both, for a shared reason: stop is detected in the API process.

**Row provenance without batch invariance.** Under block-hash sharing, the first writer's rows go to every request hitting that hash. In GRPO, 34–42% of siblings had ~0.3% of their prompt rows differ from the KV they actually used. B returns rows matching each request's own KV. A deduplicated delivery should share keys only for blocks reused from the same slot (estimated at 1–6% more pool).

**Coverage of B** (cached rows compared with the computing request's rows; 0 mismatched rows unless noted):
- Validated: local prefix caching, preemption, ngram speculative decoding, sleep/wake + `reset_prefix_cache`, TP2, DP2.
- Models: Qwen3-30B, Qwen3.6 (Gated DeltaNet hybrid), gpt-oss-20b (sliding window), DeepSeek-V2-Lite (MLA). Hybrid models need a block-outer KV layout.
- KV connectors: `OffloadingConnector`, `SimpleCPUOffloadConnector`, `MultiConnector`, and `MooncakeStoreConnector`, including two instances over RDMA.
- P/D: `NixlConnector` 1P1D, including heterogeneous TP and MLA. This needed the NIXL producer side to split DRAM and VRAM descriptors. `MooncakeConnector` P/D: 0 with batch invariance; without it, mismatches appear only in positions D recomputes.
- PP2 (+ offload): each stage caches its own layers, and rows are stitched at request end with #61897's aggregator.
- Rejected at config time: NIXL push, MoRIIO, `LMCacheConnectorV1`, `LMCacheMPConnector`, PP + NIXL, DCP/PCP > 1.

B does not get every KV path for free. Its guarantee is that the rows live as long as the KV wherever the KV goes, but each connector that moves KV must handle one host-resident layer.

### Proposal

1. Use B for the cache segment: "KV hit ⇒ rows exist" holds by construction across every KV tier, with no second LRU to coordinate. This resolves contract 3 for R3. Contracts 2 and 4 then apply only to the delivery segment.
2. Before merging, fix B's delivery layer:
   - a failed put drops only that request;
   - deduplicate by block hash, sharing keys only for slot-reused blocks;
   - avoid variable-size large objects;
   - read hit rows from the host layer in EngineCore;
   - allocate the host layer on one TP rank.
3. Mooncake asks: take a lease or reference at write time, so never-read objects are not evicted first, and allow deleting leased objects. This is the storage half of contract 2.
4. Independent of this choice: #61086, #61291, the stage-stitching half of PP support, capture fixes.

Open questions: a connector capability flag instead of a deny list for the host-resident layer; host memory for indexer replay at 61 layers (~121 GiB per engine), which needs per-layer opt-in or another tier.

## Contracts still to define

1. **Output coordinates and identity.** R3 describes executed tokens; logprobs describe predictions and need an explicit target-token convention. Specify dtype/shape, raw versus processed probabilities, model/weight version and reuse compatibility. DSA/index data remains a follow-up; non-token-aligned data needs a concrete use case before adding machinery.
2. **Shared retention across inference and training.** Store one logical R3 object and let multiple training batches reference it. Track acquire/release separately from inference-cache retention; registration must precede exposing a key to a consumer. Reclaim only after all required uses finish. A generation change isolates identity but is not a deletion signal for old training data. This protocol is proposed, not implemented by #57968 or #58328.
3. **Eviction policy versus protection.** The storage/lifecycle layer decides which objects are eligible; a plugin chooses which eligible objects to evict. It must not override live-consumer protection. KV hits also require matching auxiliary data: define coordinated retention or invalidation/recomputation when R3 is unavailable. Sharing a hash does not ensure joint availability.
4. **Bounded capacity and failures.** Define backpressure or tiering when protected data fills the store, cleanup of abandoned consumer holds, and observability for retained bytes and consumer lag. TTL alone cannot guarantee delivery to arbitrarily slow consumers. Remote-store failures also need an explicit request/engine failure policy.
5. **Compatibility validation.** Preserve partial outputs, async ownership and SHM behavior as storage/schema/packing changes compose. Test actual row provenance and slot reuse, plus prefix hits, chunked prefill, speculative rejection, preemption, reset, PD and parallel layouts. Support and performance claims remain tied to the configurations exercised in each PR.

AI assistance was used to research and update this RFC. Linked experiments are reported by their implementation PRs; this roadmap update did not rerun them.
