# Theory: where the post-pause time and memory go

Notation:
- P = output positions per request (245,760);
- k = top-k (128), so each position has S = k + 1 = 129 slots (the sampled token plus the top-k);
- N = concurrent requests aborted by one pause;
- B = the busiest Python API server's request count.

Constants come from measurements on GB200 nodes; each is followed by its source key.

## 1. Data volume

### 1.1 Entries
E = P × S = 245,760 × 129 = **31,703,040 entries per request**.

### 1.2 Default JSON

| Frontend | Bytes per request | Bytes per entry | n = 32 | bs = 256 |
|---|---|---|---|---|
| Rust | 3,408,730,749 [S6] | 3,408,730,749 / 31,703,040 = **107.52** | 109.08 GB | 872.6 GB [S3] |
| Python | 70,388,073,920 / 32 = 2,199,627,310 [S7] | **69.38** | 70.39 GB [S7] | 563.10 GB [S7] |

- **The two default schemas differ**, so the sizes differ. Each frontend's bytes equal its own base:
  - Rust: sha256 `572252e6…` at base and at every PR tip.
  - Python: golden scenarios, and byte-length lists equal across arms.
- **Bodies per n = 32 cell:** 109.08 GB is 32 × 3,408,730,749 B. A harness body can differ from the
  genbench payload by a few hundred bytes (ids, `created`); the per-cell totals in results.md are
  the measured ones.

### 1.3 Compact
The top-k block is two base64 strings over little-endian arrays, `int32[P×k]` ids and
`float32[P×k]` logprobs. The sampled ids remain in `choices[0].token_ids`.

| Part | Bytes |
|---|---|
| Raw arrays: 245,760 × 128 × (4 + 4) | 251,658,240 |
| Base64 (× 4/3; 125,829,120 B per array is divisible by 3, so no padding) | 335,544,320 |
| Remainder: the JSON `token_ids` list, block keys and envelope (337,043,223 − 335,544,320) | 1,498,903 (≈ 6.10 B per id) |
| **Measured, genbench, sha256 `1a5360f0…`** [S6] | **337,043,223** |
| Measured in the harness (10,785,391,520 / 32) [S7] | 337,043,485 |

Ratios:
- default / compact: Rust 3,408,730,749 / 337,043,223 = **10.11×**; Python 2,199,627,310 /
  337,043,485 = **6.53×**.
- Per position, compact carries 337,043,223 / 245,760 ≈ **1,371 B**.

### 1.4 Wire floor
The client NIC is 200 Gb/s = 25 GB/s [S14]. Minimum transfer time = bytes / 25 GB/s:

| | n = 32 | bs = 256 |
|---|---|---|
| Compact | 10.79 GB → **0.43 s** (measured floor 0.434 s for 10.86 GB with framing [S14]) | 86.28 GB → 3.45 s |
| Python default | 70.39 GB → 2.82 s | 563.1 GB → 22.5 s |
| Rust default | 109.08 GB → 4.36 s | 872.6 GB → 34.9 s |

## 2. Python object model
**Base.** `LogprobsProcessor` turns every engine row into `Logprob` objects held in per-position
dicts while consuming. The default builder then creates a pydantic object per entry and renders
it.

Base-path bench, one request of P × S, default format, one core [S13]:

| Quantity | Base | / entry (÷ 31,703,040) |
|---|---|---|
| GC-tracked objects alive at the abort | 32,316,015 | 1.02 |
| RSS after consumption | 9.12 GB | 288 B |
| Consumption | 376.74 µs/position → 92.59 s | 2.92 µs |
| Build + render | 134.72 s | 4.25 µs |
| of which gen-2 GC in the build window | 37.19 s (5 gen-2 collections) [S22] | |
| VmHWM | 39.38 GB | 1.24 KB |

Bench memory figures here are the kernel kB counters divided by 10⁶ (so "39.4 GB" is 39.4 × 10⁶ KiB ≈ 40.3 × 10⁹ B ≈ 37.6 GiB); per-entry byte figures derived from them are therefore about 2.4% low.

**Consequences:**
- **Per-process memory grows with the requests a server holds.** At n = 32 DP8, servers reached
  up to 36 GiB (the largest process; 28–31 of 32 exceeded 10 GiB), and the node fell below its 64 GiB free-memory floor after about 8 minutes [S2][S8].
  At bs = 256, the stack without P3 reached 30–75 GiB per process [S8].
- **Post-abort CPU per request is about 127 s (r14 bench) to 135 s (r21–r23 benches)**, all of it
  on one thread [S12][S13][S22].
- **Base builds on the event-loop thread.** Its `/pause` reply waits when `/pause` is accepted by
  a busy server:
  - 132–140 s in the r14 base runs [S12];
  - 141 s in the final n = 32 harness [S1];
  - 144.9 s in dp-round3 A2 [S11].

  When `/pause` lands on an idle server it returns in 0.07 s (dp-round3 A1).

### 2.1 Why array storage alone is slower
T2 in the r14 ablation keeps engine rows as arrays but leaves the legacy renderer [S12]:

| | T0 base | T2 storage only |
|---|---|---|
| Consumption | 373 µs/pos (91.8 s) | 0.34 µs/pos (0.08 s) |
| Build after the abort | 127 s | **148 s**, of which 94 s materialises `Logprob` objects from the arrays |
| HWM | 39.2 GB | 34.3 GB |
| n = 32 server ready (last) | 157–270 s | 433–453 s |

- **Why T2 loses.** The legacy renderer needs the objects, so T2 builds them after the abort, which
  is the critical path, instead of during consumption.
- **What P3 does instead.** It enables array storage only for requests its object-free renderer
  serves.

### 2.2 P3
Same bench, same body sha `e663245f53` [S13]:

| | P3 | Base / P3 |
|---|---|---|
| Consumption | 0.335 µs/position (0.083 s) | 1,125× |
| Build + render | 4.44 s | 30.3× |
| VmHWM | 5.60 GB | 7.0× |
| RSS after consumption | 1.27 GB | 7.2× |
| Live objects at the abort | 613,313 | 52.7× |

`VLLM_GENERATE_ARRAY_LOGPROBS=0` gives 377.32 µs/pos and 134.98 s, which is base parity [S13].

## 3. Time model

### 3.1 Python, default format
**Per server:**
- Large builds go to one sequential builder thread.
- The build is GIL-bound: only 10–15% (numpy, byte joins) runs without the GIL [S9].
- So within a server the responses become ready at a fixed spacing `c`.

**Predicted tail:**
```
T_ready(last) ≈ t0 + c × B
T_all_parsed ≈ a + c × B
```

**Measured:**

| Tree | c | Source |
|---|---|---|
| In-tree `214f522` / P3 / stack, single server, N = 8 or 10 | build CPU 4.397–4.426 s per request; wall 4.41–4.44 s; builds overlap 1 | [S9] |
| Fit over 9 cohort-def-wide runs (bs = 256, wide client) | **a = 30.7 s, c = 4.62 s, R² = 0.958**; mean residual in-tree −0.03 s (n = 6), stack +0.06 s (n = 3) | [S9] |
| Earlier in-tree `5b3a5f8`, n = 32 single node | 7.6 s + 5.4 s × (B − 1); predicted 18.4 / 23.8 / 29.2 s for B = 3 / 4 / 5, measured 18.6–20.5 / 23.6 / 28.0 s | [S11] |
| Base, n = 32 | server ready 157 s at B = 1, 270–280 s at B = 2, 411 s at B = 3 | [S12] |

**More build threads** (single server, N = 10) [S9]:

| Build threads | All parsed | HWM per process |
|---|---|---|
| 1 | 47.3 s | 8.8 GiB |
| 2 | 42.5 s | 12.9 GiB |
| 4 | 42.8 s | 18.8 GiB |
| 10 | 46.1 s | 37.5 GiB |

The full cohort-def-wide cell with 2 threads took 64.1 s [61.7–66.0], against 76.1 s
[72.2–78.0] with 1 [S9].

### 3.2 Rust, default format
Rust renders each response in a task on the multi-threaded request runtime:

| One 245,760 × 129 response, one core, median of 3 [S6] | Time |
|---|---|
| Base eager serde tree + `axum::Json` | 6.77 s |
| R-A streamed writer, no cache | 1.10 s (4.48 µs/position) |
| Old PR 2 with the fragment cache | 0.71 s |
| R-B compact (245,760 × 128), encode + render | 0.125 s (median of 3) at `bc60a7b3ed`; 0.141 s (1 run) at the final `d1d9770b7b` with the int32 check [S26] |

- **n = 32.** With 32–64 request threads all renders proceed concurrently, so the cell is bound by
  bytes, not by render.
- **Client limit.** Client v3 on 8 cores parses 109.1 GB of Rust default bodies in 18.14 s on
  loopback (6.0 GB/s) [S15].
- **Measured cell:** R-A all parsed 20.73 s (20.31, 21.15) [S4].
- **Cache.** With the cache (0.71 s per render), the cell took 20.75 s, which confirms the cell is
  client-bound [S4].

### 3.3 Compact
n = 32 split (Rust, client v3, NIC sampled every ~2 ms) [S14]:

| Stage | Time |
|---|---|
| Pause → first byte at the client NIC | 5–7 ms |
| Last response headers | 0.012–0.022 s |
| All bodies received | 0.519–0.594 s (floor 0.434 s; 72-core client 0.458–0.468 s) |
| Client after the last byte: sha256 6.3–7.0 CPU-s + parse/validate 4.6 CPU-s on 8 cores | 1.16–1.33 s |
| **All parsed** | **1.74–1.87 s** |

Python compact reaches server ready at 0.06–0.10 s and all parsed at 1.72–1.78 s [S12]; at
DP8 n32-rl it is 1.80 s [S7].

## 4. Distribution model
**Shared accept queue (base).**
- One listening socket is shared by all workers. A connection goes to whichever worker's event
  loop accepts first.
- A worker blocked in a CPU-bound build or consumption does not accept, so connections pile onto
  free workers. This is a skew that follows load, not chance.
- **Worst case:** the old 2,000-request GPU run gave one worker 1,492 requests (history file).
- **Best case:** when all connections arrive before any worker is busy, the spread can be perfect.
  The n = 16 base run A1 gave 16 × 1, and A2 gave 3, 2, 2, 2, 1 × 7 [S21].

**`SO_REUSEPORT` hashing (#58278).**
- Each worker has its own socket, and the kernel hashes the 4-tuple, so the spread is uniform
  random.
- For n balls in m bins the busiest bin follows the balls-into-bins maximum. For n = m = 32
  [S11]:

  | B | 2 | 3 | 4 | 5 | 6+ |
  |---|---|---|---|---|---|
  | P(busiest = B) | 0.03 | 0.52 | 0.36 | 0.08 | 0.01 |

- **Cost.** With serial per-server rendering, the expected extra tail is `c × (E[B] − n/m)`. It is
  large in base (c ≈ 120–135 s) and small after P3 (c ≈ 4.6 s).

**Observed at n = 32 over 64 servers (the two-node harness):**
- Listeners gave B = 2–3, against 1–2 for the shared queue [S12][S16].
- With P3-equivalent code, listeners took 19.05 / 20.21 s against 18.17 / 18.42 s without them
  [S16].
- The bs = 256 cohort cells without listeners had B = 7–13 [S9].

{{TBD:dp-round12 — #58278 at DP8: B distribution and all-parsed with and without listeners, in
the default and compact cells; conclusion on when it helps}}

## 5. Ingest model (pre-pause)
**What the frontend must sustain.** The real engine emits one token per request per step. At 25–50
tokens/s per sequence, DP8 serving 256 sequences in total means ≈ 6–13 k outputs/s (256 × 25–50 tokens/s) [S10].

**Measured ingest at real granularity** (1 token per step, int32 ids, 2 aux frames per output)
[S10]:

| Frontend | Ingest | Headroom |
|---|---|---|
| Rust, 1 process | ≥ 150–160 k outputs/s (151 k default, 159 k RL-lean) | ≥ 12× |
| Python, 64 processes | ≥ 73–77 k outputs/s | ≥ 5× |

- Both are lower bounds, limited by the mock: Rust is capped by the mock's rank-0 reducer at 1.0
  core.
- **Post-pause metrics are equal** between 1,024-token and 1-token steps (A/B, same bytes):

  | Cell | 1,024-token steps (A) | 1-token steps (B) |
  |---|---|---|
  | Rust RL-lean | 0.400 s | 0.393 s |
  | Rust default | 4.27 s | 4.34 s |
  | Python RL-lean | 0.282 s | 0.277 s |
  | Python default | 4.44 s | 4.39 s |

**Mock-only Rust gains** (1,024-row steps, so per-position cost dominates):

| Change | Effect | Source |
|---|---|---|
| Wire decode | Micro-bench 0.99 → 0.42 µs/position (reports/`rust-round4.md`, `rs-pr-stack/PR1.md`); n = 32 pre-pause consumption 14.5 → 10.2 s | [S4][S17] |
| Parallel decode | DP8 bs = 256 ingest RL-lean 0.81 → 1.27 M positions/s, default 0.95 → 1.46; post-pause, HWM and bytes unchanged | [S18] |

- **Next Rust limit:** a single ZMQ PULL receive task at ≈ 2.4 GB/s of engine payload [S25].
- Real engines are 12×+ below these rates, so neither change helps there.

## 6. Client ceiling
Client v3 (native; 8 cores; sha256 plus strict semantic validation), loopback replay [S15]:

| Cohort | GB | All parsed | Rate |
|---|---|---|---|
| Rust default ×32 | 109.1 | 18.14 s | 6.0 GB/s |
| Rust compact ×256 | 87.3 | 9.09 s | 9.6 GB/s |

- **At DP8 bs = 256** the RL compact cohort takes 10.85–10.97 s with this client, and 4.18–4.68 s
  with the 64-thread wide client [S2][S7]. The wire floor is 3.45 s.
- **A real trainer's decode rate** (base64 + `numpy.frombuffer`) is not measured here (RFC §8).

## 7. DP pause consensus
- **Real `DPEngineCoreProc`.** It runs the pause consensus all-reduce only when
  `step_counter % 32 == 0`, and idle ranks run real dummy forwards. A pause can therefore wait up
  to ~32 forward steps: **≈ 0.3–1.6 s** at 10–50 ms per step.
- **The mock** uses 1 ms dummy steps, i.e. 20–70 ms [S10].
- **Effect on comparisons:** the difference is engine-side and identical for both frontends, so
  no frontend comparison changes. Absolute pause times at DP > 1 in the RFC are understated by it.

## 8. Memory model

| Representation | Measure | Source |
|---|---|---|
| Engine row on the wire, k = 128 (int32 ids, f32 logprobs, int64 rank) | 129 × 8 + 8 = 1,040 B/position; 1,099 B with real msgpack framing (mock: 1,560 B with int64 ids) | [S10] |
| Python base | 9.12 GB after consumption (37.1 KB/position); 39.4 GB peak during build (160 KB/position) | [S13] |
| Python P3 | 1.27 GB after consumption; 5.6 GB peak, which includes the joined 2.2 GB body | [S13] |
| Rust base eager tree, n = 32 | 199–209 GiB (R0) and 215–222 GiB (eager toggle R4e), i.e. ≈ 6.2–6.9 GiB per request | [S2][S3][S17] |
| Rust R-A streamed, n = 32 | 14.47 GiB | [S4] |
| Rust R-B compact, n = 32 | 11.86 GiB with the presize; 14.83 GiB without | [S5] |

**Streamed render (R-A).**
- The writer task renders 64-position chunks into a channel bounded at 2 chunks.
- A dropped body stops the task. An early stop fails the body: hyper aborts the connection rather
  than ending truncated JSON cleanly.
- Peak memory per in-flight default response is therefore O(chunk).

**Segment presize (R-B).**
- The compact base64 body is a list of ≤ 1 MiB segments sent without copying.
- **Before:** each segment grew by doubling from 1 KiB, which left freed intermediate blocks in the
  allocator.
- **After:** reserving 1 MiB once per segment, and shrinking the last one, cut n32-rl peak RSS from
  14.83 GiB (14.63, 15.03) to 11.86 GiB (11.85, 11.86). All parsed was 1.81 vs 1.85 s, with
  overlapping runs, and bodies were identical [S5].
- The genbench peak fell from 414 MiB to 378 MiB [S6].

**Python, 2 build threads.** Two 245,760-position builds in memory at once add ≈ 4 GiB per busy
server: 8.0 → 11.8 GiB in the full cell [S9].
