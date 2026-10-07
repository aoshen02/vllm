# Full results

All times are seconds after the pause unless stated. Values are medians [min–max] over 3 repeats
unless stated. HWM is peak RSS: per process for Python, total for Rust (one process). Source keys
as in the RFC.

## 1. DP8 reference run, earlier candidates (dp-round6) [S2]
Rust `7dec4604e2` (1 process, 64 threads), Python in-tree `214f522` (32 servers per node).

| Cell | Rust | Python |
|---|---|---|
| cohort-rl-v3: all parsed | 10.97 [10.86–11.01] | 10.85 [10.83–10.89] |
| cohort-rl-v3: pause / consumption / ingest | 0.030 s / 49.5 s / 1.27 M pos/s | 0.161 s / 21.4 s / 2.95 M pos/s |
| cohort-rl-wide: all parsed | 4.68 [4.53–4.68] | 4.24 [4.14–4.26] |
| cohort-def-wide: all parsed | 59.4 [58.9–60.7], 873 GB | 81.1 [73.0–83.4], 563 GB |
| wf-paced: pause / partials (max) / resume → first token | 0.059 / 0.555 / 0.290 | 0.066 / 0.513 / 0.294 |
| n32-def: candidate | 20.6 [20.2–21.0] | 21.8 [20.2–23.6] (busiest 3 [2–3]) |
| n32-def: base | 33.2 [32.6–33.2], HWM 199 GiB | FAIL ×3: memory floor (node free 61.9–63.0 GiB, 36 GiB per process, after about 8 min) |
| n32-rl | 1.74 [1.73–1.75] | 1.82 [1.77–1.88] |
| Frontend HWM, bs = 256 cells | 111–116 GiB | 4–8 GiB per process |

## 2. Python stack, DP8 final validation (dp-round10) [S7]
Arms:
- **B:** the combined tip `b27991bf03` = P1 `b1eaa4badc` + P2 `b2f0977148` + P3 `c635bcc564` + plugin, with no listeners.
- **A:** in-tree `214f522`, ABBA in the default cells.

{{TBD: replace with the run on the final Python tips}}

| Cell | Arm | Status | All parsed | Received | Pause | Busiest | HWM GiB |
|---|---|---|---|---|---|---|---|
| cohort-def-wide | A | 3 PASS | 67.045 [63.722–74.376] | 64.809 | 0.064 | 8 [7–10] | 7.572 |
| cohort-def-wide | B | 3 PASS | 76.066 [72.212–77.965] | 73.845 | 0.082 | 10 [9–10] | 7.992 |
| cohort-rl-v3 | B | 3 PASS | 10.872 [10.808–10.9] | 10.056 | 0.18 | 8 | 3.831 |
| cohort-rl-wide | B | 3 PASS | 4.178 [4.163–4.238] | 3.911 | 0.163 | 7 [7–8] | 3.814 |
| n32-def | A | 3 PASS | 21.624 [20.12–21.8] | 16.4 | 0.047 | 2 | 5.956 |
| n32-def | B | 3 PASS | 21.589 [21.543–21.727] | 16.655 | 0.04 | 2 | 5.979 |
| n32-rl | B | 3 PASS | 1.8 [1.784–1.817] | 0.558 | 0.066 | 2 | 2.041 |
| wf-paced | B | 3 PASS | pause 0.078, partials 0.584, resume → first token 0.294; 196 aborted, 196 resubmitted and finished | | | 12 [12–13] | 1.958 |

**Bytes:** 1 distinct per-request byte-length list per cell.

| Cell | Bytes |
|---|---|
| cohort-def-wide | 563,104,591,360 |
| n32-def | 70,388,073,920 |
| cohort-rl-* | 86,283,132,160 |
| n32-rl | 10,785,391,520 |

**Without P3** (dp-round9) [S8]:
- Both default cells FAIL 3/3 at the node memory floor, with the largest server at 35.6–36.2 GiB (n32-def) and 30–75 GiB (cohort-def-wide).
- The PR2 core-only tree and the frozen baseline fail the same way.

## 3. Rust stack (round 16 / 17) [S4][S5][S6][S26]

**n32-def ABBA:** old PR 2 (with the fragment cache) against R-A (without it).

| Arm | All parsed | HWM |
|---|---|---|
| Old PR 2 | 20.75 s (20.36, 21.14) | 14.46 GiB |
| R-A | 20.73 s (20.31, 21.15) | 14.47 GiB |

Bodies are identical in both arms.

**n32-rl ABBA:** R-B before and after the segment presize.

| Arm | All parsed | HWM |
|---|---|---|
| Without presize | 1.81 s (1.75, 1.88) | 14.83 GiB (14.63, 15.03) |
| With presize | 1.85 s (1.83, 1.87) | 11.86 GiB (11.85, 11.86) |

Bodies are identical in both arms.

**Base n32-def** (fv/pr4-47813db315, arm A, 3 runs) [S3]: 32.91 [32.49–33.43] s, HWM 201.6 GiB.

**Render micro-bench** (one core, median of 3) [S6]:

| Build | Time |
|---|---|
| Base eager serde | 6.77 s |
| R-A | 1.10 s |
| Old PR 2 with cache | 0.71 s |
| R-B compact, encode + render | 0.125 s at `bc60a7b3ed`; 0.141 s (1 run) at the final `d1d9770b7b`, with the int32 check |

**Final checks** [S26]:
- R-A `e9c06ad173`: 1611 passed.
- R-B `d1d9770b7b`: 1623 passed.
- Clippy 0, fmt clean, shas unchanged.

**bs = 256 cells on the previous Rust tips** (fv/pr4-47813db315 with parallel decode; fv/pr3-2595b4aa67 without) [S3]:

| Cell | PR 4 tip | PR 3 tip |
|---|---|---|
| cohort-rl-v3 | 10.91 [10.88–10.97] | 10.92 |
| cohort-rl-wide | 4.55 [4.41–5.09] | — |
| cohort-def-wide | 63.3 [62.7–65.0] (HWM 111 GiB) | — |
| wf-paced: pause / partials / resume | 0.071 / 0.579 / 0.290 | — |
| n32-def | 21.13 [20.19–21.96] | 20.52 |

## 4. Python ablation, n = 32, earlier two-node harness (r14) [S12]

| Step | Busiest | Consumption | Pause | Server ready | All parsed | Frontend CPU | HWM |
|---|---|---|---|---|---|---|---|
| T0 base | 2 / 1 | 322.3 / 172.8 | 139.8 / 138.7 | 270.5 / 156.6 | 273.5 / 159.5 | 4,520 / 4,595 | 41.0 / 33.2 GiB |
| T1 + listeners | 2 / 3 | 326.8 / 489.8 | 134.4 / 132.4 | 280.2 / 410.7 | 283.2 / 413.9 | 4,458 / 4,347 | 40.9 / 49.2 |
| T2 + storage only | 3 / 3 | 10.0 / 9.8 | 0.031 / 0.032 | 433.3 / 453.2 | 436.4 / 456.4 | 4,851 / 4,837 | 37.7 / 37.4 |
| T3 = T4 default | 3 / 2 | 10.0 / 9.8 | 0.032 / 0.031 | 15.8 / 11.6 | 18.8 / 18.5 | 203 / 213 | 6.3 / 6.3 |
| T4 compact | 2 / 3 | 10.1 / 9.9 | 0.032 / 0.032 | 0.06 / 0.10 | 1.72 / 1.78 | 2.8 / 3.3 | 2.0 / 2.4 |

**Single request** (cn01):

| Step | Consumption | Build after the abort | HWM |
|---|---|---|---|
| T0 | 373 µs/pos | 127 s | 39.2 GB |
| T2 | 0.34 µs/pos | 148 s, of which 94 s is materialisation | 34.3 GB |
| T4 | 0.34 µs/pos | 4.5 s | 5.6 GB |

## 5. Rust ablation, deployable single process, DP1, n = 32 (r12) [S17]

| Step | All parsed | Consumption | HWM |
|---|---|---|---|
| R0 base | 34.28 / 33.42 | 15.0 / 14.1 | 209 / 201 GiB |
| R1 + decode/accumulator | 34.46 / 32.61 | 9.8 / 10.1 | 198 / 206 |
| R4e (eager render) | 34.08 / 33.77 | 9.7 / 10.4 | 222 / 215 |
| R4 streamed render | 20.46 / 20.82 | 11.0 / 9.6 | 14.2 / 14.4 |
| R4 compact | 1.83 / 1.78 | 10.0 / 16.2 | 14.6 / 14.5 |

## 6. Python single-server render scaling (dp-round11) [S9]

| Arm | N | All parsed | Build CPU per request | HWM |
|---|---|---|---|---|
| P3 `c635bcc564` | 10 | 47.31 [47.22–47.46] | 4.426 s | 8.78 GiB |
| In-tree `214f522` | 10 | 47.09 [47.02–47.16] | 4.404 s | 8.76 GiB |
| Stack + plugin | 10 | 47.03 [46.75–47.36] | 4.397 s | 8.79 GiB |
| P3 | 8 | 38.12 [37.93–38.31] | 4.402 s | 8.70 GiB |
| In-tree | 8 | 38.19 [38.13–38.32] | 4.397 s | 8.72 GiB |
| P3, 2 build threads | 10 | 42.51 (n = 2) | 4.797 s | 12.95 GiB |
| P3, 4 build threads | 10 | 42.75 (n = 2) | 5.053 s | 18.79 GiB |
| P3, 10 build threads | 10 | 46.11 (n = 2) | 5.412 s | 37.52 GiB |

**Full cohort-def-wide cell with 2 build threads** (3 runs): 64.1 [61.7–66.0] s against 76.1 s with 1 thread; HWM 11.8 GiB.

| Busiest | Measured | 1-thread fit | Difference |
|---|---|---|---|
| 10 | 64.1 | 76.9 | −12.8 |
| 8 | 61.7 | 67.7 | −6.0 |
| 9 | 66.0 | 72.3 | −6.3 |

{{TBD: round 24 decision and the measurement on the final tip}}

## 7. Mock granularity A/B (dp-round7) [S10]
DP8, n = 256, fixed 16,384 tokens, wide client, 2 runs per arm.
- **A:** 1,024 tokens per step, int64 ids.
- **B:** 1 token per step, int32 ids.

| Cell | A: all parsed | B: all parsed | A: ingest | B: ingest |
|---|---|---|---|---|
| Rust RL-lean | 0.400 | 0.393 | 819 k pos/s | 159 k outputs/s |
| Rust default | 4.27 | 4.34 | 898 k | 151 k |
| Python RL-lean | 0.282 | 0.277 | 1,150 k | 77 k |
| Python default | 4.44 | 4.39 | 1,140 k | 73 k |

## 8. Request distribution: shared socket vs `SO_REUSEPORT` (#58278) vs round-robin (dp-round12) [S24]

**Tree:** the Python round-24 tip `7e533ad05d` plus #58278's listener lines (35 +/− lines, identical to
`gh pr diff 58278`), with the plugin allowlisted in every arm.

**Environment:**
- DP8 mock engines;
- client v3.1, one connection per request;
- wide client for L4, L16 and RL.

**Arms:**
- S: shared socket (`--no-listeners`).
- R: per-server `SO_REUSEPORT` listeners.
- RR: one port per server, with request i sent to server i mod k (harness options `--round-robin-ports` / `--per-server-ports`).

All 49 runs PASS.

| Load | Arm | Runs | Busiest per run | Busiest median | Per-server sd | All parsed, s, median [min–max] | Peak memory per process, GiB |
|---|---|---|---|---|---|---|---|
| L1: 32 requests, 32 servers, default | S | 6 | 2,2,2,2,2,2 | 2 | 0.46 | 21.2 [20.5–21.6] | 6.13 |
| | R | 6 | 5,3,4,3,5,3 | 3.5 | 1.06 | 22.2 [20.0–30.6] | 6.47 |
| | RR | 3 | 1,1,1 | 1 | 0 | 22.0 [21.8–22.0] | 5.78 |
| L4: 256 requests, 64 servers, default | S | 6 | 7,8,8,8,8,7 | 8 | 1.46 | 61.6 [59.6–64.4] | 7.38 |
| | R | 6 | 8,10,10,10,8,10 | 10 | 1.95 | 66.2 [59.8–69.0] | 7.86 |
| | RR | 2 | 4,4 | 4 | 0 | 51.0 [50.2–51.8] | 6.85 |
| L16: 256 requests, 16 servers, default | S | 2 | 19,18 | 18.5 | 1.41 | 101.8 [99.9–103.7] | 10.58 |
| | R | 2 | 21,25 | 23 | 3.37 | 117.1 [110.0–124.2] | 11.25 |
| | RR | 2 | 16,16 | 16 | 0 | 102.1 [92.6–111.7] | 9.62 |
| RL-lean: 256 requests, 64 servers | S | 6 | 8,9,9,10,7,8 | 8.5 | 1.50 | 4.23 [4.17–4.35] | 4.02 |
| | R | 6 | 9,9,8,10,12,9 | 9 | 1.99 | 4.14 [4.13–4.29] | 4.17 |
| | RR | 2 | 4,4 | 4 | 0 | 4.29 [4.27–4.30] | 3.02 |

The balls-into-bins table and the re-fit are in [theory.md §4](theory.md).

**Earlier n = 32 results, two-node harness:**
- With the in-tree fast render (r15) [S16]: 19.05 / 20.21 s with listeners against 18.17 / 18.42 s without. RL-lean is equal (1.83–1.89 s).
- With base code, see §4 (T1).
