"""Experiment-only mock of ONE data-parallel engine rank (no model).

Speaks the engine side of the real vLLM DP protocol, mirroring
DPEngineCoreProc (vllm/v1/engine/core.py):

* frontends: one DEALER (identity = rank as 2-byte LE) + one PUSH per API
  server, ready response first (data_parallel_size/rank set);
* coordinator: XSUB to the coordinator back-publish socket (subscribe b"\\x01",
  wait b"READY", then receive START_DP_WAVE), PUSH to the coordinator
  back-output socket (scheduler_stats with step_counter/current_wave,
  wave_complete from rank 0, start_wave);
* lockstep stepping: every busy-loop iteration is one "forward"; with
  --lockstep (default) every forward is a collective over the DP group (MoE
  all-to-all stand-in), idle-but-running ranks execute dummy steps, and the
  finished/pause consensus is evaluated only when step_counter % 32 == 0
  (ParallelConfig.sync_dp_state semantics: OR(unfinished), AND(pending_pause));
* pause_scheduler(abort): abort everything (abort outputs to the owning
  frontend), PAUSED_NEW, two-phase DP pause (pending_pause -> consensus ->
  ignore_start_dp_wave), utility reply deferred until the engine is idle;
  resume_scheduler: same guards and all-reduce barrier as DPEngineCoreProc;
* START_DP_WAVE / stale-wave handling as DPEngineCoreProc.add_request and
  _handle_client_request.

The DP process group is a small ZMQ ROUTER hosted by rank 0 (sequence-numbered
collectives; a kind mismatch between ranks is logged and fatal, as a real
mismatched collective would hang).

Generation: each forward emits min(--tokens-per-step, remaining) tokens per
running request. A request finishes with LENGTH at max_tokens; if max_tokens
exceeds --output-tokens it holds after --output-tokens until aborted (the
legacy barrier/pause workload). Continuations (prompt longer than
--input-tokens) continue the deterministic token sequence at offset
len(prompt) - input_tokens.

Outputs and utility replies go to output socket number client_index (the
endpoint order = the frontends' client indices), like
EngineCoreProc.process_output_sockets (default since dp-round2).
--route-by connection reproduces the dp-round1 runs, which routed back on
the arriving connection. Mismatches between client_index and connection
index are counted and reported.
"""

import argparse
import asyncio
import importlib.util
import json
import time
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio

from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

spec = importlib.util.spec_from_file_location(
    "payload",
    Path(__file__).resolve().parent.parent / "genopt" / "frontend-mock-payload.py",
)
payload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(payload)

START_DP_WAVE = EngineCoreRequestType.START_DP_WAVE.value


class Log:
    def __init__(self, path, rank):
        self.out = open(path, "a", buffering=1)
        self.rank = rank

    def __call__(self, event, **fields):
        self.out.write(
            json.dumps(
                {"event": event, "rank": self.rank, "wall": time.time(),
                 "mono": time.monotonic(), **fields}
            )
            + "\n"
        )


class DPGroup:
    """Sequence-numbered collectives over ZMQ; rank 0 hosts the reducer."""

    def __init__(self, ctx, rank, size, address, log):
        self.rank, self.size, self.log = rank, size, log
        self.seq = 0
        self.sock = ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.IDENTITY, rank.to_bytes(2, "little"))
        if rank == 0:
            self.server = ctx.socket(zmq.ROUTER)
            self.server.bind(address)
            self.server_task = asyncio.create_task(self._serve())
        self.sock.connect(address)

    async def _serve(self):
        pending = {}
        while True:
            ident, seq, body = await self.server.recv_multipart()
            seq = int(seq)
            msg = json.loads(body)
            pending.setdefault(seq, {})[ident] = msg
            if len(pending[seq]) == self.size:
                msgs = pending.pop(seq)
                kinds = {m["kind"] for m in msgs.values()}
                if len(kinds) != 1:
                    self.log("collective_mismatch", seq=seq,
                             kinds={str(int.from_bytes(k, "little")): m["kind"] for k, m in msgs.items()})
                    reply = {"error": f"collective kind mismatch at seq {seq}: {sorted(kinds)}"}
                else:
                    reply = {
                        "unfinished": any(m["unfinished"] for m in msgs.values()),
                        "pending_pause": all(m["pending_pause"] for m in msgs.values()),
                    }
                data = json.dumps(reply).encode()
                for ident in msgs:
                    await self.server.send_multipart([ident, data])

    async def reduce(self, kind, unfinished, pending_pause=False):
        self.seq += 1
        await self.sock.send_multipart(
            [str(self.seq).encode(),
             json.dumps({"kind": kind, "unfinished": unfinished, "pending_pause": pending_pause}).encode()]
        )
        reply = json.loads(await self.sock.recv())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["unfinished"], reply["pending_pause"]


class Req:
    __slots__ = ("request", "conn", "offset", "max_tokens", "generated", "first_sent")

    def __init__(self, request, conn, offset):
        self.request, self.conn, self.offset = request, conn, offset
        self.max_tokens = request.sampling_params.max_tokens
        self.generated = 0
        self.first_sent = False


class Engine:
    def __init__(self, args):
        self.args = args
        self.rank = args.rank
        self.log = Log(args.log, args.rank)
        self.pool = payload.load_token_pool(args.token_pool)
        self.encoder = MsgpackEncoder()
        self.req_decoder = MsgpackDecoder(EngineCoreRequest)
        self.util_decoder = MsgpackDecoder()
        # DPEngineCoreProc state
        self.engines_running = False
        self.current_wave = 0
        self.step_counter = 0
        self.pending_pause = False
        self.ignore_start_dp_wave = False
        self.paused = False  # PauseState.PAUSED_NEW
        self.running: dict[str, Req] = {}
        self.held: dict[str, Req] = {}  # reached --output-tokens, waiting for abort
        self.queued: list = []  # adds received while paused (flushed on resume)
        self.idle_callbacks: list = []
        self.last_counts = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stats = {"admitted": 0, "tokens": 0, "finished_length": 0, "aborted": 0,
                      "client_index_mismatch": 0, "dummy_steps": 0, "steps": 0}

    # ------------------------------------------------------------ wiring
    async def connect(self):
        a = self.args
        self.ctx = zmq.asyncio.Context()
        endpoints = json.loads(Path(a.endpoints_json).read_text())
        self.dealers, self.pushes = [], []
        ident = self.rank.to_bytes(2, "little")
        for e in endpoints:
            d = self.ctx.socket(zmq.DEALER)
            d.setsockopt(zmq.IDENTITY, ident)
            d.connect(e["input"])
            p = self.ctx.socket(zmq.PUSH)
            p.connect(e["output"])
            self.dealers.append(d)
            self.pushes.append(p)
        self.coord_in = self.ctx.socket(zmq.XSUB)
        self.coord_in.connect(a.coord_in)
        self.coord_out = self.ctx.socket(zmq.PUSH)
        self.coord_out.connect(a.coord_out)
        await self.coord_in.send(b"\x01")
        ready = await self.coord_in.recv()
        assert ready == b"READY", ready
        self.log("coordinator_ready")
        self.group = DPGroup(self.ctx, self.rank, a.dp_size, a.dp_group_address, self.log)
        resp = EngineCoreReadyResponse(
            max_model_len=1048576, num_gpu_blocks=0, block_size=16,
            dp_stats_address=None, dtype="bfloat16", vllm_version="mock-dp",
            world_size=1, data_parallel_size=a.dp_size, tensor_parallel_size=1,
            pipeline_parallel_size=1, decode_context_parallel_size=1,
            data_parallel_rank=self.rank, max_num_seqs=256,
            max_num_batched_tokens=8192, instance_id=f"mock-dp-{self.rank}",
            supports_lora=False, max_loras=0,
        )
        ready_bytes = msgspec.msgpack.encode(resp)
        for d in self.dealers:
            await d.send(ready_bytes)
        for i, d in enumerate(self.dealers):
            asyncio.create_task(self._reader(i, d))
        asyncio.create_task(self._coord_reader())
        self.log("ready", frontends=len(self.dealers))

    async def _reader(self, conn, sock):
        while True:
            frames = await sock.recv_multipart()
            await self.queue.put(("fe", conn, frames))

    async def _coord_reader(self):
        while True:
            frames = await self.coord_in.recv_multipart()
            if frames[0] == b"READY":
                continue
            await self.queue.put(("coord", None, frames))

    async def send(self, conn, outputs: EngineCoreOutputs):
        outputs.engine_index = self.rank
        await self.pushes[conn].send_multipart(self.encoder.encode(outputs), copy=False)

    async def send_coord(self, outputs: EngineCoreOutputs):
        outputs.engine_index = self.rank
        await self.coord_out.send_multipart(self.encoder.encode(outputs))

    # ------------------------------------------------------------ requests
    def has_requests(self):
        return bool(self.running or self.held or self.queued)

    def has_unfinished(self):
        return bool(self.running or self.held)

    def has_work(self):
        return self.engines_running or self.has_requests()

    async def handle(self, item):
        source, conn, frames = item
        if source == "coord":
            kind, data = frames[0], frames[1]
            assert kind == START_DP_WAVE, kind
            new_wave, exclude = msgspec.msgpack.decode(data)
            acted = False
            if not self.ignore_start_dp_wave and exclude != self.rank and new_wave >= self.current_wave:
                self.current_wave = new_wave
                if not self.engines_running:
                    self.engines_running = True
                    acted = True
            # The Python frontend's FIRST_REQ carries the engine identity
            # bytes, which the coordinator forwards as-is; like the real
            # engine (int != bytes), such an exclusion never matches.
            self.log("start_dp_wave_recv", wave=new_wave,
                     exclude=exclude if not isinstance(exclude, bytes) else "bytes:" + exclude.hex(),
                     ignored=self.ignore_start_dp_wave, acted=acted, current_wave=self.current_wave)
            return
        kind = frames[0]
        data = frames[1:]
        if kind == EngineCoreRequestType.ADD.value:
            request = self.req_decoder.decode(data)
            if request.client_index != conn:
                self.stats["client_index_mismatch"] += 1
            conn = self.target(request.client_index, conn)
            if self.paused:
                self.queued.append((request, conn))
                self.log("queued_while_paused", request=request.request_id)
                return
            self.add_request(request, conn)
        elif kind == EngineCoreRequestType.ABORT.value:
            ids = self.util_decoder.decode(data)
            await self.abort(ids)
        elif kind == EngineCoreRequestType.UTILITY.value:
            client_index, call_id, method, margs = self.util_decoder.decode(data)
            await self.utility(self.target(client_index, conn), client_index, call_id, method, margs)
        else:
            raise RuntimeError(f"Unsupported request type {kind!r}")

    def target(self, client_index, conn):
        """Output socket for a request/utility: by client_index like
        EngineCoreProc.process_output_sockets (default), or by the arriving
        connection (--route-by connection, the dp-round1 behavior)."""
        if self.args.route_by == "connection":
            return conn
        if not 0 <= client_index < len(self.pushes):
            raise IndexError(f"client_index {client_index} has no output socket")
        return client_index

    def add_request(self, request, conn):
        a = self.args
        if request.request_id in self.running or request.request_id in self.held:
            raise RuntimeError(f"Duplicate request ID {request.request_id}")
        prompt_len = len(request.prompt_token_ids or [])
        offset = prompt_len - a.input_tokens
        sp = request.sampling_params
        if offset < 0 or sp is None or sp.logprobs != 128 or not (
            sp.ignore_eos or (sp._eos_token_id is None and not sp.stop_token_ids)
        ):
            raise RuntimeError(f"Request differs from experiment: prompt={prompt_len} sp={sp!r}")
        self.running[request.request_id] = Req(request, conn, offset)
        self.stats["admitted"] += 1
        self.log("admitted", request=request.request_id, conn=conn,
                 client_index=request.client_index, req_wave=request.current_wave,
                 current_wave=self.current_wave, running=self.engines_running,
                 offset=offset, max_tokens=sp.max_tokens)
        # DPEngineCoreProc.add_request
        if request.current_wave != self.current_wave:
            if request.current_wave > self.current_wave:
                self.current_wave = request.current_wave
            elif not self.engines_running and not self.paused:
                self.engines_running = True
                self.log("stale_wave_request", req_wave=request.current_wave,
                         current_wave=self.current_wave)
                self._coord_pending.append(EngineCoreOutputs(start_wave=self.current_wave))

    async def abort(self, ids):
        by_conn = {}
        for rid in ids:
            r = self.running.pop(rid, None) or self.held.pop(rid, None)
            if r is not None:
                by_conn.setdefault(r.conn, []).append(rid)
        for conn, rids in by_conn.items():
            self.stats["aborted"] += len(rids)
            await self.send(conn, EngineCoreOutputs(
                outputs=[EngineCoreOutput(request_id=rid, new_token_ids=[], finish_reason=FinishReason.ABORT)
                         for rid in rids],
                finished_requests=set(rids)))
        if by_conn:
            self.log("aborted", count=sum(len(v) for v in by_conn.values()))

    async def reply(self, conn, call_id, result=None, failure=None):
        await self.send(conn, EngineCoreOutputs(utility_output=UtilityOutput(
            call_id=call_id, failure_message=failure,
            result=UtilityResult(result) if failure is None else None)))

    async def utility(self, conn, client_index, call_id, method, margs):
        self.log("utility", method=method, conn=conn, client=client_index)
        if method == "pause_scheduler":
            mode = margs[0] if margs else "abort"
            if mode != "abort":
                await self.reply(conn, call_id, failure=f"mock supports abort pause only, got {mode}")
                return
            await self.abort(list(self.running) + list(self.held))
            self.paused = True
            # DPEngineCoreProc._pause_complete: two-phase pause.
            self.pending_pause = True
            self.engines_running = True
            self.log("pause_pending", wave=self.current_wave)
            self.idle_callbacks.append((conn, call_id, method))
        elif method == "resume_scheduler":
            if self.pending_pause or (self.engines_running and self.ignore_start_dp_wave):
                await self.reply(conn, call_id, failure=(
                    "resume_scheduler called while pause is still in flight. "
                    "Wait for the pause future to resolve before resuming."))
                return
            if self.engines_running:
                self.log("resume_ignored_running")
                await self.reply(conn, call_id)
                return
            self.paused = False
            self.ignore_start_dp_wave = False
            for request, rconn in self.queued:
                self.add_request(request, rconn)
            self.queued.clear()
            unfinished, _ = await self.group.reduce("resume_barrier", self.has_unfinished())
            if unfinished:
                self.engines_running = True
            self.log("resumed", global_unfinished=unfinished, wave=self.current_wave)
            await self.reply(conn, call_id)
        elif method == "is_scheduler_paused":
            await self.reply(conn, call_id, self.paused)
        elif method == "get_supported_tasks":
            await self.reply(conn, call_id, ("generate",))
        elif method in ("reset_mm_cache", "reset_prefix_cache", "reset_encoder_cache"):
            await self.reply(conn, call_id)
        else:
            await self.reply(conn, call_id, failure=f"Unsupported mock utility: {method}")

    # ------------------------------------------------------------ loop
    async def publish_counts(self):
        counts = (len(self.running) + len(self.held), len(self.queued))
        if counts != self.last_counts:
            self.last_counts = counts
            await self.send_coord(EngineCoreOutputs(scheduler_stats=SchedulerStats(
                num_running_reqs=counts[0], num_waiting_reqs=counts[1],
                step_counter=self.step_counter, current_wave=self.current_wave,
                kv_cache_usage=0.0)))

    async def flush_coord(self):
        while self._coord_pending:
            out = self._coord_pending.pop(0)
            await self.send_coord(out)
            self.log("start_wave_sent", wave=out.start_wave, reason="stale_request")

    async def step(self) -> bool:
        a = self.args
        if self.paused or not self.running:
            return False
        per_conn = {}
        finished = {}
        for rid, r in list(self.running.items()):
            limit = min(r.max_tokens, a.output_tokens)
            count = min(a.tokens_per_step, limit - r.generated)
            if count <= 0:
                continue
            out = payload.make_engine_output(rid, self.pool, r.offset + r.generated, count,
                                             prompt_tokens=a.input_tokens)
            if a.int32_ids:
                # Real sampler: logprob token ids are int32 (Sampler.gather_logprobs
                # "Use int32 to reduce the tensor size"); ranks stay int64.
                out.new_logprobs = out.new_logprobs._replace(
                    logprob_token_ids=out.new_logprobs.logprob_token_ids.astype("int32"))
            r.generated += count
            self.stats["tokens"] += count
            if not r.first_sent:
                r.first_sent = True
                self.log("first_token", request=rid, offset=r.offset)
            if r.generated >= r.max_tokens:
                out.finish_reason = FinishReason.LENGTH
                finished.setdefault(r.conn, set()).add(rid)
                del self.running[rid]
                self.stats["finished_length"] += 1
                self.log("finished_length", request=rid, tokens=r.generated)
            elif r.generated >= a.output_tokens:
                self.held[rid] = self.running.pop(rid)
                self.log("held", request=rid, tokens=r.generated)
            per_conn.setdefault(r.conn, []).append(out)
        for conn, outs in per_conn.items():
            await self.send(conn, EngineCoreOutputs(outputs=outs, finished_requests=finished.get(conn)))
        if a.step_ms:
            await asyncio.sleep(a.step_ms / 1000)
        return bool(per_conn)

    async def global_unfinished(self, local_unfinished):
        self.step_counter += 1
        at_sync = self.step_counter % 32 == 0
        if self.args.lockstep or at_sync:
            unfinished, consensus = await self.group.reduce(
                "sync" if at_sync else "step", local_unfinished, self.pending_pause)
        if not at_sync:
            return True
        if consensus:
            self.ignore_start_dp_wave = True
            self.pending_pause = False
            self.log("pause_consensus", wave=self.current_wave, step=self.step_counter)
        return unfinished

    async def run(self):
        self._coord_pending = []
        await self.connect()
        while True:
            # _process_input_queue
            while not self.has_work():
                for conn, call_id, method in self.idle_callbacks:
                    await self.reply(conn, call_id)
                    self.log("pause_complete", method=method, conn=conn, wave=self.current_wave)
                self.idle_callbacks.clear()
                await self.handle(await self.queue.get())
                await self.flush_coord()
            while not self.queue.empty():
                await self.handle(self.queue.get_nowait())
            await self.flush_coord()
            was_running = self.engines_running
            await self.publish_counts()
            executed = await self.step()
            await self.publish_counts()
            local_unfinished = self.has_unfinished()
            if executed:
                self.stats["steps"] += 1
            else:
                if not local_unfinished and not self.engines_running:
                    continue
                self.stats["dummy_steps"] += 1
                await asyncio.sleep(self.args.idle_step_ms / 1000)
            self.engines_running = await self.global_unfinished(local_unfinished)
            if not self.engines_running:
                if self.rank == 0:
                    await self.send_coord(EngineCoreOutputs(wave_complete=self.current_wave))
                    self.log("wave_complete_sent", wave=self.current_wave)
                self.log("wave_end", wave=self.current_wave, stats=self.stats)
                self.current_wave += 1
                self.step_counter = 0
            elif not was_running and self.rank == 0 and not self.pending_pause:
                await self.send_coord(EngineCoreOutputs(start_wave=self.current_wave))
                self.log("start_wave_sent", wave=self.current_wave, reason="rank0_started")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--dp-size", type=int, required=True)
    p.add_argument("--endpoints-json", required=True)
    p.add_argument("--coord-in", required=True, help="coordinator back-publish (XPUB) address")
    p.add_argument("--coord-out", required=True, help="coordinator back-output (PULL) address")
    p.add_argument("--dp-group-address", required=True)
    p.add_argument("--token-pool", type=Path, required=True)
    p.add_argument("--input-tokens", type=int, default=16384)
    p.add_argument("--output-tokens", type=int, default=245760)
    p.add_argument("--tokens-per-step", type=int, default=1024)
    p.add_argument("--step-ms", type=float, default=0.0)
    p.add_argument("--lockstep", type=int, default=1)
    p.add_argument("--int32-ids", action="store_true",
                   help="opt-in: send logprob token ids as int32 like the real sampler (default int64, as all earlier runs)")
    p.add_argument("--idle-step-ms", type=float, default=1.0,
                   help="duration of a dummy forward (idle-but-running rank, or only held requests)")
    p.add_argument("--log", required=True)
    p.add_argument("--route-by", choices=("client-index", "connection"), default="client-index",
                   help="client-index = real engine routing; connection = dp-round1 runs")
    asyncio.run(Engine(p.parse_args()).run())
