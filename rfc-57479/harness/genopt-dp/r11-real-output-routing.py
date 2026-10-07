"""Round 11 check: how the REAL engine output thread routes outputs.

Runs the unmodified vllm.v1.engine.core.EngineCoreProc.process_output_sockets
(the engine's output IO thread) on a minimal stand-in `self` (only the
attributes that method reads: output_queue and _send_msg_tracking_payload),
with two frontend output (PULL) sockets, like an engine launched for two API
servers. Outputs are queued exactly as the scheduler/utility paths queue them:
(client_index, EngineCoreOutputs) where client_index = request.client_index
(Scheduler.update_from_output groups by request.client_index; utility
replies use the client_index carried in the utility request).

Case A: two frontends with distinct client indices 0 and 1 (Python API
servers): each gets its own outputs.
Case B: two frontend processes that both report client_index 0 (every Rust
bootstrapped frontend: the CLI has no client-index option): every output,
including the second process's request and its utility reply, is delivered to
output socket 0, and frontend 1 receives nothing.
"""

import json
import queue
import sys
import threading
import time

import zmq

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, UtilityOutput, UtilityResult
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.serial_utils import MsgpackDecoder


class Stub:
    _send_msg_tracking_payload = staticmethod(EngineCoreProc._send_msg_tracking_payload)

    def __init__(self):
        self.output_queue = queue.Queue()


def run_case(name, items):
    ctx = zmq.Context()
    pulls, addrs = [], []
    for _ in range(2):
        s = ctx.socket(zmq.PULL)
        port = s.bind_to_random_port("tcp://127.0.0.1")
        pulls.append(s)
        addrs.append(f"tcp://127.0.0.1:{port}")
    stub = Stub()
    t = threading.Thread(target=EngineCoreProc.process_output_sockets,
                         args=(stub, addrs, None, 0), daemon=True)
    t.start()
    for client_index, outputs in items:
        stub.output_queue.put_nowait((client_index, outputs))
    time.sleep(0.5)
    decoder = MsgpackDecoder(EngineCoreOutputs)
    received = {0: [], 1: []}
    for i, s in enumerate(pulls):
        while s.poll(100):
            out = decoder.decode(s.recv_multipart())
            received[i] += [o.request_id for o in out.outputs]
            if out.utility_output is not None:
                received[i].append(f"utility:{out.utility_output.call_id}")
    stub.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)
    t.join(timeout=5)
    ctx.destroy(linger=0)
    print(json.dumps({"case": name, "frontend_socket_0": received[0], "frontend_socket_1": received[1]}))
    return received


def outs(rid):
    return EngineCoreOutputs(outputs=[EngineCoreOutput(request_id=rid, new_token_ids=[1, 2])])


def util(call_id):
    return EngineCoreOutputs(utility_output=UtilityOutput(call_id=call_id, result=UtilityResult(None)))


if __name__ == "__main__":
    import vllm.v1.engine.core as core
    print(json.dumps({"engine_core_module": core.__file__}))
    a = run_case("A distinct client indices (Python API servers)",
                 [(0, outs("req-from-frontend-0")), (1, outs("req-from-frontend-1")),
                  (0, util(10)), (1, util(11))])
    b = run_case("B both frontends report client_index 0 (Rust bootstrapped frontends)",
                 [(0, outs("req-from-frontend-0")), (0, outs("req-from-frontend-1")),
                  (0, util(10)), (0, util(11))])
    ok = (a[0] == ["req-from-frontend-0", "utility:10"] and a[1] == ["req-from-frontend-1", "utility:11"]
          and b[1] == [] and sorted(b[0]) == sorted(["req-from-frontend-0", "req-from-frontend-1",
                                                      "utility:10", "utility:11"]))
    print(json.dumps({"result": "CONFIRMED: real engine routes by client_index; duplicate index 0 starves frontend 1"
                      if ok else "UNEXPECTED"}))
    sys.exit(0 if ok else 1)
