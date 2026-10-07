"""Run the REAL vLLM DPCoordinatorProc on fixed TCP addresses, plus a spy.

The coordinator is vllm.v1.engine.coordinator.DPCoordinatorProc from the
PYTHONPATH source (unmodified; run_coordinator is called exactly as
DPCoordinator.__init__ does, minus the address pipe). Launch with
VLLM_LOGGING_LEVEL=DEBUG to get its wave transitions in the log.

The spy thread is an extra frontend-side subscriber on the front publish
socket (like an API server's stats task) that logs every publish
(counts, wave, engines_running) with timestamps; it never sends anything but
the standard subscription.
"""

import argparse
import hashlib
import json
import threading
import time

import msgspec
import zmq


def spy(address, path):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.connect(address)
    with open(path, "a", buffering=1) as out:
        while True:
            buf = sock.recv()
            counts, wave, running = msgspec.msgpack.decode(buf)
            out.write(json.dumps({"event": "publish", "wall": time.time(), "mono": time.monotonic(),
                                  "counts": counts, "wave": wave, "running": running}) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine-count", type=int, required=True)
    p.add_argument("--front-publish", required=True, help="tcp://ip:port frontends connect to")
    p.add_argument("--back-output", required=True)
    p.add_argument("--back-publish", required=True)
    p.add_argument("--spy-log", required=True)
    p.add_argument("--source-log", required=True)
    a = p.parse_args()
    import vllm.v1.engine.coordinator as coordinator

    with open(a.source_log, "w") as f:
        json.dump({"module": coordinator.__file__,
                   "sha256": hashlib.sha256(open(coordinator.__file__, "rb").read()).hexdigest()}, f)
    threading.Thread(target=spy, args=(a.front_publish, a.spy_log), daemon=True).start()
    coordinator.DPCoordinatorProc.run_coordinator(
        engine_count=a.engine_count,
        front_publish_address=a.front_publish,
        back_output_address=a.back_output,
        back_publish_address=a.back_publish,
        zmq_addr_pipe=None,
    )
