"""HTTP-framing fake server for client v3 (test only). Copied from the Claude
audit (audits/client3-claude-work/fakesrv.py) and extended with round-2 modes.

Honours the harness protocol (consumption log written once all N requests
arrived; responses held until POST /pause) and answers with a recorded body
using the framing selected by --mode.
"""
import argparse
import gzip
import json
import random
import re
import socket
import threading
import time
from pathlib import Path

a = None
lock = threading.Lock()
cv = threading.Condition(lock)
state = {"arrived": 0, "paused": False}


def read_request(c):
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = c.recv(65536)
        if not d:
            return None, None, None
        buf += d
    head, rest = buf.split(b"\r\n\r\n", 1)
    m = re.search(rb"(?i)content-length:\s*(\d+)", head)
    cl = int(m.group(1)) if m else 0
    while len(rest) < cl:
        d = c.recv(1 << 20)
        if not d:
            break
        rest += d
    path = head.split(b" ")[1].decode()
    return path, head, rest


def send_chunked(c, body, sizes, ext=b"", trailer=b"", terminate=True, upper=False, pad=False):
    i = 0
    r = random.Random(1)
    while i < len(body):
        n = min(r.choice(sizes), len(body) - i)
        h = (b"%X" if upper else b"%x") % n
        if pad:
            h = b"000" + h
        c.sendall(h + ext + b"\r\n" + body[i:i + n] + b"\r\n")
        i += n
    if terminate:
        c.sendall(b"0" + ext + b"\r\n" + trailer + b"\r\n")


def handle(c):
    try:
        path, head, req = read_request(c)
        if path is None:
            return
        if path.startswith("/pause"):
            with cv:
                state["paused"] = True
                cv.notify_all()
            if a.mode == "pause-trickle":
                # headers promptly, then the body one byte per second, never finishing
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\n\r\n")
                for _ in range(1000):
                    c.sendall(b" ")
                    time.sleep(1)
                return
            if a.mode == "pause500":
                c.sendall(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            else:
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            return
        m = re.search(rb'"request_id": "mock-(\d+)"', req)
        idx = int(m.group(1))
        body = (Path(a.bodies) / f"body-{idx:04d}.json").read_bytes()
        if a.mode == "early" and idx == 0:
            c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
            return
        with cv:
            state["arrived"] += 1
            if state["arrived"] == a.requests:
                recs = [{"request_id": f"mock-{k:04d}", "tokens": a.tokens, "logprob_positions": a.tokens}
                        for k in range(a.requests)]
                if a.mode == "dupconsumed":
                    recs[-1]["request_id"] = recs[0]["request_id"]
                if a.mode == "badcount":
                    recs[-1]["tokens"] = a.tokens - 1
                if a.mode == "noreqid":
                    del recs[-1]["request_id"]
                if a.mode == "nonstr-reqid":
                    recs[-1]["request_id"] = a.requests - 1
                if a.mode == "wrong-reqid-set":
                    recs[-1]["request_id"] = "mock-0005"
                if a.mode == "extra-consumed":
                    recs.append(dict(recs[0], request_id="mock-0099"))
                text = "".join(json.dumps(r) + "\n" for r in recs)
                tmp = Path(a.log) / ".tmp"
                tmp.write_text(text)
                tmp.rename(Path(a.log) / "consumed-fake.jsonl")
            while not state["paused"]:
                cv.wait()
        mode = a.mode
        ok = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        if mode in ("cl", "early", "pause500", "dupconsumed", "badcount", "noreqid", "nonstr-reqid",
                    "wrong-reqid-set", "extra-consumed", "pause-trickle"):
            c.sendall(ok + b"Content-Length: %d\r\n\r\n" % len(body) + body)
        elif mode == "status500" and idx == 1:
            c.sendall(b"HTTP/1.1 500 Oops\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        elif mode == "status500":
            c.sendall(ok + b"Content-Length: %d\r\n\r\n" % len(body) + body)
        elif mode == "chunked":
            c.sendall(ok + b"Transfer-Encoding: chunked\r\n\r\n")
            send_chunked(c, body, [1, 2, 3, 7, 64, 4095, 4096, 65536, 70001])
        elif mode == "chunked-ext-trailer":
            c.sendall(ok + b"Transfer-Encoding: chunked\r\nTrailer: X-T\r\n\r\n")
            send_chunked(c, body, [1000, 65536], ext=b";foo=bar", trailer=b"X-T: 1\r\nX-U: 2\r\n", upper=True, pad=True)
        elif mode == "chunked-te-and-cl":
            c.sendall(ok + b"Content-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n")
            send_chunked(c, body, [65536])
        elif mode == "chunked-trickle":
            # tiny TCP writes: chunk-size lines / CRLFs split across reads
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            c.sendall(ok + b"Transfer-Encoding: chunked\r\n\r\n")
            data = b""
            for i in range(0, len(body), 1 << 20):
                part = body[i:i + (1 << 20)]
                data += b"%x\r\n" % len(part) + part + b"\r\n"
            data += b"0\r\n\r\n"
            j = 0
            r = random.Random(2)
            while j < len(data):
                # around chunk headers send 1 byte at a time
                k = 1 if r.random() < 0.002 else r.choice([3, 5, 100000])
                c.sendall(data[j:j + k])
                j += k
        elif mode == "chunked-noterm":
            c.sendall(ok + b"Transfer-Encoding: chunked\r\n\r\n")
            send_chunked(c, body, [65536], terminate=False)
        elif mode == "chunked-trunc-at-boundary":
            # valid chunked framing, but the body stops at a chunk boundary (half of it)
            c.sendall(ok + b"Transfer-Encoding: chunked\r\n\r\n")
            send_chunked(c, body[: (len(body) // 2 // 65536) * 65536], [65536])
        elif mode == "chunked-trailing-junk":
            c.sendall(ok + b"Transfer-Encoding: chunked\r\n\r\n")
            send_chunked(c, body, [65536])
            c.sendall(b"GARBAGE AFTER TERMINATOR")
        elif mode == "close":
            c.sendall(ok + b"Connection: close\r\n\r\n" + body)
        elif mode == "cl-short":
            c.sendall(ok + b"Content-Length: %d\r\n\r\n" % (len(body) - 7) + body)
        elif mode == "cl-long":
            c.sendall(ok + b"Content-Length: %d\r\n\r\n" % (len(body) + 10) + body)
        elif mode == "cl-dup-conflict":
            c.sendall(ok + b"Content-Length: %d\r\nContent-Length: %d\r\n\r\n" % (len(body) + 10, len(body)) + body)
        elif mode == "gzip":
            z = gzip.compress(body)
            c.sendall(ok + b"Content-Encoding: gzip\r\nContent-Length: %d\r\n\r\n" % len(z) + z)
        elif mode == "100cont":
            c.sendall(b"HTTP/1.1 100 Continue\r\n\r\n" + ok + b"Content-Length: %d\r\n\r\n" % len(body) + body)
        elif mode == "stall":
            c.sendall(ok + b"Content-Length: %d\r\n\r\n" % len(body) + body[: len(body) // 2])
            time.sleep(a.stall)
            c.sendall(body[len(body) // 2:])
        elif mode == "head-stall":
            time.sleep(a.stall)
            c.sendall(ok + b"Content-Length: %d\r\n\r\n" % len(body) + body)
        elif mode == "http10":
            c.sendall(b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n" + body)
        elif mode == "lf-only":
            c.sendall(b"HTTP/1.1 200 OK\nContent-Length: %d\n\n" % len(body) + body)
        elif mode == "huge-cl":
            c.sendall(ok + b"Content-Length: 18446744073709551615\r\n\r\n" + body)
        elif mode == "huge-chunk":
            c.sendall(ok + b"Transfer-Encoding: chunked\r\n\r\nffffffffffffffff\r\n" + body)
        elif mode == "cl-dup-same":
            c.sendall(ok + b"Content-Length: %d\r\nContent-Length: %d\r\n\r\n" % (len(body), len(body)) + body)
        elif mode == "te-gzip-chunked":
            c.sendall(ok + b"Transfer-Encoding: gzip, chunked\r\n\r\n")
            send_chunked(c, body, [65536])
        elif mode == "te-twice":
            c.sendall(ok + b"Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n\r\n")
            send_chunked(c, body, [65536])
        elif mode == "chunk-lf-delim":
            c.sendall(ok + b"Transfer-Encoding: chunked\r\n\r\n")
            c.sendall(b"%x\r\n" % len(body) + body + b"\n0\r\n\r\n")
        elif mode == "chunk-size-lf":
            c.sendall(ok + b"Transfer-Encoding: chunked\r\n\r\n")
            c.sendall(b"%x\n" % len(body) + body + b"\r\n0\r\n\r\n")
        elif mode == "bad-trailer":
            c.sendall(ok + b"Transfer-Encoding: chunked\r\n\r\n")
            send_chunked(c, body, [65536], trailer=b"Bad Trailer Line\r\n")
        elif mode == "obs-fold":
            c.sendall(ok + b"X-A: 1\r\n  folded\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        elif mode == "reason-ctl":
            c.sendall(b"HTTP/1.1 200 O\x01K\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        elif mode == "no-reason":
            c.sendall(b"HTTP/1.1 200\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        elif mode == "te-empty":
            c.sendall(ok + b"Transfer-Encoding:\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        elif mode == "te-comma-chunked":
            c.sendall(ok + b"Transfer-Encoding: ,chunked,\r\n\r\n")
            send_chunked(c, body, [65536])
        elif mode == "no-response":
            time.sleep(100)
        else:
            raise SystemExit(f"unknown mode {mode}")
    except Exception as e:  # noqa: BLE001
        print("handler error", repr(e), flush=True)
    finally:
        try:
            c.shutdown(socket.SHUT_WR)
            time.sleep(0.2)
        except OSError:
            pass
        c.close()


def main():
    global a
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--bodies", required=True)
    p.add_argument("--requests", type=int, default=2)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--log", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--stall", type=float, default=0)
    a = p.parse_args()
    Path(a.log).mkdir(parents=True, exist_ok=True)
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", a.port))
    s.listen(64)
    print("listening", a.port, flush=True)
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


main()
