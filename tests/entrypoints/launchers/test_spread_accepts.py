# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NoSignalServer(spread_accepts=True): API workers on one shared socket accept
one connection at a time, so a burst of connections is spread over them."""

import asyncio
import datetime
import multiprocessing
import os
import socket
import ssl
from argparse import Namespace
from collections import Counter

import pytest
import uvicorn
import uvloop

from vllm.entrypoints.launchers.api_server.entry import shares_socket
from vllm.entrypoints.launchers.launcher import NoSignalServer

TIMEOUT_S = 30


@pytest.mark.parametrize(
    ("api_server_count", "expected"),
    [(None, False), (0, False), (1, False), (2, True), (8, True)],
)
def test_spread_only_with_several_api_workers(api_server_count, expected):
    assert shares_socket(Namespace(api_server_count=api_server_count)) is expected


def _bound_socket() -> socket.socket:
    # Like setup_server: bound, not listening (uvicorn / the acceptor listen).
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    return sock


def _app(hold_s: float = 0.0):
    pid = str(os.getpid()).encode()

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if hold_s:
            await asyncio.sleep(hold_s)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", str(len(pid)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": pid})

    return app


def _run(coro, loop_impl: str):
    if loop_impl == "uvloop":
        return uvloop.run(coro)
    return asyncio.run(coro)


async def _get(port: int, keep_alive: int = 1, ssl_ctx=None) -> list[bytes]:
    """keep_alive requests on one connection; returns the response bodies."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection("127.0.0.1", port, ssl=ssl_ctx), TIMEOUT_S
    )
    bodies = []
    for i in range(keep_alive):
        connection = b"close" if i == keep_alive - 1 else b"keep-alive"
        writer.write(
            b"GET / HTTP/1.1\r\nHost: t\r\nConnection: " + connection + b"\r\n\r\n"
        )
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), TIMEOUT_S)
        assert head.startswith(b"HTTP/1.1 200")
        length = int(head.split(b"content-length: ")[1].split(b"\r\n")[0])
        bodies.append(await asyncio.wait_for(reader.readexactly(length), TIMEOUT_S))
    writer.close()
    return bodies


async def _start(server: NoSignalServer, sock: socket.socket) -> asyncio.Task:
    task = asyncio.create_task(server.serve(sockets=[sock]))

    async def started():
        while not server.started:
            if task.done():
                task.result()  # startup failed: raise its error
            await asyncio.sleep(0.01)

    await asyncio.wait_for(started(), TIMEOUT_S)
    return task


async def _serve(sock, spread_accepts, requests, **config_kwargs):
    config = uvicorn.Config(_app(), lifespan="on", log_level="warning", **config_kwargs)
    server = NoSignalServer(config, spread_accepts=spread_accepts)
    task = await _start(server, sock)
    try:
        return server, await requests(sock.getsockname()[1])
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, TIMEOUT_S)


@pytest.mark.parametrize("loop_impl", ["asyncio", "uvloop"])
@pytest.mark.parametrize("http", ["h11", "httptools"])
@pytest.mark.parametrize("spread_accepts", [True, False])
def test_request_and_keep_alive_served(spread_accepts, http, loop_impl):
    if http == "httptools":
        pytest.importorskip("httptools")
    sock = _bound_socket()

    async def requests(port):
        return await _get(port), await _get(port, keep_alive=3)

    server, (one, three) = _run(
        _serve(sock, spread_accepts, requests, http=http), loop_impl
    )
    assert one == [str(os.getpid()).encode()]
    assert len(three) == 3
    # Only spread_accepts replaces uvicorn's asyncio server with the acceptor.
    assert bool(server.servers) is not spread_accepts
    sock.close()


@pytest.mark.parametrize("loop_impl", ["asyncio", "uvloop"])
def test_acceptor_stopped_on_shutdown(loop_impl):
    sock = _bound_socket()
    fd = sock.fileno()

    async def run():
        server, _ = await _serve(sock, True, _get)
        loop = asyncio.get_running_loop()
        # No reader, no pending re-arm and no listener left after shutdown.
        assert not loop.remove_reader(fd)
        assert not server._accept_resume and not server._connecting
        assert sock.fileno() == -1
        await asyncio.sleep(0.01)
        assert not loop.remove_reader(fd)

    _run(run(), loop_impl)


def test_protocol_error_closes_connection_and_keeps_serving():
    sock = _bound_socket()

    async def run():
        config = uvicorn.Config(_app(), lifespan="on", log_level="warning")
        config.load()
        protocol_class = config.http_protocol_class
        calls = []

        def flaky_protocol(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("protocol setup failed")
            return protocol_class(**kwargs)

        config.http_protocol_class = flaky_protocol
        server = NoSignalServer(config, spread_accepts=True)
        task = await _start(server, sock)
        try:
            port = sock.getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            # The failed connection is closed, not leaked or left hanging.
            assert await asyncio.wait_for(reader.read(), TIMEOUT_S) == b""
            writer.close()
            assert await _get(port) == [str(os.getpid()).encode()]
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, TIMEOUT_S)

    uvloop.run(run())


def _self_signed(tmp_path):
    x509 = pytest.importorskip("cryptography.x509")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    keyfile, certfile = tmp_path / "key.pem", tmp_path / "cert.pem"
    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(keyfile), str(certfile)


@pytest.mark.parametrize("loop_impl", ["asyncio", "uvloop"])
def test_tls_request_served_through_acceptor(tmp_path, loop_impl):
    keyfile, certfile = _self_signed(tmp_path)
    sock = _bound_socket()
    client_ctx = ssl.create_default_context()
    client_ctx.check_hostname = False
    client_ctx.verify_mode = ssl.CERT_NONE

    async def requests(port):
        return await _get(port, ssl_ctx=client_ctx)

    _, bodies = _run(
        _serve(sock, True, requests, ssl_keyfile=keyfile, ssl_certfile=certfile),
        loop_impl,
    )
    assert bodies == [str(os.getpid()).encode()]
    sock.close()


def _worker(sock, ready, stop):
    async def main():
        config = uvicorn.Config(_app(hold_s=0.2), lifespan="on", log_level="warning")
        server = NoSignalServer(config, spread_accepts=True)
        task = await _start(server, sock)
        ready.release()
        await asyncio.get_running_loop().run_in_executor(None, stop.wait)
        server.should_exit = True
        await task

    # vLLM API workers run on uvloop, whose accept loop drains whole bursts.
    uvloop.run(main())


@pytest.mark.skipif((os.cpu_count() or 1) < 8, reason="needs idle cores per worker")
def test_burst_spread_over_workers():
    workers, connections = 4, 400
    sock = _bound_socket()
    ctx = multiprocessing.get_context("spawn")
    ready, stop = ctx.Semaphore(0), ctx.Event()
    procs = [
        ctx.Process(target=_worker, args=(sock, ready, stop)) for _ in range(workers)
    ]
    for p in procs:
        p.start()
    try:
        for _ in range(workers):
            assert ready.acquire(timeout=120)

        async def burst():
            port = sock.getsockname()[1]
            results = await asyncio.gather(*(_get(port) for _ in range(connections)))
            return Counter(body for (body,) in results)

        counts = uvloop.run(burst())
        assert sum(counts.values()) == connections
        # Fair share 100. Spread: busiest 100-111 in 6 runs; without spreading
        # the first worker to wake takes about half of the burst (203-215).
        assert max(counts.values()) <= 3 * connections // (2 * workers), counts
    finally:
        stop.set()
        for p in procs:
            p.join(30)
            if p.is_alive():
                p.kill()
                p.join()
        sock.close()
