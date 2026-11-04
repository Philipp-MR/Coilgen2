"""Team workspaces keep their identity and verified HTTP/WS boundaries across logins."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("aiohttp")

from aiohttp import ClientSession, CookieJar, WSMsgType, web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from yarl import URL  # noqa: E402

from coilgen.web.__main__ import main  # noqa: E402
from coilgen.web.hub import Hub, UserProcess  # noqa: E402
from coilgen.web.workspaces import WorkspaceAccounts, exclusive_host  # noqa: E402


PRIVATE = "private example passphrase"
OTHER = "distinct private passphrase"


def test_fresh_and_restarted_worker_pin_late_loaded_native_libraries(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "20")
    monkeypatch.setenv("COILGEN_WEB_PASSWORD", "private-team-password")
    spawned = []

    def launch(*args, **options):
        process = Mock()
        process.poll.return_value = None
        spawned.append((args[0], options["env"]))
        return process

    async def ready(*args):
        return None, Mock()

    monkeypatch.setattr("coilgen.web.hub.subprocess.Popen", launch)
    monkeypatch.setattr("coilgen.web.hub.free_port", lambda: 12345)
    monkeypatch.setattr("coilgen.web.hub.asyncio.open_connection", ready)
    hub = Hub(tmp_path, password="team", threads_per_worker=2)

    async def scenario():
        for _ in range(2):
            await hub.ensure_user("owner")
            hub.stop_user("owner")

    asyncio.run(scenario())
    assert len(spawned) == 2
    for args, env in spawned:
        assert args[args.index("--threads") + 1] == "2"
        for key in (
            "OPENBLAS_NUM_THREADS",
            "GOTO_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            assert env[key] == "2"
        assert "COILGEN_WEB_PASSWORD" not in env
    assert os.environ["OPENBLAS_NUM_THREADS"] == "20"


def test_disconnected_idle_worker_survives_until_the_last_job_is_terminal(tmp_path):
    hub = Hub(tmp_path, password="team", idle_minutes=1)
    owner = tmp_path / "users" / "owner"
    first = owner / "jobs" / "first" / "job.json"
    last = owner / "jobs" / "last" / "job.json"
    first.parent.mkdir(parents=True)
    last.parent.mkdir(parents=True)
    first.write_text(json.dumps({"status": "completed"}))
    worker = Mock()
    worker.poll.return_value = None
    hub.users["owner"] = UserProcess("owner", 1, worker, owner, last_seen=float("-inf"))
    sid = hub.new_session("owner")

    async def scenario():
        for status in ("queued", "running", "cancelling"):
            last.write_text(json.dumps({"status": status}))
            await hub.reap_idle()
            assert hub.sessions[sid] == "owner"
            worker.terminate.assert_not_called()
        last.write_text("{")
        await hub.reap_idle()
        worker.terminate.assert_not_called()
        last.write_text(json.dumps({"status": "cancelled"}))
        await hub.reap_idle()
        worker.terminate.assert_called_once()
        assert sid not in hub.sessions and "owner" not in hub.users

    asyncio.run(scenario())


def test_existing_workspace_requires_host_recovery_and_survives_reset(tmp_path, monkeypatch):
    original = tmp_path / "users" / "Anna-M"
    original.mkdir(parents=True)
    (original / "existing.txt").write_text("preserve", encoding="utf-8")
    accounts = WorkspaceAccounts(tmp_path)
    with pytest.raises(ValueError, match="occupied"):
        accounts.set_passphrase("anna-m", PRIVATE)
    with pytest.raises(ValueError, match="incorrect"):
        accounts.authenticate("Anna-M", PRIVATE)
    with exclusive_host(tmp_path), pytest.raises(ValueError, match="Stop the hub"):
        with exclusive_host(tmp_path):
            pass
    for phrase in (PRIVATE, OTHER):
        monkeypatch.setattr("getpass.getpass", lambda prompt: phrase)
        assert main(["--data-root", str(tmp_path), "--reset-workspace-password", "anna-m"]) == 0
        assert WorkspaceAccounts(tmp_path).authenticate("ANNA-M", phrase) == "Anna-M"
    with pytest.raises(ValueError, match="incorrect"):
        accounts.authenticate("Anna-M", PRIVATE)
    assert (original / "existing.txt").read_text() == "preserve"
    assert PRIVATE.encode() not in accounts.database.read_bytes()
    assert OTHER.encode() not in accounts.database.read_bytes()


def test_concurrent_enrollment_does_not_replace_first_credential(tmp_path):
    accounts = WorkspaceAccounts(tmp_path)

    def enroll(name, phrase):
        try:
            return accounts.set_passphrase(name, phrase)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(enroll, ("Anna-M", "anna-m"), (PRIVATE, OTHER)))
    assert results.count(None) == 1
    for result, phrase in zip(results, (PRIVATE, OTHER)):
        if result:
            assert accounts.authenticate("ANNA-M", phrase) == result
        else:
            with pytest.raises(ValueError, match="incorrect"):
                accounts.authenticate("Anna-M", phrase)
    for invalid in ("Anna M", "../Anna", "CON", "local", "A" * 41):
        with pytest.raises(ValueError):
            accounts.set_passphrase(invalid, PRIVATE)
    with pytest.raises(ValueError, match="12–256"):
        accounts.set_passphrase("short", "short")
    assert len(list(accounts.users.iterdir())) == 1


def test_logout_revokes_socket_and_compressed_download_retains_exact_bytes(tmp_path):
    async def scenario():
        upstream = web.Application()
        plain = b"synthetic private workbench result" * 12
        compressed = gzip.compress(plain)
        seen = []

        async def download(request):
            seen.append((request.headers.get("cookie"), request.headers.get("authorization")))
            return web.Response(
                body=compressed,
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Type": "application/zip",
                    "Set-Cookie": "upstream=bad",
                    "Connection": "x-private-hop",
                    "X-Private-Hop": "ignore",
                },
            )

        async def echo(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    await ws.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await ws.send_bytes(message.data)
            return ws

        upstream.router.add_get("/data", download)
        upstream.router.add_get("/downloads/Δ field #1.zip", download)
        upstream.router.add_get("/downloads/Gy.coilgen.json", download)
        upstream.router.add_get("/downloads/Gy.coilgen", download)
        upstream.router.add_get("/ws", echo)
        async with TestServer(upstream) as worker:
            hub = Hub(tmp_path, password="shared")

            async def start(name):
                return SimpleNamespace(port=worker.port, sockets=0, last_seen=0)

            hub.ensure_user = start
            async with TestServer(hub.application()) as server:
                async with (
                    ClientSession(cookie_jar=CookieJar(unsafe=True), auto_decompress=False) as a,
                    ClientSession(cookie_jar=CookieJar(unsafe=True)) as b,
                ):
                    login = server.make_url("/login")
                    creds = {"name": "Anna", "password": "shared", "passphrase": PRIVATE}
                    enrolled = await a.post(login, data={**creds, "action": "create"}, allow_redirects=False)
                    own = enrolled.headers["Location"]
                    assert own.startswith("/u/") and "HttpOnly" in enrolled.headers["Set-Cookie"]
                    for action in ("open", "create"):
                        rejected = await b.post(
                            login,
                            data={**creds, "name": "ANNA", "action": action, "passphrase": OTHER},
                            allow_redirects=False,
                        )
                        assert rejected.headers["Location"].startswith("/?error=")
                    stolen = await b.get(server.make_url(own + "data"), allow_redirects=False)
                    assert stolen.headers["Location"] == "/"
                    response = await a.get(server.make_url(own + "data"), headers={"Authorization": "private-token"})
                    assert response.status == 200 and await response.read() == compressed
                    assert response.headers["Referrer-Policy"] == "same-origin"
                    assert response.headers["Content-Encoding"] == "gzip" and "Set-Cookie" not in response.headers
                    assert "Content-Disposition" not in response.headers
                    assert (
                        response.headers["Content-Type"] == "application/zip"
                        and "X-Private-Hop" not in response.headers
                    )
                    assert seen == [(None, None)]
                    uri = URL(str(server.make_url(own)) + "downloads/%CE%94%20field%20%231.zip", encoded=True)
                    head = await a.head(uri)
                    assert head.status == 200 and head.content_length == len(compressed) and await head.read() == b""
                    assert head.headers["Content-Disposition"] == (
                        "attachment; filename=\"_ field _1.zip\"; filename*=UTF-8''%CE%94%20field%20%231.zip"
                    )
                    project = await a.get(server.make_url(own + "downloads/Gy.coilgen.json"))
                    assert await project.read() == compressed
                    assert project.headers["Content-Disposition"] == (
                        "attachment; filename=\"Gy.coilgen.json\"; filename*=UTF-8''Gy.coilgen.json"
                    )
                    compact = await a.get(server.make_url(own + "downloads/Gy.coilgen"))
                    assert compact.headers["Content-Disposition"] == (
                        "attachment; filename=\"Gy.coilgen\"; filename*=UTF-8''Gy.coilgen"
                    )
                    missing = await a.get(server.make_url(own + "downloads/missing.zip"))
                    assert missing.status == 404 and "Content-Disposition" not in missing.headers
                    target = server.make_url(own + "ws")
                    foreign = await a.get(target, headers={"Upgrade": "websocket", "Origin": "http://wrong.example"})
                    assert foreign.status == 403
                    async with a.ws_connect(target, origin=str(server.make_url("/").origin())) as ws:
                        await ws.send_str("hello")
                        assert (await ws.receive()).data == "hello"
                        await ws.send_bytes(b"mesh")
                        assert (await ws.receive()).data == b"mesh"
                        foreign_logout = await a.post(server.make_url("/logout"), headers={"Origin": "null"})
                        assert foreign_logout.status == 403
                        signed_out = asyncio.create_task(
                            a.post(
                                server.make_url("/logout"),
                                allow_redirects=False,
                                headers={"Origin": str(server.make_url("/").origin())},
                            )
                        )
                        assert (await asyncio.wait_for(ws.receive(), timeout=5)).type == WSMsgType.CLOSE
                        assert (await signed_out).status == 302
                    assert (await a.get(server.make_url(own + "data"), allow_redirects=False)).headers[
                        "Location"
                    ] == "/"
            restored = Hub(tmp_path, password="shared")
            restored.ensure_user = start
            async with TestClient(TestServer(restored.application())) as client:
                previous = own.split("/")[2]
                denied = await client.get(
                    own + "data",
                    headers={"Cookie": f"coilgen_session_{client.server.port}={previous}"},
                    allow_redirects=False,
                )
                assert denied.headers["Location"] == "/"
                login = await client.post("/login", data={**creds, "name": "ANNA"}, allow_redirects=False)
                assert login.headers["Location"].startswith("/u/") and login.headers["Location"] != own

    asyncio.run(scenario())


def test_local_listener_rejects_team_login_and_non_loopback_binding(tmp_path):
    async def scenario():
        hub = Hub(tmp_path, password=None, local_user="local")
        async with TestClient(TestServer(hub.application())) as client:
            assert (await client.post("/login", data={"name": "someone_else"})).status == 403
            assert (await client.get("/", headers={"Host": "outside.example"})).status == 403
            assert not hub.sessions

    asyncio.run(scenario())
    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0", "--no-browser", "--data-root", str(tmp_path)])
