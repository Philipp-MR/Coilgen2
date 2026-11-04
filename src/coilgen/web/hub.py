"""Multi-user hub: login, one isolated workbench process per user, one port.

Each user gets ``<data_root>/users/<name>/`` (projects, jobs, uploads,
exports) and a private ``coilgen.web.app`` process on a loopback port. The hub
proxies HTTP and WebSocket traffic for ``/u/<session>/`` to that process, so
only the hub port needs to be reachable. Sessions are bound to an HttpOnly
cookie; idle processes are stopped. For access beyond a trusted local network,
put the hub behind an HTTPS reverse proxy.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hmac
import html
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from urllib.parse import quote, urlsplit

from aiohttp import ClientSession, ClientTimeout, DummyCookieJar, WSMsgType, web
from yarl import URL

from .workspaces import WorkspaceAccounts, exclusive_host, safe_user_name

COOKIE = "coilgen_session"
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def _cookie_name(request) -> str:
    # HTTP cookies are shared across ports on the same host. Keep independent hubs separate.
    return f"{COOKIE}_{request.url.port}"


def _check_origin(request):
    origin = request.headers.get("Origin")
    if origin is not None:
        try:
            parsed = urlsplit(origin)
        except ValueError as exc:
            raise web.HTTPForbidden(text="Invalid request origin.") from exc
        if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != request.host.lower():
            raise web.HTTPForbidden(text="Use the workbench on its own origin.")


def _connection_headers(headers):
    return HOP_HEADERS | {
        name.strip().lower() for value in headers.getall("Connection", ()) for name in value.split(",")
    }


def _download_disposition(path: str) -> str | None:
    if not path.startswith("downloads/"):
        return None
    name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not name.lower().endswith((".json", ".coilgen", ".zip")):
        return None
    name = "".join("_" if ord(ch) < 32 or ord(ch) == 127 else ch for ch in name)
    # Supply an ASCII fallback and the actual UTF-8 filename for HTTP clients per RFC 6266.
    fallback = "".join(ch if ch.isascii() and (ch.isalnum() or ch in " ._-") else "_" for ch in name)
    encoded = quote(name, safe="")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@dataclass
class UserProcess:
    user: str
    port: int
    process: subprocess.Popen
    workspace: Path
    last_seen: float = field(default_factory=time.monotonic)
    sockets: int = 0

    def alive(self) -> bool:
        return self.process.poll() is None

    def has_active_jobs(self) -> bool:
        from ..application.session import ACTIVE_STATUSES

        try:
            for path in (self.workspace / "jobs").glob("*/job.json"):
                with path.open(encoding="utf-8") as stream:
                    state = json.load(stream)
                status = state.get("status") if isinstance(state, dict) else None
                if not isinstance(status, str) or not status or status in ACTIVE_STATUSES:
                    return True
        except (OSError, ValueError):
            # Retry on the next pass if durable state cannot yet be read.
            return True
        return False


class Hub:
    def __init__(
        self,
        data_root: Path,
        *,
        password: str | None,
        max_users: int = 8,
        threads_per_worker: int | None = None,
        idle_minutes: float = 120.0,
        local_user: str | None = None,
        python: str | None = None,
        app_module: str = "coilgen.web.app",
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        (self.data_root / "users").mkdir(parents=True, exist_ok=True)
        if local_user is None and not password:
            raise ValueError("Team mode requires a team password.")
        self.accounts = WorkspaceAccounts(self.data_root) if local_user is None else None
        self.password = password
        self.max_users = max(1, int(max_users))
        self.idle_seconds = float(idle_minutes) * 60.0
        self.local_user = local_user
        self.python = python or sys.executable
        self.app_module = app_module
        self.users: dict[str, UserProcess] = {}
        self._starting: dict[str, asyncio.Lock] = {}
        self.sessions: dict[str, str] = {}  # session id -> user name
        self._sockets: dict[str, set[web.WebSocketResponse]] = {}
        self.client: ClientSession | None = None
        cores = os.cpu_count() or 1
        if threads_per_worker is not None and threads_per_worker < 1:
            raise ValueError("Threads per worker must be at least one.")
        self.threads = threads_per_worker if threads_per_worker is not None else max(1, cores // self.max_users)

    # -- processes -----------------------------------------------------
    async def ensure_user(self, user: str) -> UserProcess:
        async with self._starting.setdefault(user, asyncio.Lock()):
            return await self._start_user(user)

    async def _start_user(self, user: str) -> UserProcess:
        current = self.users.get(user)
        if current is not None and current.alive():
            return current
        active = [item for item in self.users.values() if item.alive()]
        if len(active) >= self.max_users:
            raise web.HTTPServiceUnavailable(text="All workbench seats are in use; try again later.")
        workspace = self.data_root / "users" / user
        workspace.mkdir(parents=True, exist_ok=True)
        port = free_port()
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}
        environment.pop("COILGEN_WEB_PASSWORD", None)
        # Solver plugins may load another BLAS library after the engine enters threadpool_limits.
        # Set its startup ceiling before either the workbench or its numerical children import it.
        for setting in (
            "OPENBLAS_NUM_THREADS",
            "GOTO_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            environment[setting] = str(self.threads)
        with (workspace / "workbench.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    self.python,
                    "-m",
                    self.app_module,
                    "--workspace",
                    str(workspace),
                    "--user",
                    user,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--threads",
                    str(self.threads),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        entry = UserProcess(user, port, process, workspace)
        self.users[user] = entry
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            if not entry.alive():
                raise web.HTTPInternalServerError(
                    text=f"Workbench for {user} stopped; see {workspace / 'workbench.log'}."
                )
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                return entry
            except OSError:
                await asyncio.sleep(0.25)
        process.terminate()
        raise web.HTTPGatewayTimeout(text="The workbench did not start in time.")

    def stop_user(self, user: str):
        entry = self.users.pop(user, None)
        if entry is not None and entry.alive():
            entry.process.terminate()
            try:
                entry.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                entry.process.kill()
        for sid in [sid for sid, name in self.sessions.items() if name == user]:
            self.sessions.pop(sid, None)

    async def reap(self):
        while True:
            await asyncio.sleep(60)
            await self.reap_idle()

    async def reap_idle(self):
        for user, entry in list(self.users.items()):
            if not entry.alive():
                self.stop_user(user)
                continue
            if entry.sockets or time.monotonic() - entry.last_seen <= self.idle_seconds:
                continue
            if await asyncio.to_thread(entry.has_active_jobs):
                continue
            # A user may have returned while the worker's durable queue was being inspected.
            if (
                self.users.get(user) is entry
                and not entry.sockets
                and time.monotonic() - entry.last_seen > self.idle_seconds
            ):
                self.stop_user(user)

    # -- sessions ------------------------------------------------------
    def new_session(self, user: str) -> str:
        sid = secrets.token_urlsafe(18)
        self.sessions[sid] = user
        return sid

    def session_user(self, request, sid: str) -> str | None:
        user = self.sessions.get(sid)
        if user is None or request.cookies.get(_cookie_name(request)) != sid:
            return None
        return user

    async def _revoke_session(self, sid: str):
        self.sessions.pop(sid, None)
        for socket_ in tuple(self._sockets.pop(sid, ())):
            await socket_.close()

    def _redirect_to_session(self, request, sid: str) -> web.Response:
        response = web.HTTPFound(f"/u/{sid}/")
        response.set_cookie(
            _cookie_name(request), sid, httponly=True, samesite="Strict", secure=request.secure, path="/"
        )
        return response

    # -- handlers ------------------------------------------------------
    async def index(self, request):
        sid = request.cookies.get(_cookie_name(request), "")
        if sid in self.sessions and self.sessions[sid] in self.users and self.users[self.sessions[sid]].alive():
            raise web.HTTPFound(f"/u/{sid}/")
        if self.local_user is not None:
            await self.ensure_user(self.local_user)
            raise self._redirect_to_session(request, self.new_session(self.local_user))
        return web.Response(text=_login_page(error=request.query.get("error", "")), content_type="text/html")

    async def login(self, request):
        _check_origin(request)
        if self.accounts is None:
            raise web.HTTPForbidden(text="The local workbench only opens its own workspace.")
        form = await request.post()
        shared = str(form.get("password", "")).encode("utf-8")
        if not hmac.compare_digest(shared, self.password.encode("utf-8")):
            await asyncio.sleep(1.0)
            raise web.HTTPFound("/?error=" + _quote("Incorrect team password."))
        try:
            name = safe_user_name(form.get("name", ""))
            private = form.get("passphrase", "")
            action = form.get("action", "open")
            if private == self.password:
                raise ValueError("The private workspace passphrase must differ from the shared team password.")
            if action == "create":
                user = await asyncio.to_thread(self.accounts.set_passphrase, name, private)
            elif action == "open":
                user = await asyncio.to_thread(self.accounts.authenticate, name, private)
            else:
                raise ValueError("Choose Open existing workspace or Create workspace.")
        except ValueError as exc:
            raise web.HTTPFound("/?error=" + _quote(str(exc)))
        try:
            await self.ensure_user(user)
        except web.HTTPServiceUnavailable as exc:
            if action == "create":
                raise web.HTTPFound(
                    "/?error="
                    + _quote(
                        "Workspace created. All seats are in use. Choose Open existing workspace when a seat is free."
                    )
                ) from exc
            raise
        await self._revoke_session(request.cookies.get(_cookie_name(request), ""))
        raise self._redirect_to_session(request, self.new_session(user))

    async def logout(self, request):
        if request.method == "GET":
            return web.Response(
                text='<form method="post" action="/logout"><button>Confirm logout</button></form>',
                content_type="text/html",
            )
        _check_origin(request)
        await self._revoke_session(request.cookies.get(_cookie_name(request), ""))
        response = web.HTTPFound("/")
        response.del_cookie(_cookie_name(request), path="/")
        raise response

    async def upload_project(self, request):
        from .input_policy import PROJECT_UPLOAD_LIMIT, stage_project_upload

        user = self.session_user(request, request.match_info["sid"])
        if user is None:
            return web.json_response({"error": "Open your workspace before uploading a project."}, status=403)
        try:
            _check_origin(request)
            if request.content_type != "multipart/form-data":
                raise ValueError("Upload the project as a multipart file.")
            if request.content_length and request.content_length > PROJECT_UPLOAD_LIMIT + 65536:
                raise OverflowError("Project exceeds the 512 MiB upload limit.")
            ticket = await stage_project_upload(self.data_root / "users" / user, await request.multipart())
            return web.json_response(ticket, headers={"Cache-Control": "no-store"})
        except web.HTTPForbidden:
            return web.json_response({"error": "Use the workbench on its own origin."}, status=403)
        except OverflowError as exc:
            return web.json_response({"error": str(exc)}, status=413)
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def proxy(self, request):
        sid = request.match_info["sid"]
        user = self.session_user(request, sid)
        if user is None:
            raise web.HTTPFound("/")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            _check_origin(request)
        entry = await self.ensure_user(user)
        entry.last_seen = time.monotonic()
        # Keep escaped filename characters and query strings exactly as sent by the browser.
        tail = request.raw_path.split("/", 3)[3]
        target = URL(f"http://127.0.0.1:{entry.port}/{tail}", encoded=True)
        if request.headers.get("upgrade", "").lower() == "websocket":
            _check_origin(request)
            return await self._proxy_websocket(request, target.with_scheme("ws"), entry, sid)
        blocked = _connection_headers(request.headers) | {"cookie", "authorization", "content-encoding"}
        headers = {key: value for key, value in request.headers.items() if key.lower() not in blocked}
        body = await request.read()
        async with self.client.request(
            request.method, target, headers=headers, data=body, allow_redirects=False
        ) as upstream:
            # The internal client keeps encoded bytes intact. Stream them with the corresponding
            # encoding and length, including the declared size on a HEAD download.
            response = web.StreamResponse(status=upstream.status)
            hidden = (_connection_headers(upstream.headers) - {"content-length"}) | {"set-cookie"}
            for key, value in upstream.headers.items():
                if key.lower() not in hidden:
                    response.headers.add(key, value)
            response.headers["Cache-Control"] = "private, no-store"
            # Chromium uses this policy to populate Origin on POST forms such as logout.
            # Keep the session-bearing URL private when following an external link.
            response.headers["Referrer-Policy"] = "same-origin"
            if request.method in ("GET", "HEAD") and upstream.status in (200, 206):
                disposition = _download_disposition(request.match_info["tail"])
                if disposition:
                    response.headers["Content-Disposition"] = disposition
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(256 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response

    async def _proxy_websocket(self, request, target, entry, sid):
        browser = web.WebSocketResponse(max_msg_size=0, autoping=True)
        await browser.prepare(request)
        if sid not in self.sessions:
            await browser.close()
            return browser
        self._sockets.setdefault(sid, set()).add(browser)
        entry.sockets += 1
        try:
            async with self.client.ws_connect(target, max_msg_size=0) as upstream:

                async def forward(source, sink):
                    async for message in source:
                        entry.last_seen = time.monotonic()
                        if message.type == WSMsgType.TEXT:
                            await sink.send_str(message.data)
                        elif message.type == WSMsgType.BINARY:
                            await sink.send_bytes(message.data)
                        elif message.type in {WSMsgType.CLOSE, WSMsgType.ERROR}:
                            break

                tasks = [
                    asyncio.ensure_future(forward(browser, upstream)),
                    asyncio.ensure_future(forward(upstream, browser)),
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._sockets.get(sid, set()).discard(browser)
            entry.sockets -= 1
            entry.last_seen = time.monotonic()
            await browser.close()
        return browser

    async def session_root(self, request):
        raise web.HTTPFound(f"/u/{request.match_info['sid']}/")

    # -- application ---------------------------------------------------
    def application(self) -> web.Application:
        @web.middleware
        async def local_host_only(request, handler):
            if self.local_user is not None and not is_loopback_host(request.url.host or ""):
                raise web.HTTPForbidden(text="Open the local workbench through its loopback address.")
            return await handler(request)

        app = web.Application(client_max_size=512 * 1024**2, middlewares=[local_host_only])
        app.router.add_get("/", self.index)
        app.router.add_post("/login", self.login)
        app.router.add_post("/logout", self.logout)
        app.router.add_get("/logout", self.logout)
        app.router.add_get("/u/{sid}", self.session_root)
        app.router.add_post("/u/{sid}/upload/project", self.upload_project)
        app.router.add_route("*", "/u/{sid}/{tail:.*}", self.proxy)

        async def lifetime(_app):
            with exclusive_host(self.data_root):
                async with ClientSession(
                    timeout=ClientTimeout(total=None, sock_connect=30),
                    auto_decompress=False,
                    cookie_jar=DummyCookieJar(),
                    skip_auto_headers={"Accept-Encoding"},
                ) as self.client:
                    reaper = asyncio.create_task(self.reap())
                    try:
                        yield
                    finally:
                        reaper.cancel()
                        await asyncio.gather(reaper, return_exceptions=True)
                        for user in list(self.users):
                            self.stop_user(user)

        app.cleanup_ctx.append(lifetime)
        return app


def _quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text)


def _login_page(error: str = "") -> str:
    message = f'<p class="error">{html.escape(error)}</p>' if error else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoilGen2 workbench</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #f4f6f8; color: #1d2733; margin: 0; }}
main {{ max-width: 380px; margin: 12vh auto; padding: 32px; background: #fff; border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,.08); }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} p {{ color: #4a5868; font-size: 14px; }}
label {{ display: block; margin-top: 16px; font-size: 13px; }}
input {{ width: 100%; box-sizing: border-box; padding: 10px; margin-top: 4px; border: 1px solid #c3ccd6; border-radius: 6px; }}
button {{ margin-top: 20px; width: 100%; padding: 11px; border: 0; border-radius: 6px; background: #002d5c; color: #fff; font-size: 15px; }}
.error {{ color: #b3261e; }}
</style></head><body><main>
<h1>CoilGen2 workbench</h1><p>Enter the team password supplied by the host and your private workspace passphrase.
New here? Choose an unused name and a private passphrase of 12–256 characters, then select Create workspace.</p>{message}
<form method="post" action="/login">
<label>Workspace name<input name="name" autocomplete="username" required maxlength="40" pattern="[A-Za-z0-9_\\-]+" aria-describedby="name-help" title="Use letters A–Z, digits, hyphens or underscores."></label>
<small id="name-help">Use 1–40 letters A–Z, digits, hyphens or underscores. Names are case-insensitive.</small>
<label>Team password<input name="password" type="password" required autocomplete="off"></label>
<label>Private workspace passphrase<input name="passphrase" type="password" required maxlength="256" autocomplete="current-password"></label>
<button type="submit" name="action" value="open">Open existing workspace</button>
<button type="submit" name="action" value="create">Create workspace</button>
</form><p>If you used this workspace with the earlier team login or lost its passphrase, ask the host operator to reset it.</p>
</main></body></html>"""
