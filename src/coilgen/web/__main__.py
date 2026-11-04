"""Start the CoilGen2 browser workbench hub.

Local (default): http://127.0.0.1:8765, no login, opens the default browser.
Team: ``--team`` listens on all interfaces, requires a password and gives every
participant a private session. Example::

    python -m coilgen.web --team --password hackathon2026
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import secrets
import socket
import sys
import webbrowser


def _lan_addresses():
    addresses = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    return sorted(address for address in addresses if not address.startswith("127."))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="CoilGen2 browser workbench (shared design core with the desktop app)."
    )
    parser.add_argument("--team", action="store_true", help="Serve several users on the local network.")
    parser.add_argument("--host", default=None, help="Listen address (default 127.0.0.1, or 0.0.0.0 with --team).")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--password", default=os.environ.get("COILGEN_WEB_PASSWORD"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("COILGEN_WEB_DATA", Path.home() / "CoilGen2 Browser Workspaces")),
        help="Where each user's projects, jobs and exports are stored.",
    )
    parser.add_argument("--max-users", type=int, default=8)
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        help="Fix native threads per study for reproducible comparisons (default: CPUs divided by seats).",
    )
    parser.add_argument("--idle-minutes", type=float, default=120.0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--reset-workspace-password",
        metavar="NAME",
        help="Enroll or recover an existing team workspace on this host. Stop the hub first.",
    )
    options = parser.parse_args(argv)
    if options.threads_per_worker is not None and options.threads_per_worker < 1:
        parser.error("--threads-per-worker must be at least 1")

    from aiohttp import web

    from .hub import Hub, is_loopback_host
    from .workspaces import WorkspaceAccounts, exclusive_host, safe_user_name

    if options.reset_workspace_password is not None:
        try:
            name = safe_user_name(options.reset_workspace_password)
            root = options.data_root.expanduser().resolve()
            with exclusive_host(root):
                accounts = WorkspaceAccounts(root)
                private = getpass.getpass("New private workspace passphrase (12–256 characters): ")
                if private != getpass.getpass("Confirm private passphrase: "):
                    raise ValueError("Passphrases do not match.")
                if private == options.password:
                    raise ValueError("Choose a private passphrase different from the shared team password.")
                directory = accounts.set_passphrase(name, private, recover=True)
        except (EOFError, ValueError) as exc:
            parser.error(str(exc) or "A private passphrase is required.")
        print(f"Private passphrase set for {directory}. Existing workspace files were preserved.")
        return 0
    host = options.host or ("0.0.0.0" if options.team else "127.0.0.1")
    if not options.team and not is_loopback_host(host):
        parser.error("Local mode must bind a loopback address. Use --team to host on the network.")
    password = options.password
    if options.team and not password:
        password = secrets.token_urlsafe(6)
    hub = Hub(
        options.data_root,
        password=password if options.team else None,
        max_users=options.max_users if options.team else 1,
        threads_per_worker=options.threads_per_worker,
        idle_minutes=options.idle_minutes,
        local_user=None if options.team else "local",
    )
    browser_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    browser_address = f"[{browser_host}]" if ":" in browser_host else browser_host
    local_url = f"http://{browser_address}:{options.port}/"
    print("CoilGen2 browser workbench")
    print(f"  Workspaces: {hub.data_root}")
    if options.team:
        addresses = [browser_host]
        if host == "0.0.0.0":
            addresses = _lan_addresses() or ["<this computer's address>"]
        for address in addresses:
            if ":" in address:
                address = f"[{address}]"
            print(f"  Team URL:   http://{address}:{options.port}/")
        print(f"  Password:   {password}")
        print(f"  Seats:      {hub.max_users} users")
    print(f"  Compute:    {hub.threads} native threads per computation")
    print(f"  Local URL:  {local_url}")
    print("Press Ctrl+C to stop.")
    sys.stdout.flush()
    if not options.no_browser:
        webbrowser.open(local_url)
    web.run_app(hub.application(), host=host, port=options.port, print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
