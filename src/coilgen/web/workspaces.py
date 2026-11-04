"""Host-owned credential registry for the team workbench."""

from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import hmac
from pathlib import Path, PureWindowsPath
import re
import secrets
import sqlite3


def safe_user_name(value: str) -> str:
    """Validate rather than rewrite identities; filesystem case aliases share one owner."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value):
        raise ValueError("Use 1–40 letters A–Z, digits, hyphens or underscores for the workspace name.")
    if value.lower() == "local" or PureWindowsPath(value).is_reserved():
        raise ValueError("That workspace name is reserved.")
    return value


@contextmanager
def exclusive_host(data_root: Path):
    """Only a stopped hub may be recovered; SQLite releases the lock even after a crash."""
    data_root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(data_root / ".hub-lock.sqlite3", timeout=0)) as lock:
        try:
            lock.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower():
                raise ValueError("Stop the hub using this data root before continuing.") from exc
            raise
        yield


def _digest(passphrase: str, salt: bytes) -> bytes:
    return hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)


class WorkspaceAccounts:
    def __init__(self, data_root: Path):
        self.users = data_root / "users"
        self.users.mkdir(parents=True, exist_ok=True)
        self.database = data_root / ".workspace-accounts.sqlite3"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS accounts "
                "(identity TEXT PRIMARY KEY, directory TEXT NOT NULL, salt BLOB NOT NULL, verifier BLOB NOT NULL)"
            )

    def _existing_directory(self, name: str) -> Path | None:
        matches = [path for path in self.users.iterdir() if path.name.lower() == name.lower()]
        if not matches:
            return None
        path = matches[0]
        if len(matches) != 1 or not path.is_dir() or path.resolve() != self.users.resolve() / path.name:
            raise ValueError("Conflicting workspace paths require inspection by the host operator.")
        return path

    def set_passphrase(self, name: str, passphrase: str, *, recover: bool = False) -> str:
        name = safe_user_name(name)
        if not isinstance(passphrase, str) or not 12 <= len(passphrase) <= 256:
            raise ValueError("Choose a private workspace passphrase of 12–256 characters.")
        salt = secrets.token_bytes(16)
        verifier = _digest(passphrase, salt)
        identity = name.lower()
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            account = db.execute("SELECT directory FROM accounts WHERE identity=?", (identity,)).fetchone()
            existing = self._existing_directory(name)
            if recover:
                if existing is None:
                    raise ValueError("Recovery requires an existing workspace directory.")
                directory = existing.name
            else:
                if existing is not None or account is not None:
                    raise ValueError(
                        "Workspace name is occupied. Open it with its private passphrase or use another name."
                    )
                directory = name
            db.execute(
                "INSERT INTO accounts(identity, directory, salt, verifier) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(identity) DO UPDATE SET directory=excluded.directory, salt=excluded.salt, "
                "verifier=excluded.verifier",
                (identity, directory, salt, verifier),
            )
            if not recover:
                # Never adopt a pre-existing directory, even if the registry was lost.
                (self.users / directory).mkdir()
        return directory

    def authenticate(self, name: str, passphrase: str) -> str:
        name = safe_user_name(name)
        with closing(sqlite3.connect(self.database)) as db:
            account = db.execute(
                "SELECT directory, salt, verifier FROM accounts WHERE identity=?", (name.lower(),)
            ).fetchone()
        if (
            account is None
            or not isinstance(passphrase, str)
            or not 12 <= len(passphrase) <= 256
            or not hmac.compare_digest(_digest(passphrase, account[1]), account[2])
        ):
            raise ValueError("Workspace name or private passphrase is incorrect.")
        if self._existing_directory(name) != self.users / account[0]:
            raise ValueError("Workspace files require inspection by the host operator.")
        return account[0]
