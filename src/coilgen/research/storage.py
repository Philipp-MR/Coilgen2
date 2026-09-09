from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np

from ..persistence import (
    atomic_json as atomic_json,
    file_digest as file_digest,
    json_value as json_value,
    source_digest,
)


def operator_fingerprint(value: Any) -> Any:
    """Small provenance description without copying matrices into JSON."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays cannot identify numerical operators.")
        return {
            "shape": list(value.shape),
            "dtype": array.dtype.str,
            "sha256": sha256(memoryview(array).cast("B")).hexdigest(),
        }
    if is_dataclass(value):
        return {f.name: operator_fingerprint(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): operator_fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [operator_fingerprint(v) for v in value]
    return json_value(value)


def environment_manifest() -> dict:
    packages = {}
    for name in (
        "coilgen-python",
        "numpy",
        "scipy",
        "cvxpy",
        "clarabel",
        "scs",
        "ecos",
        "numba",
        "llvmlite",
        "threadpoolctl",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def request_identity(request: dict, resolved: dict) -> tuple[str, dict]:
    package_root = Path(__file__).resolve().parents[1]
    inputs = {}
    for key, value in resolved.items():
        if isinstance(value, str) and any(token in key for token in ("file", "path")):
            path = Path(value).expanduser()
            if path.is_file():
                inputs[key] = {"path": str(path.resolve()), "sha256": file_digest(path)}
    manifest = {
        "schema": "coilgen-research-request-v1",
        "request": request,
        "resolved_config": resolved,
        "source_sha256": source_digest(package_root),
        "input_files": inputs,
        "environment": environment_manifest(),
    }
    # A display name does not change a numerical design.
    # The resolved native configuration already captures numerical controls.
    # Relative/absolute spelling and a display name must not create mismatches
    # when a request crosses a process boundary.
    identity = dict(manifest, request={k: v for k, v in request.items() if k not in {"name", "config"}})
    encoded = json.dumps(
        json_value(identity, nonfinite="raise"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256(encoded).hexdigest(), manifest


def verified_result(directory: Path, digest: str) -> dict | None:
    """Reuse only a complete result whose request and every artifact still match."""
    try:
        value = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if value.get("request_sha256") != digest or not value.get("artifacts"):
            return None
        for item in value["artifacts"]:
            path = (directory / item["name"]).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or not item.get("sha256")
                or file_digest(path) != item["sha256"]
            ):
                return None
        return value
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return None


def verify_input_files(manifest: dict) -> None:
    """Do not certify a result if its external inputs changed during execution."""
    for key, item in manifest.get("input_files", {}).items():
        path = Path(item["path"])
        if not path.is_file() or file_digest(path) != item["sha256"]:
            raise RuntimeError(f"Input {key} changed during execution: {path}. Rerun with stable input files.")


class DirectoryOwner:
    """Exclusive OS lock, acquired before inspecting or recovering durable jobs.

    The lock file remains in place: deleting it would permit two owners to lock
    different inodes. The OS releases the lock after a crashed owner exits.
    """

    def __init__(self, directory: str | Path, *, filename: str = ".owner.lock"):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / filename
        self._file = self.path.open("a+b")
        try:
            if self.path.stat().st_size == 0:
                self._file.write(b"\0")
                self._file.flush()
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            self._file = None
            raise RuntimeError(f"Another runtime already owns {directory}.") from exc

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
