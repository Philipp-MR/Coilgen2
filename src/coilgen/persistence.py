"""Shared source fingerprints and durable text persistence, without solver dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator, TextIO
import uuid


def json_value(value: Any, *, nonfinite: str = "null") -> Any:
    """Convert numerical results to strict JSON; use ``raise`` for input identity.

    Missing results become null by default. Nonfinite design inputs must instead
    raise so that they cannot acquire the same fingerprint as an omitted setting.
    """
    if nonfinite not in {"null", "raise"}:
        raise ValueError("nonfinite must be 'null' or 'raise'.")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if nonfinite == "raise":
            raise ValueError("Nonfinite values cannot identify reproducible design inputs.")
        return None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_value(asdict(value), nonfinite=nonfinite)
    if isinstance(value, Mapping):
        return {str(k): json_value(v, nonfinite=nonfinite) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v, nonfinite=nonfinite) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist(), nonfinite=nonfinite)
    raise TypeError(f"Cannot save {type(value).__name__} as reproducible JSON.")


@contextmanager
def atomic_text_writer(path: str | Path, *, newline: str = "\n") -> Iterator[TextIO]:
    """Replace a complete UTF-8 file only after writing and flushing succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline=newline) as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        # Windows readers and background scanners can briefly deny replacement.
        # Retry only its sharing/access errors; retain atomic replacement and
        # propagate a persistent denial without touching the previous file.
        delays = iter((0.01, 0.02, 0.04, 0.08, 0.16))
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                delay = next(delays, None)
                if getattr(exc, "winerror", None) not in {5, 32, 33} or delay is None:
                    raise
                time.sleep(delay)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: str | Path, text: str) -> None:
    with atomic_text_writer(path) as output:
        output.write(text)


def atomic_json(path: Path, value: Any) -> None:
    """Persist result JSON atomically, representing unavailable values as null."""
    text = json.dumps(json_value(value), allow_nan=False, indent=2) + "\n"
    atomic_text(path, text)


def file_digest(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_digest(package_root: Path, *, exclude: tuple[str, ...] = ()) -> str:
    """Hash relative names and contents of all Python modules, including subpackages.

    The digest does not depend on checkout location, timestamps or generated
    bytecode. Length-delimited names and fixed-size content hashes avoid ambiguous
    concatenations when files are renamed, added or removed.
    """
    package_root = Path(package_root)
    if not package_root.is_dir():
        raise NotADirectoryError(package_root)
    digest = sha256()
    for path in sorted(package_root.rglob("*.py"), key=lambda p: p.relative_to(package_root).as_posix()):
        if path.is_file():
            relative = path.relative_to(package_root).as_posix()
            if any(relative == entry or relative.startswith(entry.rstrip("/") + "/") for entry in exclude):
                continue
            name = relative.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(bytes.fromhex(file_digest(path)))
    return digest.hexdigest()


def numerical_source_digest(package_root: Path) -> str:
    """Conservative engine identity independent of presentation and adapters.

    New numerical modules remain included automatically. The small exclusion
    list contains only software credits, entry points, GUI/project editing, plotting and job
    adapters; the numerical pipeline, config, kernels and validation stay hashed.
    Full-source provenance is still retained separately in run manifests.
    """
    return source_digest(
        package_root,
        exclude=(
            "gui",
            "application",
            "cli.py",
            "__main__.py",
            "__init__.py",
            "attribution.py",
            "visualization.py",
            "matplotlib_visualization.py",
            "plot_book.py",
            "swept_mesh_inspector.py",
            "gradient_slice_inspector.py",
            "project.py",
            "research/application.py",
            "research/jobs.py",
            "research/requests.py",
            "research/storage.py",
            "research/studies.py",
            "research/presets.py",
        ),
    )
