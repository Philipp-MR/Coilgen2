"""Transient Windows replace denial never exposes a partial scientific text file."""

import errno
from pathlib import Path
import time

import pytest

from coilgen import persistence


def _denied(path, winerror):
    error = PermissionError(errno.EACCES, "Fixture replace denied", str(path))
    error.winerror = winerror
    return error


@pytest.mark.parametrize("winerror", (5, 32, 33))
def test_transient_windows_denial_replaces_only_complete_exact_utf8(tmp_path, monkeypatch, winerror):
    path = tmp_path / "job.json"
    previous = b'{"status":"running"}\n'
    path.write_bytes(previous)
    text = '{"status":"completed","label":"Gradient γ — µT/m"}\n'
    original, attempts = persistence.os.replace, []

    def replace(source, target):
        attempts.append((Path(source), Path(target)))
        assert Path(source).read_bytes() == text.encode("utf-8")
        assert Path(target).read_bytes() == previous
        if len(attempts) <= 3:
            raise _denied(target, winerror)
        return original(source, target)

    monkeypatch.setattr(persistence.os, "replace", replace)
    with persistence.atomic_text_writer(path) as output:
        output.write(text[:25])
        output.write(text[25:])
    assert len(attempts) == 4
    assert path.read_bytes() == text.encode("utf-8")
    assert list(tmp_path.iterdir()) == [path]


def test_permanent_windows_denial_is_bounded_preserves_old_file_and_raises(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    previous = b'{"quality":"FAIL","current_A":2}\n'
    path.write_bytes(previous)
    modified, attempts = path.stat().st_mtime_ns, []

    def replace(source, target):
        attempts.append(Path(source))
        raise _denied(target, 5)

    monkeypatch.setattr(persistence.os, "replace", replace)
    start = time.monotonic()
    with pytest.raises(PermissionError, match="Fixture replace denied"):
        persistence.atomic_text(path, "a complete replacement\n")
    assert 1 < len(attempts) <= 10 and time.monotonic() - start < 2
    assert path.read_bytes() == previous and path.stat().st_mtime_ns == modified
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("winerror", (None, 112))
def test_unrelated_permission_error_is_not_retried_or_hidden(tmp_path, monkeypatch, winerror):
    path = tmp_path / "result.json"
    path.write_text("old evidence", encoding="utf-8")
    attempts = []

    def replace(_source, target):
        attempts.append(target)
        raise _denied(target, winerror)

    monkeypatch.setattr(persistence.os, "replace", replace)
    with pytest.raises(PermissionError):
        persistence.atomic_text(path, "new evidence")
    assert attempts == [path]
    assert path.read_text(encoding="utf-8") == "old evidence"
    assert list(tmp_path.iterdir()) == [path]
