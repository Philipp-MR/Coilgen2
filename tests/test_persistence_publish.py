"""Exclusive CAD publication tolerates Windows locks without overwriting competing files."""

import os
from types import SimpleNamespace

import pytest

from coilgen import persistence


def windows_lock(code):
    error = PermissionError("file temporarily locked")
    error.winerror = code
    return error


def exclusive_rename(source, target):
    if os.name == "nt":
        os.rename(source, target)
    else:
        os.link(source, target)
        source.unlink()


@pytest.mark.parametrize("code", [5, 32, 33])
def test_exclusive_publication_retries_transient_windows_locks(tmp_path, monkeypatch, code):
    source, target = tmp_path / "staged.stl", tmp_path / "published.stl"
    source.write_bytes(b"exact conductor mesh")
    calls, sleeps = [], []

    def rename(first, second):
        calls.append((first, second))
        if len(calls) == 1:
            raise windows_lock(code)
        # Model Windows' exclusive rename on both CI platforms.
        exclusive_rename(first, second)

    monkeypatch.setattr(persistence, "os", SimpleNamespace(name="nt", rename=rename))
    monkeypatch.setattr(persistence.time, "sleep", sleeps.append)
    persistence.publish_exclusive(source, target)
    assert target.read_bytes() == b"exact conductor mesh" and not source.exists()
    assert len(calls) == 2 and sleeps == [0.01]


def test_exclusive_publication_bounds_persistent_lock_retries(tmp_path, monkeypatch):
    source, target = tmp_path / "staged.stl", tmp_path / "published.stl"
    source.write_bytes(b"retained staging geometry")
    calls, sleeps = [], []
    failure = windows_lock(32)

    def rename(*args):
        calls.append(args)
        raise failure

    monkeypatch.setattr(persistence, "os", SimpleNamespace(name="nt", rename=rename))
    monkeypatch.setattr(persistence.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        persistence.publish_exclusive(source, target)
    assert caught.value is failure
    assert len(calls) == 6 and sum(sleeps) == pytest.approx(0.31)
    assert source.read_bytes() == b"retained staging geometry" and not target.exists()


def test_publication_retry_cannot_replace_a_competing_writer(tmp_path, monkeypatch):
    source, target = tmp_path / "staged.stl", tmp_path / "published.stl"
    source.write_bytes(b"staged geometry")
    calls = []

    def rename(first, second):
        calls.append((first, second))
        if len(calls) == 1:
            second.write_bytes(b"competing geometry")
            raise windows_lock(32)
        exclusive_rename(first, second)

    monkeypatch.setattr(persistence, "os", SimpleNamespace(name="nt", rename=rename))
    monkeypatch.setattr(persistence.time, "sleep", lambda _: None)
    with pytest.raises(FileExistsError):
        persistence.publish_exclusive(source, target)
    assert len(calls) == 2
    assert source.read_bytes() == b"staged geometry" and target.read_bytes() == b"competing geometry"


def test_publication_does_not_retry_unclassified_permission_errors(tmp_path, monkeypatch):
    calls, sleeps = [], []

    def rename(*args):
        calls.append(args)
        raise PermissionError("permanent permission failure")

    monkeypatch.setattr(persistence, "os", SimpleNamespace(name="nt", rename=rename))
    monkeypatch.setattr(persistence.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError, match="permanent"):
        persistence.publish_exclusive(tmp_path / "source", tmp_path / "target")
    assert len(calls) == 1 and sleeps == []
