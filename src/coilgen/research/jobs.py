from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import uuid

from .requests import ResearchRequest
from .storage import DirectoryOwner, atomic_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    max_workers: int = 1
    threads_per_worker: int = 1
    max_pending: int = 32
    timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        for name in ("max_workers", "threads_per_worker", "max_pending"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_workers * self.threads_per_worker > (os.cpu_count() or 1):
            raise ValueError("Workers × native threads exceeds the available CPU budget.")
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite.")


def _worker(request: dict, directory: str, events, cancel, threads: int) -> None:
    """Spawn target; only JSON-compatible requests and explicit IPC cross here."""
    from threadpoolctl import threadpool_limits
    from ..progress import DesignCancelled, progress_context
    from .application import run_design

    last_progress = 0.0

    def sink(event: dict) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if event.get("label") != "CoilGen" and now - last_progress < 0.15:
            return
        last_progress = now
        try:
            events.put_nowait({"kind": "progress", **event})
        except queue.Full:
            pass

    try:
        with threadpool_limits(limits=threads), progress_context(sink, cancel.is_set):
            run_design(ResearchRequest.from_dict(request), Path(directory), threads=threads)
        events.put({"kind": "completed"}, timeout=5)
    except DesignCancelled:
        events.put({"kind": "cancelled"}, timeout=5)
    except BaseException as exc:
        (Path(directory) / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        try:
            events.put({"kind": "failed", "error": f"{type(exc).__name__}: {exc}"}, timeout=5)
        except queue.Full:
            pass


class JobManager:
    """Bounded local process jobs with durable state and cancellation.

    A completed result is atomic. Restarting the runtime marks unfinished work
    interrupted rather than guessing completion or silently reusing its files.
    Results are never selected as best solely because execution completed.
    """

    def __init__(self, directory: str | Path, policy: RuntimePolicy | None = None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.policy = policy or RuntimePolicy()
        self._owner = DirectoryOwner(self.directory)
        self._context = mp.get_context("spawn")
        self._lock = threading.RLock()
        self._states: dict[str, dict] = {}
        self._active: dict[str, tuple] = {}
        self._stop = threading.Event()
        for path in self.directory.glob("*/job.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if state.get("status") in {"queued", "running", "cancelling"}:
                    state.update(
                        status="interrupted",
                        error="The runtime stopped before this job completed.",
                        updated_at=_now(),
                    )
                    atomic_json(path, state)
                if state.get("id") == path.parent.name:
                    self._states[state["id"]] = state
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        self._thread = threading.Thread(target=self._monitor, name="CoilGen jobs", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._owner.close()
            raise

    def submit(self, value: ResearchRequest | dict) -> dict:
        request = ResearchRequest.from_dict(value.to_dict() if isinstance(value, ResearchRequest) else value)
        config = request.to_config(threads=self.policy.threads_per_worker)
        request = replace(request, config=asdict(config))
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("The runtime is shutting down.")
            pending = sum(s["status"] in {"queued", "running", "cancelling"} for s in self._states.values())
            if pending >= self.policy.max_pending:
                raise ValueError(f"The queue is full ({self.policy.max_pending} pending jobs).")
            id_ = uuid.uuid4().hex
            directory = self.directory / id_
            directory.mkdir()
            atomic_json(directory / "request.json", request.to_dict())
            state = {
                "id": id_,
                "name": request.name,
                "status": "queued",
                "created_at": _now(),
                "updated_at": _now(),
                "stage": "Waiting for worker",
                "progress": 0.0,
                "log": [],
                "error": None,
            }
            self._states[id_] = state
            self._save(state)
            return dict(state)

    def list(self) -> list[dict]:
        with self._lock:
            return [json.loads(json.dumps(s)) for s in reversed(list(self._states.values()))]

    def get(self, id_: str) -> dict:
        with self._lock:
            if id_ not in self._states:
                raise KeyError(id_)
            return json.loads(json.dumps(self._states[id_]))

    def cancel(self, id_: str) -> dict:
        with self._lock:
            state = self._states[id_]
            if state["status"] == "queued":
                state.update(status="cancelled", stage="Cancelled before execution")
            elif state["status"] == "running":
                self._active[id_][2].set()
                state.update(status="cancelling", stage="Cancelling", cancel_time=time.monotonic())
            self._save(state)
            return dict(state)

    def result(self, id_: str) -> dict:
        state = self.get(id_)
        if state["status"] != "completed":
            raise ValueError("This job has no completed result.")
        result = json.loads((self.directory / id_ / "result.json").read_text(encoding="utf-8"))
        result.update(id=id_, name=state["name"])
        return result

    def artifact(self, id_: str, name: str) -> Path:
        self.get(id_)
        root = self.directory / id_
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(name)
        # Only a request/result manifest or explicitly listed result artifact is public.
        allowed = {"request.json", "result.json", "manifest.json", "error.log"}
        result_path = root / "result.json"
        if result_path.is_file():
            allowed.update(a["name"] for a in json.loads(result_path.read_text(encoding="utf-8")).get("artifacts", []))
        if name not in allowed:
            raise FileNotFoundError(name)
        return path

    def _save(self, state: dict) -> None:
        state["updated_at"] = _now()
        atomic_json(self.directory / state["id"] / "job.json", state)

    def _monitor(self) -> None:
        while not self._stop.wait(0.10):
            with self._lock:
                for id_, entry in list(self._active.items()):
                    process, events, cancel, started = entry
                    state = self._states[id_]
                    try:
                        while True:
                            event = events.get_nowait()
                            kind = event["kind"]
                            if kind == "progress" and state["status"] != "cancelling":
                                state["stage"] = str(event["message"])
                                if event.get("label") == "CoilGen":
                                    state["progress"] = max(state["progress"], min(0.98, event["percent"] / 100))
                                state["log"] = (state["log"] + [f"{event.get('label', '')}: {event['message']}"])[-160:]
                            elif kind == "completed":
                                state.update(status="completed", stage="Complete", progress=1.0)
                            elif kind in {"failed", "cancelled"}:
                                state.update(status=kind, stage=kind.capitalize(), error=event.get("error"))
                            self._save(state)
                    except queue.Empty:
                        pass
                    timed_out = time.monotonic() - started > self.policy.timeout_seconds
                    force_cancel = state["status"] == "cancelling" and time.monotonic() - state["cancel_time"] > 1.0
                    if (
                        process.is_alive()
                        and state["status"] in {"running", "cancelling"}
                        and (timed_out or force_cancel)
                    ):
                        process.terminate()
                        process.join(timeout=2)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=2)
                        state.update(
                            status="failed" if timed_out else "cancelled",
                            stage="Time limit exceeded" if timed_out else "Cancelled",
                            error=f"Worker exceeded {self.policy.timeout_seconds:g} s." if timed_out else None,
                        )
                        self._save(state)
                    if not process.is_alive():
                        process.join(timeout=0.1)
                        if state["status"] in {"running", "cancelling"}:
                            # The queue feeder can trail the atomic result. A clean
                            # exit plus a valid result is evidence; a crash is not.
                            completed = False
                            if process.exitcode == 0 and not cancel.is_set():
                                try:
                                    payload = json.loads(
                                        (self.directory / id_ / "result.json").read_text(encoding="utf-8")
                                    )
                                    completed = isinstance(payload.get("summary"), dict) and bool(
                                        payload.get("request_sha256")
                                    )
                                except (OSError, ValueError, AttributeError):
                                    pass
                            status = "cancelled" if cancel.is_set() else "completed" if completed else "failed"
                            state.update(
                                status=status,
                                stage=status.capitalize(),
                                progress=1.0 if completed else state["progress"],
                                error=None
                                if cancel.is_set() or completed
                                else f"Worker exited without a result (exit {process.exitcode}).",
                            )
                            self._save(state)
                        events.close()
                        del self._active[id_]
                for id_, state in self._states.items():
                    if len(self._active) >= self.policy.max_workers:
                        break
                    if state["status"] != "queued":
                        continue
                    events = self._context.Queue(maxsize=256)
                    cancel = self._context.Event()
                    try:
                        request = json.loads((self.directory / id_ / "request.json").read_text(encoding="utf-8"))
                        process = self._context.Process(
                            target=_worker,
                            args=(request, str(self.directory / id_), events, cancel, self.policy.threads_per_worker),
                        )
                        process.start()
                    except Exception as exc:
                        events.close()
                        state.update(
                            status="failed", stage="Worker could not start", error=f"{type(exc).__name__}: {exc}"
                        )
                        self._save(state)
                        continue
                    self._active[id_] = (process, events, cancel, time.monotonic())
                    state.update(status="running", stage="Starting isolated worker")
                    self._save(state)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        with self._lock:
            for id_, (process, events, cancel, _started) in self._active.items():
                cancel.set()
                if process.is_alive():
                    process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                events.close()
                self._states[id_].update(status="interrupted", stage="Runtime stopped")
                self._save(self._states[id_])
            self._active.clear()
            for state in self._states.values():
                if state["status"] == "queued":
                    state.update(status="interrupted", stage="Runtime stopped before execution")
                    self._save(state)
            self._owner.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
