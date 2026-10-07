from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback
import uuid

from .requests import ResearchRequest
from .storage import DirectoryOwner, atomic_json, file_digest, request_identity, verified_result, verify_input_files
from ..pipeline_stages import PipelineStage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_completed_boundary(state: dict, event: dict) -> None:
    """Persist monotonic completed boundaries; activity and percentages prove none."""
    if event.get("label") != "CoilGen":
        return
    try:
        stage = PipelineStage(event.get("completed_stage"))
        limit = PipelineStage(state.get("pipeline_stage", "evaluated"))
        previous = PipelineStage(state["completed_stage"]).index if state.get("completed_stage") else -1
    except (ValueError, TypeError):
        return
    completed = [item.value for item in PipelineStage if item.index <= stage.index]
    if event.get("completed_stages") != completed or not previous <= stage.index <= limit.index:
        return
    state.update(completed_stage=stage.value, completed_stages=completed)
    if event.get("stage_preview") == "stage_preview.json":
        state.update(stage_preview="stage_preview.json", stage_preview_stage=stage.value)
        state.pop("stage_preview_error", None)
    elif isinstance(event.get("stage_preview_error"), str):
        state["stage_preview_error"] = event["stage_preview_error"][:512]


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
            if event.get("completed_stage") is not None:
                events.put({"kind": "progress", **event}, timeout=0.25)
            else:
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

    def submit(self, value: ResearchRequest | dict, *, reuse_active: bool = False) -> dict:
        """Queue work, optionally reusing an exactly identified active request.

        Reuse checks stored provenance and study controls under the queue lock.
        Missing legacy controls never establish a match. Ordinary submissions
        continue to create independent jobs, including identical requests.
        """
        if not isinstance(reuse_active, bool):
            raise TypeError("reuse_active must be a boolean.")
        request = ResearchRequest.from_dict(value.to_dict() if isinstance(value, ResearchRequest) else value)
        config = request.to_config(threads=self.policy.threads_per_worker)
        controls = {key: item for key, item in request.to_dict().items() if key not in {"name", "config"}}
        request = ResearchRequest.from_config(config, name=request.name, **controls)
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("The runtime is shutting down.")
            if (
                reuse_active
                and request.project_id.strip()
                and request.project_fingerprint.strip()
                and request.project_revision is not None
            ):
                identity = (
                    "project_id",
                    "project_fingerprint",
                    "project_revision",
                    "pipeline_stage",
                    "validation_resolution",
                    "design_margin",
                    "fabricate",
                    "kernel_backend",
                )
                candidate_digest = None
                for state in reversed(self._states.values()):
                    if state.get("status") in {"queued", "running", "cancelling"} and all(
                        key in state and state[key] == getattr(request, key) for key in identity
                    ):
                        # Project fingerprints are caller-supplied provenance.
                        # Also prove actual numerical settings, input contents,
                        # checkpoint bytes, engine source and solver versions.
                        if candidate_digest is None:
                            candidate_digest, _manifest = request_identity(request.to_dict(), asdict(config))
                        if state.get("request_sha256") == candidate_digest:
                            return deepcopy(state)
            pending = sum(s["status"] in {"queued", "running", "cancelling"} for s in self._states.values())
            if pending >= self.policy.max_pending:
                raise ValueError(f"The queue is full ({self.policy.max_pending} pending jobs).")
            id_ = uuid.uuid4().hex
            directory = self.directory / id_
            directory.mkdir()
            # Queue execution can start much later. A run must use the submitted
            # mesh/target/checkpoint, not whatever happens to occupy that path
            # when a worker starts. Files are owned by this run before queuing.
            replacements = {}
            input_directory = directory / "inputs"
            for key, value in request.config.items():
                if not key.endswith("_file") or not isinstance(value, str):
                    continue
                source = Path(value)
                if not source.is_file():
                    continue
                input_directory.mkdir(exist_ok=True)
                copied = input_directory / f"{key}{source.suffix}"
                before = file_digest(source)
                shutil.copyfile(source, copied)
                if before != file_digest(copied) or before != file_digest(source):
                    raise RuntimeError(f"Input {key} changed while its job snapshot was being created.")
                replacements[key] = str(copied)
            if request.resume_checkpoint:
                source = Path(request.resume_checkpoint).expanduser().resolve()
                if not source.is_file():
                    raise ValueError(f"Resume checkpoint does not exist: {source}.")
                input_directory.mkdir(exist_ok=True)
                copied = input_directory / "resume.cgstage"
                before = file_digest(source)
                shutil.copyfile(source, copied)
                if before != file_digest(copied) or before != file_digest(source):
                    raise RuntimeError("Resume checkpoint changed while its job snapshot was being created.")
                request = replace(request, resume_checkpoint=str(copied))
            request = replace(request, config={**request.config, **replacements})
            digest, _manifest = request_identity(
                request.to_dict(), asdict(request.to_config(threads=self.policy.threads_per_worker))
            )
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
                "request_sha256": digest,
                "project_id": request.project_id,
                "project_revision": request.project_revision,
                "project_fingerprint": request.project_fingerprint,
                "project_model_fingerprint": request.project_model_fingerprint,
                "pipeline_stage": request.pipeline_stage,
                "validation_resolution": request.validation_resolution,
                "design_margin": request.design_margin,
                "fabricate": request.fabricate,
                "kernel_backend": request.kernel_backend,
                "completed_stage": None,
                "completed_stages": [],
            }
            self._states[id_] = state
            self._save(state)
            return deepcopy(state)

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
            return deepcopy(state)

    def result(self, id_: str) -> dict:
        state = self.get(id_)
        if state["status"] != "completed":
            raise ValueError("This job has no completed result.")
        result = self._verified_completion(id_)
        if result is None:
            raise ValueError("Completed result is missing or its evidence integrity check failed.")
        result.update(id=id_, name=state["name"])
        return result

    def _verified_completion(self, id_: str) -> dict | None:
        """Completion requires the submitted identity and every evidence artifact."""
        directory = self.directory / id_
        digest = self._states[id_].get("request_sha256")
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            # Older durable jobs are readable, but still require consistency
            # between their result and provenance manifest.
            digest = digest or manifest.get("request_sha256")
            if not isinstance(digest, str) or manifest.get("request_sha256") != digest:
                return None
            result = verified_result(directory, digest)
            if result is None or not isinstance(result.get("summary"), dict):
                return None
            verify_input_files(manifest)
            return result
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
            return None

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
                            if state.get("ipc_error"):
                                break
                            try:
                                event = events.get_nowait()
                            except (EOFError, OSError, ValueError) as exc:
                                # Killing a process during its queue feeder write
                                # can break this channel. Keep supervising the
                                # process/result and other jobs independently.
                                state["ipc_error"] = f"{type(exc).__name__}: {exc}"
                                self._save(state)
                                break
                            kind = event["kind"]
                            if kind == "progress" and state["status"] != "cancelling":
                                _record_completed_boundary(state, event)
                                state["stage"] = str(event["message"])
                                if event.get("label") == "CoilGen" and event.get("percent") is not None:
                                    state["progress"] = max(state["progress"], min(0.98, event["percent"] / 100))
                                state["log"] = (state["log"] + [f"{event.get('label', '')}: {event['message']}"])[-160:]
                            elif kind == "completed":
                                if cancel.is_set():
                                    state.update(status="cancelled", stage="Cancelled")
                                elif self._verified_completion(id_) is not None:
                                    state.update(status="completed", stage="Complete", progress=1.0)
                                else:
                                    state.update(
                                        status="failed",
                                        stage="Invalid completion evidence",
                                        error="Worker result failed its submitted identity or evidence integrity check.",
                                    )
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
                            completed = (
                                process.exitcode == 0
                                and not cancel.is_set()
                                and self._verified_completion(id_) is not None
                            )
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
