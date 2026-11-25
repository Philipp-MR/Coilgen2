"""Structural contracts for immutable results and queued design execution.

Adapters can implement these contracts without inheriting application classes.
Legacy minimal adapters remain supported by the existing service fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from ..research.requests import ResearchRequest


@runtime_checkable
class RunRecordView(Protocol):
    @property
    def directory(self) -> Path: ...

    @property
    def metadata(self) -> dict: ...

    @property
    def metadata_view(self) -> Mapping[str, Any]: ...

    @property
    def execution_times(self) -> dict[str, str]: ...

    @property
    def array_names(self) -> tuple[str, ...]: ...

    @property
    def arrays(self) -> dict[str, np.ndarray]: ...

    def array(self, name: str, *, copy: bool = True) -> np.ndarray: ...

    def select_arrays(self, names: Iterable[str] | None = None, *, copy: bool = True) -> dict[str, np.ndarray]: ...


@runtime_checkable
class JobRuntime(Protocol):
    @property
    def directory(self) -> Path: ...

    def submit(self, value: ResearchRequest | dict, *, reuse_active: bool = False) -> dict: ...

    def list(self) -> list[dict]: ...

    def get(self, id_: str) -> dict: ...

    def cancel(self, id_: str) -> dict: ...

    def close(self) -> None: ...
