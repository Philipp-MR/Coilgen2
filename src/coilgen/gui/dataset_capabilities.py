"""Presentation adapter for the shared headless scientific evidence catalog."""

from __future__ import annotations

from dataclasses import dataclass

from ..application.evidence import FIELD_KEYS, PATH_PREFIXES, REQUIRED_STEPS, WIRE_FIELDS
from ..application.evidence import dataset_capabilities as _dataset_capabilities
from ..application.evidence import points_available
from ..application.evidence import resolve_dataset as _resolve_dataset


_STEP_LABELS = {
    "support": "Define support",
    "target": "Define target and ROI",
    "prepared": "Prepare",
    "sheet": "Optimize sheet",
    "contours": "Generate contours",
    "routed": "Connect / sweep",
    "evaluated": "Evaluate",
    "fabrication_review": "Fabrication review",
}
REQUIRED_STAGES = {key: _STEP_LABELS[step] for key, step in REQUIRED_STEPS.items()}


@dataclass(frozen=True, slots=True)
class DatasetCapability:
    """Keep the established plot-menu labels and positional constructor."""

    available: bool
    reason: str = ""
    required_stage: str = ""
    invalid: bool = False

    @property
    def explanation(self):
        return self.reason + (f" Required step: {self.required_stage}." if self.required_stage else "")


def _present(capability):
    return DatasetCapability(
        capability.available,
        capability.reason,
        _STEP_LABELS.get(capability.required_step, ""),
        capability.invalid,
    )


def resolve_dataset(arrays, key, *, original=False):
    return _present(_resolve_dataset(arrays, key, original=original))


def dataset_capabilities(arrays, *, original=False):
    return {key: _present(capability) for key, capability in _dataset_capabilities(arrays, original=original).items()}


__all__ = [
    "DatasetCapability",
    "FIELD_KEYS",
    "WIRE_FIELDS",
    "PATH_PREFIXES",
    "REQUIRED_STAGES",
    "resolve_dataset",
    "points_available",
    "dataset_capabilities",
]
