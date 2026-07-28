from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CoilApplication(str, Enum):
    """Broad design intent for context-aware coil evaluation."""

    FREE = "free"
    MR_GRADIENT = "mr_gradient"
    MR_SHIM = "mr_shim"
    MR_RF = "mr_rf"


class EvaluationFeature(str, Enum):
    """Evaluation/reporting feature controlled by a design context."""

    FIELD_ERROR = "field_error"
    ELECTRICAL = "electrical"
    ROUTING = "routing"
    MANUFACTURING = "manufacturing"
    GRADIENT_FIT = "gradient_fit"
    TARGET_MODE_FIT = "target_mode_fit"
    CONCOMITANT_FIELD = "concomitant_field"
    SHIM_HOMOGENEITY = "shim_homogeneity"
    SHIM_SPHERICAL_HARMONICS = "shim_spherical_harmonics"
    RF_SCREENING = "rf_screening"


_DEFAULT_FEATURES: dict[CoilApplication, frozenset[EvaluationFeature]] = {
    CoilApplication.FREE: frozenset(
        {
            EvaluationFeature.FIELD_ERROR,
            EvaluationFeature.ELECTRICAL,
            EvaluationFeature.ROUTING,
            EvaluationFeature.MANUFACTURING,
        }
    ),
    CoilApplication.MR_GRADIENT: frozenset(
        {
            EvaluationFeature.FIELD_ERROR,
            EvaluationFeature.ELECTRICAL,
            EvaluationFeature.ROUTING,
            EvaluationFeature.MANUFACTURING,
            EvaluationFeature.GRADIENT_FIT,
            EvaluationFeature.TARGET_MODE_FIT,
            EvaluationFeature.CONCOMITANT_FIELD,
        }
    ),
    CoilApplication.MR_SHIM: frozenset(
        {
            EvaluationFeature.FIELD_ERROR,
            EvaluationFeature.ELECTRICAL,
            EvaluationFeature.ROUTING,
            EvaluationFeature.MANUFACTURING,
            EvaluationFeature.SHIM_HOMOGENEITY,
            EvaluationFeature.SHIM_SPHERICAL_HARMONICS,
        }
    ),
    CoilApplication.MR_RF: frozenset(
        {
            EvaluationFeature.ELECTRICAL,
            EvaluationFeature.MANUFACTURING,
            EvaluationFeature.RF_SCREENING,
        }
    ),
}

_APPLICATION_ALIASES = {
    "free_coil": CoilApplication.FREE,
    "freefield": CoilApplication.FREE,
    "free_field": CoilApplication.FREE,
    "generic": CoilApplication.FREE,
    "none": CoilApplication.FREE,
    "gradient": CoilApplication.MR_GRADIENT,
    "gradient_coil": CoilApplication.MR_GRADIENT,
    "mr_gradient_coil": CoilApplication.MR_GRADIENT,
    "mri_gradient": CoilApplication.MR_GRADIENT,
    "mri_gradient_coil": CoilApplication.MR_GRADIENT,
    "shim": CoilApplication.MR_SHIM,
    "shim_coil": CoilApplication.MR_SHIM,
    "b0_shim": CoilApplication.MR_SHIM,
    "b0_shim_coil": CoilApplication.MR_SHIM,
    "mr_shim_coil": CoilApplication.MR_SHIM,
    "mri_shim": CoilApplication.MR_SHIM,
    "mri_shim_coil": CoilApplication.MR_SHIM,
    "rf": CoilApplication.MR_RF,
    "rf_coil": CoilApplication.MR_RF,
    "mr_rf_coil": CoilApplication.MR_RF,
    "mri_rf": CoilApplication.MR_RF,
    "mri_rf_coil": CoilApplication.MR_RF,
}


@dataclass(frozen=True, slots=True)
class CoilDesignContext:
    """Typed design context used to select domain-specific evaluation."""

    application: CoilApplication = CoilApplication.MR_GRADIENT
    extra_features: frozenset[EvaluationFeature] = field(default_factory=frozenset)
    disabled_features: frozenset[EvaluationFeature] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "application", normalize_coil_application(self.application))
        object.__setattr__(
            self,
            "extra_features",
            frozenset(normalize_evaluation_feature(feature) for feature in self.extra_features),
        )
        object.__setattr__(
            self,
            "disabled_features",
            frozenset(normalize_evaluation_feature(feature) for feature in self.disabled_features),
        )

    @classmethod
    def free(cls, **kwargs: Any) -> "CoilDesignContext":
        return cls(CoilApplication.FREE, **kwargs)

    @classmethod
    def mr_gradient(cls, **kwargs: Any) -> "CoilDesignContext":
        return cls(CoilApplication.MR_GRADIENT, **kwargs)

    @classmethod
    def mr_shim(cls, **kwargs: Any) -> "CoilDesignContext":
        return cls(CoilApplication.MR_SHIM, **kwargs)

    @classmethod
    def mr_rf(cls, **kwargs: Any) -> "CoilDesignContext":
        return cls(CoilApplication.MR_RF, **kwargs)

    @property
    def key(self) -> str:
        return self.application.value

    @property
    def is_mr(self) -> bool:
        return self.application in {
            CoilApplication.MR_GRADIENT,
            CoilApplication.MR_SHIM,
            CoilApplication.MR_RF,
        }

    @property
    def features(self) -> frozenset[EvaluationFeature]:
        return (_DEFAULT_FEATURES[self.application] | self.extra_features) - self.disabled_features

    def enabled(self, feature: EvaluationFeature | str) -> bool:
        return normalize_evaluation_feature(feature) in self.features

    def feature_keys(self) -> tuple[str, ...]:
        return tuple(sorted(feature.value for feature in self.features))


def normalize_coil_application(value: CoilApplication | str) -> CoilApplication:
    if isinstance(value, CoilApplication):
        return value
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _APPLICATION_ALIASES:
        return _APPLICATION_ALIASES[key]
    try:
        return CoilApplication(key)
    except ValueError as exc:
        allowed = ", ".join(application.value for application in CoilApplication)
        raise ValueError(f"Unknown coil design context {value!r}. Expected one of: {allowed}.") from exc


def normalize_evaluation_feature(value: EvaluationFeature | str) -> EvaluationFeature:
    if isinstance(value, EvaluationFeature):
        return value
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return EvaluationFeature(key)
    except ValueError as exc:
        allowed = ", ".join(feature.value for feature in EvaluationFeature)
        raise ValueError(f"Unknown evaluation feature {value!r}. Expected one of: {allowed}.") from exc


def normalize_design_context(value: CoilDesignContext | CoilApplication | str | None) -> CoilDesignContext:
    if value is None:
        return CoilDesignContext()
    if isinstance(value, CoilDesignContext):
        return value
    return CoilDesignContext(normalize_coil_application(value))


def design_context_key(value: CoilDesignContext | CoilApplication | str | None) -> str:
    return normalize_design_context(value).key
