"""User-facing recomputation plans derived from the pipeline reuse contract."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CoilGenConfig


@dataclass(frozen=True, slots=True)
class DependencyPlan:
    changed_fields: tuple[str, ...]
    stages: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def earliest_stage(self) -> str | None:
        return self.stages[0] if self.stages else None

    @property
    def requires_computation(self) -> bool:
        return bool(self.stages)


def dependency_plan(old: CoilGenConfig, new: CoilGenConfig) -> DependencyPlan:
    """Describe invalidation; it never promises automatic partial reuse.

    Unknown inputs conservatively invalidate preparation. Wire dimensions,
    material and current/spacing limits remain upstream because they may alter
    the sheet objective, constraint construction or contour conditioning.
    """
    from ..pipeline_stages import changed_config_fields, config_field_stage, invalidated_stage, PipelineStage

    changed = tuple(sorted(changed_config_fields(old, new)))
    classified = [
        (
            name,
            min(
                (
                    stage
                    for stage in (config_field_stage(name, old), config_field_stage(name, new))
                    if stage is not None
                ),
                key=lambda stage: stage.index,
                default=None,
            ),
        )
        for name in changed
    ]
    ordered = tuple(PipelineStage)
    earliest = invalidated_stage(old, new)
    if earliest is None:
        return DependencyPlan(
            changed, (), ("Display/export settings do not change the numerical model.",) if changed else ()
        )
    stages = tuple(stage.value for stage in ordered[ordered.index(earliest) :])
    reasons = tuple(
        f"{name}: invalidate {stage.value} and downstream results." for name, stage in classified if stage is not None
    )
    return DependencyPlan(changed, stages, reasons)
