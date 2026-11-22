"""Compatibility imports for engine-owned regional requirements.

The implementation lives in coilgen.research_requirements and participates
in numerical source identity. Existing project and script imports remain valid.
"""

from .. import research_requirements as _engine
from ..research_requirements import (
    FieldSamplingError as FieldSamplingError,
    DesiredDataset as DesiredDataset,
    RegionalRequirement as RegionalRequirement,
    PhysicalScenario as PhysicalScenario,
    ResearchRequirements as ResearchRequirements,
    requirements_stage_fingerprints as requirements_stage_fingerprints,
    requirements_target as requirements_target,
    requirements_evaluation_config as requirements_evaluation_config,
    requirements_validation_config as requirements_validation_config,
    prepare_requirement_constraints as prepare_requirement_constraints,
    evaluate_requirements as evaluate_requirements,
    nominal_requirements as nominal_requirements,
    named_independent_sheet_validation as named_independent_sheet_validation,
    requirements_constraint_rows as requirements_constraint_rows,
)


def __getattr__(name):
    return getattr(_engine, name)
