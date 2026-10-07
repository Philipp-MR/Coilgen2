"""Portable scientific projects and application operations, without GUI imports."""

from .dependencies import DependencyPlan, dependency_plan
from .project import PROJECT_SCHEMA, ProjectDocument
from .service import DesignService, RunRecord
from .preflight import PreflightIssue, PreflightReport, check_preflight
from .evidence import SourceHealth, VerificationReport, dataset_capabilities, resolve_dataset, verify_run_source
from .execution import ExecutionEvent, ExecutionPlan, RunActions, execution_reuse, plan_execution, run_actions
from .assembly import ASSEMBLY_SCHEMA, ChannelDocument, ProjectAssembly
from .bundles import RunArchive
from .design_studies import (
    DesignStudyState,
    RunReference,
    create_variant,
    design_study_state,
    pin_baseline,
    set_study_notes,
    set_study_notes_batch,
)

__all__ = [
    "PROJECT_SCHEMA",
    "ProjectDocument",
    "DesignService",
    "RunRecord",
    "PreflightIssue",
    "PreflightReport",
    "check_preflight",
    "SourceHealth",
    "VerificationReport",
    "dataset_capabilities",
    "resolve_dataset",
    "verify_run_source",
    "ExecutionEvent",
    "ExecutionPlan",
    "RunActions",
    "execution_reuse",
    "plan_execution",
    "run_actions",
    "RunArchive",
    "DependencyPlan",
    "dependency_plan",
    "ASSEMBLY_SCHEMA",
    "ChannelDocument",
    "ProjectAssembly",
    "DesignStudyState",
    "RunReference",
    "create_variant",
    "design_study_state",
    "pin_baseline",
    "set_study_notes",
    "set_study_notes_batch",
]
