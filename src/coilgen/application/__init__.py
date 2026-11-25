"""Portable application services, imported only when requested."""

from importlib import import_module as _import_module

_EXPORT_MODULES = {
    "PROJECT_SCHEMA": "project",
    "ProjectDocument": "project",
    "DesignService": "service",
    "RunRecord": "service",
    "RunRecordView": "interfaces",
    "JobRuntime": "interfaces",
    "PreflightIssue": "preflight",
    "PreflightReport": "preflight",
    "check_preflight": "preflight",
    "SourceHealth": "evidence",
    "VerificationReport": "evidence",
    "dataset_capabilities": "evidence",
    "resolve_dataset": "evidence",
    "verify_run_source": "evidence",
    "ExecutionEvent": "execution",
    "ExecutionPlan": "execution",
    "RunActions": "execution",
    "execution_reuse": "execution",
    "plan_execution": "execution",
    "run_actions": "execution",
    "RunArchive": "bundles",
    "DependencyPlan": "dependencies",
    "dependency_plan": "dependencies",
    "ASSEMBLY_SCHEMA": "assembly",
    "ChannelDocument": "assembly",
    "ProjectAssembly": "assembly",
    "DesignStudyState": "design_studies",
    "RunReference": "design_studies",
    "create_variant": "design_studies",
    "design_study_state": "design_studies",
    "pin_baseline": "design_studies",
    "set_study_notes": "design_studies",
    "set_study_notes_batch": "design_studies",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORT_MODULES.keys())
