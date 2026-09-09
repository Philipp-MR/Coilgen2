from types import SimpleNamespace
import tracemalloc

import numpy as np
import pytest

from projects.general_examples.scripts import _benchmark_scaling_common as scaling
from projects.general_examples.scripts import benchmark_performance as performance
from projects.general_examples.scripts import benchmark_sweep_case_cache_scaling as cache
from projects.general_examples.scripts import generated_parameter_sweep as sweep
from projects.general_examples.scripts import run_standard_examples as standard


def test_generated_sweep_forwards_geometry_and_conductor_physics():
    parameters = sweep.get_parameters()
    parameters["coilgen"].update(
        cylinder_height=0.7,
        cylinder_radius=0.13,
        conductor_cross_section_width=0.002,
        conductor_cross_section_height=0.003,
        specific_conductivity_conductor=2.2e-8,
        inductance_source="unconnected_contours",
    )
    config = sweep._sweep_config(8, 0.3, (12, 6), 25.0, 0.004, parameters)
    assert config.cylinder_mesh_parameter_list == (0.7, 0.13, 12, 6, 1, 0, 0, 0)
    assert config.conductor_cross_section_width == 0.002
    assert config.conductor_cross_section_height == 0.003
    assert config.specific_conductivity_conductor == 2.2e-8
    assert config.inductance_source == "unconnected_contours"
    assert parameters["coilgen"]["cylinder_radius"] == 0.13


def test_performance_rows_read_current_summary_metric_names(monkeypatch):
    monkeypatch.setattr(
        performance,
        "result_summary_dict",
        lambda result: {
            "execution_time_s": 0.2,
            "symmetry_reduction_factor": 1,
            "contour_count": 8,
            "layout_mean_rel_error_percent": 2.5,
            "layout_max_rel_error_percent": 4.0,
        },
    )
    result = SimpleNamespace(regularization_diagnostics={"unknown_count": 10})
    row = performance.benchmark_row({"name": "serial", "cpu_workers": 1, "symmetry_reduction": "off"}, 1, result, 0.3)
    assert row["mean_layout_error_percent"] == 2.5
    assert row["max_layout_error_percent"] == 4.0


def test_benchmark_cache_always_executes_cold_pass_and_verifies_warm_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "example_output_dir", lambda name: tmp_path)
    monkeypatch.setattr(
        cache,
        "PARAMETERS",
        {
            "output_name": "cache_test",
            "radius_values_m": (0.16,),
            "resolution_values": (3,),
            "retain_case_results": False,
        },
    )
    original_workflow = cache.ParameterSweepWorkflow
    executions = []

    class ObservedWorkflow(original_workflow):
        def __init__(self):
            super().__init__()
            original_run = self.case_workflow.run

            def run(spec):
                executions.append(spec.name)
                return original_run(spec)

            self.case_workflow.run = run

    monkeypatch.setattr(cache, "ParameterSweepWorkflow", ObservedWorkflow)
    for _ in range(2):
        cold, warm = cache.main()
        assert len(cold.rows()) == len(warm.cached_rows) == 1
        assert not cold.case_results
        assert not warm.case_results
    assert len(executions) == 2


def test_scaling_benchmark_releases_memory_tracing_after_failed_design(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("failed design")

    monkeypatch.setattr(scaling, "run_coilgen", fail)
    with pytest.raises(RuntimeError, match="failed design"):
        scaling.run_timed_config("failed", scaling.base_benchmark_config())
    assert not tracemalloc.is_tracing()


def test_standard_runner_rejects_misspelled_selection():
    parameters = standard.get_parameters()
    parameters["run"]["selected_examples"] = ("misspelled_coil",)
    with pytest.raises(ValueError, match="misspelled_coil"):
        standard.main(parameters, headless=True)


def test_generated_sweep_preserves_previous_default_physical_geometry():
    config = sweep._sweep_config(6, 0.25, (12, 6), 50.0, 0.02, sweep.get_parameters())
    np.testing.assert_array_equal(config.cylinder_mesh_parameter_list[:2], [0.5, 0.25])
