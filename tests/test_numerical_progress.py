from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.basis import calculate_basis_functions, calculate_one_ring_by_mesh
from coilgen.config import CoilGenConfig
from coilgen.contours import extract_contours, filter_loop_significance
from coilgen.evaluation import calculate_unconnected_contour_field, evaluate_field_errors
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from coilgen.inductance import calculate_inductance_by_coil_layout
from coilgen.models import CoilPart, ContourLine, Mesh, TargetField, WirePath
from coilgen.optimization import _scan_l_curve_by_transformed_svd, _solve_stream_function_system
from coilgen.physics import biot_savart_wire_path, calculate_resistance_matrix, calculate_sensitivity_matrix
from coilgen.progress import DesignCancelled, ProgressReporter, progress_context


def _config(**options):
    return CoilGenConfig(
        progress_flag=False,
        progress_min_interval_seconds=0,
        progress_percent_step=0,
        gauss_order=2,
        **options,
    )


def _supports():
    mesh = normalize_mesh(build_cylinder_mesh(0.3, 0.1, 8, 4, 1, 0, 0, 0))
    other = deepcopy(mesh)
    other.vertices += [0, 0, 0.35]
    return [CoilPart(mesh), CoilPart(other)]


def _target():
    points = np.array([[0, 0, 0], [0.01, 0.02, 0.03], [0, 0, 0.35]])
    return TargetField(points, np.zeros_like(points), np.ones(3), np.zeros(3, bool), np.ones(3, int))


def _percents(events, label):
    return [event["percent"] for event in events if event["label"] == label and event["percent"] is not None]


def _assert_counted_progress(events, label):
    percentages = _percents(events, label)
    assert percentages[0] == 0
    assert percentages[-1] == 100
    assert percentages == sorted(percentages)
    assert sum(value == 0 for value in percentages) == 1
    assert any(0 < value < 50 for value in percentages)
    assert any(50 < value < 100 for value in percentages)


def test_basis_and_one_ring_report_completed_work_across_supports_without_changing_values():
    plain, reported = _supports(), _supports()
    calculate_basis_functions(calculate_one_ring_by_mesh(plain))
    events = []
    with progress_context(events.append):
        calculate_basis_functions(calculate_one_ring_by_mesh(reported))
    for label in ("One-ring", "Basis"):
        _assert_counted_progress(events, label)
    for expected, actual in zip(plain, reported):
        assert np.array_equal(expected.current_density_mat, actual.current_density_mat)
        for before, after in zip(expected.basis_elements, actual.basis_elements):
            for name in ("triangles", "one_ring", "area", "face_normal", "triangle_points_abc", "current"):
                assert np.array_equal(getattr(before, name), getattr(after, name))


@pytest.mark.parametrize("workers", [1, 2])
def test_sensitivity_reports_serial_and_parallel_work_without_support_resets(workers):
    parts = calculate_basis_functions(calculate_one_ring_by_mesh(_supports()))
    expected = deepcopy(parts)
    config = _config(parallel_worker_count=workers, parallel_min_work_size=0)
    calculate_sensitivity_matrix(expected, _target(), config)
    events = []
    with progress_context(events.append):
        calculate_sensitivity_matrix(parts, _target(), config)
    _assert_counted_progress(events, "Sensitivity")
    for before, after in zip(expected, parts):
        assert np.array_equal(before.sensitivity_matrix, after.sensitivity_matrix)


def test_resistance_reports_matrix_entries_without_support_resets():
    parts = calculate_basis_functions(calculate_one_ring_by_mesh(_supports()))
    expected = deepcopy(parts)
    calculate_resistance_matrix(expected, _config())
    events = []
    with progress_context(events.append):
        calculate_resistance_matrix(parts, _config())
    _assert_counted_progress(events, "Resistance")
    for before, after in zip(expected, parts):
        assert np.array_equal(before.resistance_matrix, after.resistance_matrix)


def test_contour_progress_counts_levels_across_supports_and_preserves_geometry():
    parts = _supports()
    for part in parts:
        part.stream_function = part.mesh.vertices[:, 2].copy()
        part.potential_level_list = np.linspace(part.stream_function.min(), part.stream_function.max(), 7)[1:-1]
    expected = deepcopy(parts)
    config = _config(contour_symmetry_validation="off", min_point_loop_number=3)
    extract_contours(expected, config)
    events = []
    with progress_context(events.append):
        extract_contours(parts, config)
    _assert_counted_progress(events, "Contours")
    for before, after in zip(expected, parts):
        assert len(before.contour_lines) == len(after.contour_lines) > 0
        for before_line, after_line in zip(before.contour_lines, after.contour_lines):
            assert np.array_equal(before_line.v, after_line.v)


def _loops():
    angle = np.linspace(0, 2 * np.pi, 17)
    loop = np.column_stack((0.05 * np.cos(angle), 0.05 * np.sin(angle), np.zeros_like(angle)))
    return [
        CoilPart(Mesh((loop + offset)[:3], np.array([[0, 1, 2]])), wire_path=WirePath(v=loop + offset))
        for offset in (np.zeros(3), np.array([0, 0, 0.04]))
    ]


@pytest.mark.parametrize("workers", [1, 2])
def test_inductance_reports_all_self_and_mutual_chunks_with_console_progress_disabled(workers):
    config = _config(
        skip_inductance_calculation=False,
        parallel_worker_count=workers,
        parallel_min_work_size=0,
        inductance_chunk_size=2,
    )
    expected = calculate_inductance_by_coil_layout(_loops(), config)
    events = []
    with progress_context(events.append):
        actual = calculate_inductance_by_coil_layout(_loops(), config)
    _assert_counted_progress(events, "Inductance")
    for before, after in zip(expected[1:], actual[1:]):
        assert np.array_equal(before, after)


@pytest.mark.parametrize("workers", [1, 2])
def test_wire_chunks_report_to_context_with_console_progress_disabled(workers):
    points = _loops()[0].wire_path.v
    kwargs = dict(chunk_size=2, worker_count=workers, min_work_size=0)
    expected = biot_savart_wire_path(points, _target().coords, **kwargs)
    events = []
    with progress_context(events.append):
        actual = biot_savart_wire_path(
            points, _target().coords, progress=ProgressReporter.from_config(_config(), label="Wire"), **kwargs
        )
    _assert_counted_progress(events, "Wire")
    assert np.array_equal(expected, actual)


def test_wire_chunk_progress_preserves_cancellation_without_console_output():
    events = []
    with progress_context(events.append, lambda: any(event.get("percent", 0) >= 25 for event in events)):
        with pytest.raises(DesignCancelled):
            biot_savart_wire_path(
                _loops()[0].wire_path.v,
                _target().coords,
                chunk_size=2,
                progress=ProgressReporter(label="Wire", percent_step=0, min_interval_seconds=0),
            )
    assert max(_percents(events, "Wire")) < 100


def test_transformed_svd_reports_real_candidates_and_marks_factorization_indeterminate():
    problem = SimpleNamespace(
        reduced_resistance=np.diag([1.0, 2.0, 3.0]),
        reduced_sensitivity=np.diag([1.0, 2.0, 4.0]),
        target_z=np.array([1.0, 2.0, 3.0]),
    )
    lambdas = np.logspace(-3, 3, 8)
    expected = _scan_l_curve_by_transformed_svd(problem, lambdas, _config())
    events = []
    with progress_context(events.append):
        actual = _scan_l_curve_by_transformed_svd(problem, lambdas, _config())
    assert any(event["percent"] is None and "SVD" in event["message"] for event in events)
    _assert_counted_progress(events, "L-curve")
    assert actual[2] == expected[2] == "transformed_svd"
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


@pytest.mark.parametrize("solver", ["solve", "pinv"])
def test_linear_factorizations_report_activity_without_guessing_a_percentage(solver):
    lhs, rhs = np.diag([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 9.0])
    events = []
    with progress_context(events.append):
        actual = _solve_stream_function_system(lhs, rhs, _config(optimization_solver=solver))
    assert actual == pytest.approx([1, 2, 3])
    assert events and all(event["percent"] is None for event in events)
    assert all(event["label"] == "Linear solve" for event in events)


def _contour_loops():
    parts = _loops()
    for part in parts:
        part.contour_step = 2.0
        part.contour_lines = [
            ContourLine(v=part.wire_path.v * scale, potential=scale, turn_multiplicity=index)
            for index, scale in enumerate((0.8, 1, 1.2), start=1)
        ]
    return parts


@pytest.mark.parametrize("workers", [1, 2])
def test_contour_evaluation_aggregates_supports_and_contains_nested_wire_progress(workers):
    config = _config(parallel_worker_count=workers, parallel_min_work_size=0, biot_savart_chunk_size=2)
    expected = calculate_unconnected_contour_field(_contour_loops(), config, _target())
    events = []
    with progress_context(events.append):
        actual = calculate_unconnected_contour_field(_contour_loops(), config, _target())
    percentages = _percents(events, "Field evaluation")
    assert percentages == sorted(percentages)
    assert percentages[0] == 0 and percentages[-1] == 100
    assert len(set(percentages)) >= 7
    assert np.array_equal(expected, actual)


def test_whole_field_evaluation_reserves_completion_until_current_and_metrics_are_finished(monkeypatch):
    from coilgen import evaluation

    config = _config(biot_savart_chunk_size=2, field_evaluation_current_source="stream_function")
    target = _target()
    target.b[:, 2] = [1e-6, 2e-6, 3e-6]
    sf_field = target.b * 0.8
    expected = evaluate_field_errors(_contour_loops(), config, target, sf_field)
    events = []
    original_current = evaluation._field_evaluation_current

    def check_current(*args, **kwargs):
        assert max(_percents(events, "Field evaluation")) == 95
        return original_current(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_field_evaluation_current", check_current)
    with progress_context(events.append):
        actual = evaluate_field_errors(_contour_loops(), config, target, sf_field)
    percentages = _percents(events, "Field evaluation")
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100
    assert len(set(percentages)) > 15
    for before, after in zip(expected[:4], actual[:4]):
        assert np.array_equal(before, after)
    assert expected[4] == actual[4]
    assert expected[5] == actual[5] == 2.0
    assert np.array_equal(expected[6], actual[6])
    assert np.array_equal(expected[7].b, actual[7].b)


@pytest.mark.parametrize("workers", [1, 2])
def test_loop_significance_aggregates_fields_and_pruning_review(workers):
    config = _config(
        min_loop_signifcance=0.1,
        contour_symmetry_validation="off",
        biot_savart_chunk_size=2,
        parallel_worker_count=workers,
        parallel_min_work_size=0,
    )
    expected, parts = _contour_loops(), _contour_loops()
    filter_loop_significance(expected, _target(), config)
    events = []
    with progress_context(events.append):
        filter_loop_significance(parts, _target(), config)
    percentages = _percents(events, "Loop significance")
    assert percentages == sorted(percentages)
    assert percentages[0] == 0 and percentages[-1] == 100
    assert len(set(percentages)) >= 8
    for before, after in zip(expected, parts):
        assert np.array_equal(before.loop_significance, after.loop_significance)
        for before_line, after_line in zip(before.contour_lines, after.contour_lines):
            assert np.array_equal(before_line.v, after_line.v)


def test_nested_field_progress_cancellation_never_finishes_other_supports():
    events = []
    with progress_context(events.append, lambda: any((event.get("percent") or 0) >= 20 for event in events)):
        with pytest.raises(DesignCancelled):
            calculate_unconnected_contour_field(_contour_loops(), _config(biot_savart_chunk_size=2), _target())
    percentages = _percents(events, "Field evaluation")
    assert percentages and max(percentages) < 100


def test_regularization_batches_cannot_finish_a_multipass_spacing_review(monkeypatch):
    from coilgen import optimization

    problem = SimpleNamespace(
        regularization_scale=1.0,
        normal_equation_matrices=lambda: (np.eye(2), np.eye(2), np.ones(2)),
    )
    monkeypatch.setattr(
        optimization,
        "_turn_spacing_penalty_context",
        lambda *_: {
            "enabled": True,
            "max_attempts": 3,
            "initial_weight": 1.0,
            "scaled_metric": np.eye(2),
            "threshold": 0.25,
            "growth": 2.0,
        },
    )
    monkeypatch.setattr(optimization, "_stream_function_smoothness_context", lambda *_: {"quadratic": None})
    monkeypatch.setattr(optimization, "_stream_function_smoothness_diagnostics", lambda *_: {})
    monkeypatch.setattr(optimization, "_turn_spacing_diagnostics", lambda *_: {})
    events, checks = [], []

    def scan(*args, **kwargs):
        task = ProgressReporter(label="L-curve", percent_step=0).task("candidates", 4)
        for _ in range(4):
            task.advance()
        return optimization.StreamFunctionSolution(np.ones(2), {})

    def spacing(*args):
        # Each locally complete scan still has spacing checks and possible
        # subsequent scans to perform within the enclosing solve.
        completed = [event["percent"] for event in events if event["percent"] is not None]
        assert max(completed) < 100
        checks.append(max(completed))
        return (0.1, 0.15, 0.3)[len(checks) - 1]

    monkeypatch.setattr(optimization, "_l_curve_regularization_choice", scan)
    monkeypatch.setattr(optimization, "_estimate_min_turn_distance", spacing)
    with progress_context(events.append):
        solution = optimization._select_tikhonov_regularization(problem, _config())
    assert np.array_equal(solution.reduced_stream_function, np.ones(2))
    assert checks == pytest.approx([95 / 3, 190 / 3, 95])
    percentages = [event["percent"] for event in events if event["percent"] is not None]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100


def test_optimization_cancellation_does_not_run_fallbacks_or_retry_other_solvers(monkeypatch):
    from coilgen import optimization

    def cancelled(*args, **kwargs):
        raise DesignCancelled("stop")

    monkeypatch.setattr(optimization, "_select_constrained_quadratic", cancelled)
    monkeypatch.setattr(
        optimization, "_constrained_scan_fallback", lambda *_args, **_kwargs: pytest.fail("Cancelled solve fell back")
    )
    with pytest.raises(DesignCancelled):
        optimization.solve_constrained_quadratic_stream_function(SimpleNamespace(), _config())

    calls = []

    def target_choice(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise DesignCancelled("stop retry")
        return optimization.StreamFunctionSolution(np.ones(2), {})

    monkeypatch.setattr(optimization, "_target_error_constrained_efficiency_choice", target_choice)
    monkeypatch.setattr(optimization, "_target_error_auto_retry_attempt_summary", lambda *_a, **_k: {})
    monkeypatch.setattr(optimization, "_target_error_solution_violation_ratio", lambda *_: 1.0)
    monkeypatch.setattr(optimization, "_target_error_solution_meets_constraint", lambda *_: False)
    with pytest.raises(DesignCancelled):
        optimization._target_error_constrained_efficiency_choice_with_retries(
            SimpleNamespace(), np.eye(2), np.eye(2), np.ones(2), 1.0, _config(target_error_auto_retry_max_reruns=2)
        )
    assert len(calls) == 2


def test_cvxpy_first_solver_success_does_not_count_unused_fallback_attempts(monkeypatch):
    from coilgen import optimization

    cvxpy = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cvxpy.installed_solvers():
        pytest.skip("CLARABEL is required for this solver-progress regression")
    monkeypatch.setattr(optimization, "_cvxpy_conic_solver_order", lambda *_: ["CLARABEL", "SCS"])
    block = optimization.ResidualConstraintBlock("field", np.eye(1), np.ones(1), 0.1, 1.0, 10.0, 0.0)
    events = []
    with progress_context(events.append):
        solution, diagnostics = optimization._solve_constrained_quadratic_cvxpy(
            SimpleNamespace(), np.eye(1), (block,), _config()
        )
    assert solution == pytest.approx([0.9], abs=1e-5)
    assert diagnostics["solver"] == "CLARABEL"
    attempts = [event for event in events if "total" in event]
    assert attempts and all(event["total"] == 2 and event["percent"] is None for event in attempts)
    assert max(event["completed"] for event in attempts) == 1
    assert events[-1]["percent"] == 100
    assert "after 1 solver attempt" in events[-1]["message"]
