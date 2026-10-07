"""Assembly contracts across the pipeline's routing and reporting boundaries.

Small real meshes, extracted contours, sweeps and field evaluations are used.
The prescribed inter-group route and support certificate isolate orchestration
from the separately tested route search and cross-support bridge feasibility.
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import group_route_support, group_route_validation, manufacturing, pipeline
from coilgen.config import CoilGenConfig
from coilgen.evaluation import layout_field_geometry
from coilgen.geometry import combine_meshes, normalize_mesh
from coilgen.models import Mesh, TargetField, WirePath
from coilgen.physics import biot_savart_wire_path


@pytest.fixture
def assembly_pipeline(monkeypatch):
    meshes = []
    x, y = np.meshgrid(np.linspace(-0.03, 0.03, 7), np.linspace(-0.03, 0.03, 7))
    faces = []
    for row in range(6):
        for column in range(6):
            i = 7 * row + column
            faces.extend(((i, i + 1, i + 8), (i, i + 8, i + 7)))
    for shift in (-0.045, 0.045):
        meshes.append(
            normalize_mesh(
                Mesh(
                    np.column_stack((x.ravel() + shift, y.ravel(), np.zeros(x.size))),
                    np.asarray(faces),
                )
            )
        )
    coords = np.array([[0, 0, 0.01], [0.003, 0, 0.015], [0, 0.004, 0.02], [-0.002, -0.003, 0.025]])
    target = TargetField(coords, np.full_like(coords, 1e-6), np.ones(4), np.zeros(4, bool), np.ones(4, int))
    config = CoilGenConfig(
        coil_mesh_file="create planary mesh",
        levels=1,
        min_point_loop_number=4,
        smooth_flag=False,
        skip_normal_shift=True,
        skip_inductance_calculation=True,
        conductor_cross_section_width=0.0005,
        conductor_cross_section_height=0.0005,
        field_evaluation_current_source="target_projection",
        joint_group_routing=True,
        progress_flag=False,
        parallel_worker_count=1,
    )
    state = SimpleNamespace(
        config=config,
        events=[],
        source_currents=(1.0, 1.0),
        target_scale=1.0,
        invalidate_after_adoption=False,
        presweep_candidate=False,
        presweep_part_count=1,
        sweep_call_count=0,
        final_sweep_parts=None,
        selected_body=None,
        selected_trace=None,
        baseline_field=None,
        selected_field=None,
        adopted_parts=None,
        evaluated=None,
    )

    def optimize(parts, target_field, _config):
        for part in parts:
            local = part.mesh.vertices - part.mesh.vertices.mean(axis=0)
            part.stream_function = np.sum(local[:, :2] ** 2, axis=1)
            part.mesh.stream_function = part.stream_function.copy()
            part.current_density = np.ones((part.mesh.n_faces, 3))
        return parts, combine_meshes([part.mesh for part in parts]), target_field.b.copy(), {}

    def levels(parts, _mesh, _config):
        for part, current in zip(parts, state.source_currents):
            part.potential_level_list = np.array([0.00013])
            part.contour_step = current
        return parts, 0

    real_within = pipeline.interconnect_within_groups

    def within(parts, config, **kwargs):
        state.events.append("within")
        if config.routing_circuit_scope == "single":
            assert len(parts) == 1
            provenance = parts[0].topology_diagnostics["routing_assembly"]
            assert provenance["group_source_surface_indices"] == [0, 1]
            assert provenance["contour_source_surface_indices"] == [0, 1]
        else:
            assert len(parts) == 2
            assert all("routing_assembly" not in part.topology_diagnostics for part in parts)
        return real_within(parts, config, **kwargs)

    def among(parts, _config, **_kwargs):
        state.events.append("among")
        angle = np.linspace(0, 2 * np.pi, 65)
        for part in parts:
            center = part.contour_lines[0].v[:-1].mean(axis=0)
            points = center + 0.02 * np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
            points[-1] = points[0]
            part.wire_path = WirePath(points)
        return parts

    real_sweep = pipeline.create_swept_coil_meshes
    real_realize = pipeline.realize_coil_paths
    real_tessellate = manufacturing._sweep_wire_path_with_diagnostics

    def realize(parts, config):
        state.events.append("realize")
        return real_realize(parts, config)

    def tessellate(*args, **kwargs):
        # Centerline realization is permitted before route selection, but
        # conductor boundary tessellation must wait until selection starts.
        assert "adopt" in state.events
        state.sweep_call_count += 1
        return real_tessellate(*args, **kwargs)

    def sweep(parts, config):
        if not parts:
            return real_sweep(parts, config)
        state.events.append("sweep")
        assert all(any(part is selected for selected in state.adopted_parts) for part in parts)
        state.final_sweep_parts = real_sweep(parts, config)
        return state.final_sweep_parts

    def adopt(parts, config, target_field):
        state.events.append("adopt")
        assert all(part.swept_body is None and not part.swept_bodies for part in parts)
        if not config.joint_group_routing:
            assert all(part.fabrication_path is None for part in parts)
            state.adopted_parts = parts
            return parts
        assert all(part.fabrication_path is not None for part in parts) == (not config.skip_sweep)
        state.baseline_field = sum(
            (biot_savart_wire_path(layout_field_geometry(part, config)[0], coords, worker_count=1) for part in parts),
            start=np.zeros_like(coords),
        )
        # Route selection changes the geometry. Reporting and the final audit
        # must inspect this returned route rather than its provisional baseline.
        parts = deepcopy(parts)
        for part in parts:
            center = part.wire_path.v[:-1].mean(axis=0)
            part.wire_path = WirePath(center + 0.75 * (part.wire_path.v - center))
        parts = real_realize(parts, config)
        if state.presweep_candidate:
            state.events.append("candidate_sweep")
            prepared = real_sweep(parts[: state.presweep_part_count], config)
            state.selected_body = parts[0].swept_body
            state.selected_trace = parts[0].fabrication_path
            for part in prepared:
                part.joint_route_diagnostics = {
                    "status": "accepted",
                    "selected_attempt_index": 0,
                    "attempts": [
                        {
                            "selected": True,
                            "eligible_for_adoption": True,
                            "reasons": [],
                            "validation": {
                                "route_checked": True,
                                "physical_checked": True,
                                "field_current_a": part.contour_step,
                                "route_current_refitted": False,
                                "physical_current_refitted": False,
                            },
                        }
                    ],
                }
        state.adopted_parts = parts
        total = sum(
            (biot_savart_wire_path(layout_field_geometry(part, config)[0], coords, worker_count=1) for part in parts),
            start=np.zeros_like(coords),
        )
        state.selected_field = total
        target_field.b = total * state.target_scale
        return parts

    real_body_audit = group_route_validation._check_swept_geometry

    def body_audit(part, limits):
        state.events.append("body_audit")
        assert part is state.adopted_parts[0]
        if state.presweep_candidate:
            assert part.swept_body is state.selected_body
            assert part.fabrication_path is state.selected_trace
        else:
            assert part is state.final_sweep_parts[0]
        return real_body_audit(part, limits)

    def support_audit(part, _config):
        state.events.append("support_audit")
        assert part is state.adopted_parts[0]
        reasons = ("final_support_violation",) if state.invalidate_after_adoption else ()
        return SimpleNamespace(reasons=reasons, diagnostics={"checked": True})

    real_evaluate = pipeline.evaluate_field_errors

    def evaluate(*args):
        state.events.append("evaluate")
        state.evaluated = real_evaluate(*args)
        return state.evaluated

    monkeypatch.setattr(pipeline, "load_mesh_from_config", lambda *_args: combine_meshes(meshes))
    monkeypatch.setattr(pipeline, "define_target_field", lambda *_args: target)
    monkeypatch.setattr(pipeline, "stream_function_optimization", optimize)
    monkeypatch.setattr(pipeline, "calc_potential_levels", levels)
    monkeypatch.setattr(pipeline, "interconnect_within_groups", within)
    monkeypatch.setattr(pipeline, "interconnect_among_groups", among)
    monkeypatch.setattr(pipeline, "realize_coil_paths", realize)
    monkeypatch.setattr(pipeline, "create_swept_coil_meshes", sweep)
    monkeypatch.setattr(manufacturing, "_sweep_wire_path_with_diagnostics", tessellate)
    monkeypatch.setattr(pipeline, "improve_group_routes", adopt)
    monkeypatch.setattr(group_route_validation, "_check_swept_geometry", body_audit)
    monkeypatch.setattr(group_route_support, "certify_group_route_support", support_audit)
    monkeypatch.setattr(pipeline, "evaluate_field_errors", evaluate)
    state.run = lambda: pipeline.CoilGenPipeline(config).run()
    return state


def test_default_single_assembles_before_routing_and_certifies_adopted_body(assembly_pipeline):
    case = assembly_pipeline
    assert case.config.routing_circuit_scope == "single"
    result = case.run()
    assert len(result.coil_parts) == 1
    assert case.events == ["within", "among", "realize", "adopt", "sweep", "body_audit", "support_audit", "evaluate"]
    assert case.sweep_call_count == 1
    np.testing.assert_allclose(result.field_layout_per1Amp, case.selected_field, rtol=1e-12, atol=0)
    assert np.linalg.norm(result.field_layout_per1Amp - case.baseline_field) > 1e-9
    diagnostics = result.regularization_diagnostics
    assert diagnostics["routing_source_part_count"] == 2
    assert diagnostics["routing_parts_assembled"]
    assert diagnostics["field_evaluation_geometry_source"] == "fabricated_centroid"
    certificate = result.coil_parts[0].route_diagnostics["routing_assembly_validation"]
    assert certificate["single_circuit_certified"]
    assert certificate["manufacturing"]["exact_collision_completed"]
    assert certificate["field_acceptance_checked"]
    field_gate = diagnostics["routing_assembly_fixed_current_field"]
    assert field_gate["checked"]
    assert field_gate["current_a"] == 1.0
    assert not field_gate["current_refitted"]
    assert certificate["final_fixed_current_field"] == field_gate


def test_single_assembly_without_a_swept_body_fails_before_final_evaluation(assembly_pipeline):
    case = assembly_pipeline
    case.config.skip_sweep = True
    with pytest.raises(ValueError, match="missing_swept_body"):
        case.run()
    assert case.events == ["within", "among", "realize", "adopt", "sweep"]
    assert case.evaluated is None


def test_final_independent_support_failure_rejects_after_adoption(assembly_pipeline):
    case = assembly_pipeline
    case.invalidate_after_adoption = True
    with pytest.raises(ValueError, match="final_support_violation"):
        case.run()
    assert case.adopted_parts[0].manufacturing_diagnostics["manufacturing_ok"]
    assert case.events[-3:] == ["sweep", "body_audit", "support_audit"]
    assert case.evaluated is None


def test_explicit_per_part_scope_keeps_independent_circuits_without_assembly_gate(assembly_pipeline):
    case = assembly_pipeline
    case.config.routing_circuit_scope = "per_part"
    case.config.skip_sweep = True
    result = case.run()
    assert len(result.coil_parts) == 2
    assert "body_audit" not in case.events
    assert "support_audit" not in case.events
    assert result.regularization_diagnostics["field_evaluation_geometry_source"] == "connected_wire"
    assert "routing_assembly_fixed_current_field" not in result.regularization_diagnostics


def test_accepted_candidate_reuses_its_validated_sweep_and_centroid(assembly_pipeline):
    case = assembly_pipeline
    case.presweep_candidate = True
    result = case.run()
    assert case.events == [
        "within",
        "among",
        "realize",
        "adopt",
        "candidate_sweep",
        "body_audit",
        "support_audit",
        "evaluate",
    ]
    assert case.sweep_call_count == 1
    assert case.final_sweep_parts is None
    assert result.coil_parts[0].swept_body is case.selected_body
    assert result.coil_parts[0].fabrication_path is case.selected_trace
    np.testing.assert_allclose(result.field_layout_per1Amp, case.selected_field, rtol=1e-12, atol=0)


def test_final_sweep_only_builds_unswept_selected_parts(assembly_pipeline):
    case = assembly_pipeline
    case.config.routing_circuit_scope = "per_part"
    case.presweep_candidate = True
    result = case.run()
    assert case.events == ["within", "among", "realize", "adopt", "candidate_sweep", "sweep", "evaluate"]
    assert case.sweep_call_count == 2
    assert len(case.final_sweep_parts) == 1
    assert case.final_sweep_parts[0] is result.coil_parts[1]
    assert result.coil_parts[0].swept_body is case.selected_body
    assert result.coil_parts[0].fabrication_path is case.selected_trace
    np.testing.assert_allclose(result.field_layout_per1Amp, case.selected_field, rtol=1e-12, atol=0)


def test_disabled_joint_routing_avoids_duplicate_path_realization(assembly_pipeline):
    case = assembly_pipeline
    case.config.joint_group_routing = False
    case.config.routing_circuit_scope = "per_part"
    result = case.run()
    assert case.events == ["within", "among", "adopt", "sweep", "evaluate"]
    assert case.sweep_call_count == len(result.coil_parts) == 2
    assert all(part.fabrication_path is not None and part.swept_body is not None for part in result.coil_parts)


def test_single_assembly_rejects_unequal_series_currents_before_interconnection(assembly_pipeline):
    case = assembly_pipeline
    case.source_currents = (1.0, 2.0)
    with pytest.raises(ValueError, match="equal contour_step"):
        case.run()
    assert case.events == []


def test_fitted_reporting_current_cannot_waive_final_fixed_current_field_limit(assembly_pipeline):
    case = assembly_pipeline
    case.target_scale = 3.0
    case.config.layout_target_error_rms_limit_T = 1e-12
    with pytest.raises(ValueError, match="fixed-current field constraints: layout_target_rms_limit"):
        case.run()
    assert case.evaluated[5] == pytest.approx(3.0)
    fitted_field, per_amp = case.evaluated[0], case.evaluated[2]
    np.testing.assert_allclose(fitted_field, 3.0 * per_amp, rtol=1e-12, atol=0)
    assert np.sqrt(np.mean((fitted_field[:, 2] - per_amp[:, 2]) ** 2)) > 1e-12
