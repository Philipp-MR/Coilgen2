from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig, CoilPart, Mesh, TargetField, WirePath, create_swept_coil_meshes
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from coilgen.models import WireSegmentOwner
from coilgen.physics import biot_savart_wire_path
from coilgen.research.fabrication import FabricationRecord, compile_fabrication, winding_instructions
from coilgen.research.materials import WireSheetMaterial


def _plane():
    return Mesh(
        vertices=np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]), faces=np.array([[0, 1, 2], [0, 2, 3]])
    )


def test_mesh_realization_preserves_fem_trace_and_reports_analytic_field_change():
    radius = 0.12
    parameters = (0.3, radius, 12, 6, 1, 0, 0, 0)
    support = normalize_mesh(build_cylinder_mesh(*parameters))
    midpoint = (support.vertices[0] + support.vertices[1]) / 2
    midpoint[2] = 0
    points = np.array([midpoint + [0, 0, -0.03], midpoint, midpoint + [0, 0, 0.03]])
    coordinates = np.array([[0, 0, 0.02], [0.02, 0.01, 0.04]])
    current = 2.0
    part = CoilPart(mesh=support, wire_path=WirePath(v=points), contour_step=current)
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=parameters,
        surface_is_cylinder_flag=True,
        sweep_bend_mode="fillet",
        sweep_min_bend_radius=0.002,
        conductor_cross_section_width=0.0002,
        conductor_cross_section_height=0.0002,
        target_field_component="vector",
        sweep_max_chord_error=0,
    )
    target = TargetField(
        coords=coordinates,
        b=current * biot_savart_wire_path(points, coordinates),
        weights=np.ones(2),
        suppressed=np.zeros(2, dtype=bool),
        target_field_group_inds=np.zeros(2, dtype=int),
    )
    result = SimpleNamespace(coil_parts=[part], input_data=config, target_field=target)
    mesh = compile_fabrication(result, kernel_backend="reference")
    analytic = compile_fabrication(result, kernel_backend="reference", surface_realization="analytic")
    mesh_path, analytic_path = mesh.paths[0], analytic.paths[0]
    np.testing.assert_allclose(
        mesh_path.points_m[:, :2], np.broadcast_to(midpoint[:2], mesh_path.points_m[:, :2].shape), atol=1e-14
    )
    np.testing.assert_allclose(np.linalg.norm(analytic_path.points_m[:, :2], axis=1), radius, atol=1e-14)
    assert mesh.field_metrics["surface_realization"] == "mesh"
    assert mesh.field_metrics["dsv_source_to_fabrication_delta_rms_T"] < 1e-18
    assert analytic.field_metrics["dsv_source_to_fabrication_delta_rms_T"] > 1e-10
    assert analytic_path.diagnostics["sweep_source_to_centroid_max_midpoint_displacement_m"] > 0.003
    np.testing.assert_array_equal(part.wire_path.v, points)
    assert result.input_data.sweep_surface_realization == "analytic"  # Original config unchanged.


def test_final_mounted_centroid_radius_can_reject_a_previously_passing_support_trace(tmp_path):
    angle = np.linspace(0, 2 * np.pi, 49)
    points = np.column_stack((0.04 * np.cos(angle), 0.04 * np.sin(angle), np.zeros_like(angle)))
    points[-1] = points[0]
    part = CoilPart(mesh=_plane(), wire_path=WirePath(v=points), contour_step=2.0)
    config = CoilGenConfig(
        cross_sectional_points=((-0.0122, -0.0001), (-0.0118, -0.0001), (-0.0118, 0.0001), (-0.0122, 0.0001)),
        sweep_path_simplification="off",
        sweep_max_chord_error=0,
        sweep_min_bend_radius=0.029,
    )
    create_swept_coil_meshes([part], config)
    diagnostics = part.manufacturing_diagnostics
    assert not diagnostics["sweep_min_bend_radius_violation"]
    assert diagnostics["sweep_final_min_bend_radius_violation"]
    assert diagnostics["sweep_final_min_bend_radius_m"] < 0.99 * config.sweep_min_bend_radius
    assert "minimum_bend_radius_violation" in diagnostics["manufacturing_failure_reasons"]
    assert not diagnostics["manufacturing_ok"]


def test_winding_operations_follow_provenance_independently_of_sampling_density():
    plans = []
    for count in (21, 201):
        points = np.column_stack((np.linspace(-0.1, 0.1, count), np.zeros((count, 2))))
        half = (count - 1) // 2
        owners = [WireSegmentOwner(kind="contour_loop", group_index=0, loop_index=2)] * half + [
            WireSegmentOwner(kind="inter_group_connector", connector_index=0, connector_role="primary")
        ] * half
        part = CoilPart(mesh=_plane(), wire_path=WirePath(v=points, segment_owners=owners))
        config = CoilGenConfig(
            sweep_path_simplification="off",
            sweep_max_segment_length=0,
            sweep_max_chord_error=0,
            sweep_frame_angle_tolerance_deg=0,
            conductor_cross_section_width=0.0001,
            conductor_cross_section_height=0.0001,
        )
        create_swept_coil_meshes([part], config)
        record = FabricationRecord(
            "part001_body0001", 1, None, part.fabrication_path, part.swept_body, part.manufacturing_diagnostics
        )
        plan = winding_instructions(record, checkpoint_spacing_m=0.025)
        plans.append(plan)
        assert [step["operation"] for step in plan["steps"]] == ["wind_turn", "interconnect"]
        assert plan["steps"][0]["loop_index"] == 2
        assert plan["steps"][1]["connector_index"] == 0
        assert sum(step["interval_length_m"] for step in plan["steps"]) == pytest.approx(0.2)
        stations = np.array([station["arclength_m"] for station in plan["checkpoints"]])
        assert np.max(np.diff(stations)) <= 0.025 + 1e-14
        assert not plan["layer_assignment_designed"]
        assert not plan["drive_feed_designed"]
    assert plans[0]["operation_count"] == plans[1]["operation_count"] == 2
    assert plans[1]["sampled_interval_count"] > plans[0]["sampled_interval_count"]


def test_material_homogenization_matches_parallel_wire_joule_loss():
    material = WireSheetMaterial.round_wire(0.001, 0.004, resistivity_ohm_m=1.8e-8)
    current, length, turns = 3.0, 0.4, 12
    surface_current = current / material.pitch_m
    sheet_power = material.sheet_resistance_ohm * surface_current**2 * length * turns * material.pitch_m
    wire_power = current**2 * material.resistivity_ohm_m * length * turns / material.wire_area_m2
    assert sheet_power == pytest.approx(wire_power, rel=1e-14)
    config = CoilGenConfig()
    matched = material.apply_to(config)
    assert matched.conductor_thickness == pytest.approx(np.pi * 0.001**2 / 4 / 0.004)
    assert matched.specific_conductivity_conductor == material.resistivity_ohm_m
    assert config.conductor_thickness != matched.conductor_thickness
    assert replace(matched, conductor_cross_section_width=0.01).conductor_thickness == matched.conductor_thickness


@pytest.mark.parametrize("diameter,pitch", [(0, 0.004), (0.001, 0), (0.002, 0.001), (np.nan, 0.004)])
def test_invalid_wire_sheet_material_rejected(diameter, pitch):
    with pytest.raises(ValueError):
        WireSheetMaterial.round_wire(diameter, pitch)
