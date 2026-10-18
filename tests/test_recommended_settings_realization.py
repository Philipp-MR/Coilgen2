"""Routed recommendations retain the body required by single-circuit audits."""

from dataclasses import asdict

import numpy as np
import pytest

from coilgen import CoilGenConfig, apply_recommended_settings, recommended_settings
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import create_swept_coil_meshes
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry


ROUTED_PRESETS = ("validation", "balanced", "active_shielded_gradient")


@pytest.mark.parametrize("name", ROUTED_PRESETS)
@pytest.mark.parametrize("scope", ("single", "per_part"))
def test_routed_recommendations_retain_circuit_scope_and_physical_inputs(name, scope):
    base = CoilGenConfig(
        coil_mesh_file="create bi-planar mesh",
        routing_circuit_scope=scope,
        skip_sweep=True,
        target_gradient_strength=0.017,
        conductor_cross_section_width=0.0006,
        conductor_cross_section_height=0.0004,
        sweep_min_bend_radius=0.0012,
        min_inter_group_connector_clearance=0.003,
        max_wire_current_A=24.0,
    )
    original = asdict(base)
    from_config = apply_recommended_settings(base, preset=name)
    from_mapping = CoilGenConfig(**apply_recommended_settings(original, preset=name))

    assert asdict(base) == original
    assert asdict(from_config) == asdict(from_mapping)
    assert from_config.routing_circuit_scope == scope
    assert not from_config.skip_postprocessing
    assert not from_config.skip_sweep
    assert not from_config.contour_only_sweep_mode
    for field in (
        "coil_mesh_file",
        "target_gradient_strength",
        "conductor_cross_section_width",
        "conductor_cross_section_height",
        "sweep_min_bend_radius",
        "min_inter_group_connector_clearance",
        "max_wire_current_A",
    ):
        assert getattr(from_config, field) == original[field]
    if name == "active_shielded_gradient":
        assert from_config.field_evaluation_current_source == "stream_function"
        assert from_config.target_error_constraint_mode == "separate"
        assert from_config.target_error_tolerance_percent == 5.0
        assert from_config.suppression_error_tolerance_percent == 5.0


@pytest.mark.parametrize("name", ROUTED_PRESETS)
def test_routed_recommendation_sweeps_a_closed_route_for_physical_audit(name):
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0], [-0.02, 0.02, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )
    theta = np.linspace(0, 2 * np.pi, 181)
    points = np.column_stack((0.006 * np.cos(theta), 0.006 * np.sin(theta), np.zeros(len(theta))))
    points[-1] = points[0]
    part = CoilPart(
        mesh=mesh,
        wire_path=WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * (len(points) - 1)),
    )
    config = recommended_settings(name).apply_to_config(
        CoilGenConfig(
            routing_circuit_scope="single",
            skip_sweep=True,
            conductor_cross_section_width=0.0006,
            conductor_cross_section_height=0.0004,
            sweep_min_bend_radius=0.0012,
            sweep_bend_mode="fillet",
            sweep_max_frame_rotation_deg=5.0,
            sweep_max_chord_error=5e-6,
        )
    )

    create_swept_coil_meshes([part], config)

    assert part.fabrication_path is not None
    assert part.fabrication_path.closed
    assert part.swept_body is not None
    assert len(part.swept_body.vertices) > 0 and len(part.swept_body.faces) > 0
    assert part.manufacturing_diagnostics["manufacturing_ok"]


@pytest.mark.parametrize("name", ("quick", "quick_debug", "local_b0_shim"))
def test_contour_only_recommendations_keep_their_no_sweep_scope(name):
    config = recommended_settings(name).apply_to_config(CoilGenConfig(routing_circuit_scope="single"))
    assert config.routing_circuit_scope == "single"
    assert config.skip_postprocessing
    assert config.skip_sweep
