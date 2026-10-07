"""Disjoint support indexing and directed coverage of one assembly circuit."""

from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.connector_bridge_validation import _validate_support
from coilgen.contours import extract_contours
from coilgen.geometry import normalize_mesh
from coilgen.group_routing import GroupRoutingLimits, propose_joint_group_routes
from coilgen.joint_cut_variants import prepare_joint_group_context
from coilgen.layout import calculate_group_centers, interconnect_within_groups, topological_loop_grouping
from coilgen.manufacturing import _SurfaceProjector
from coilgen.models import CoilPart, ContourSparsificationResult, Mesh
from coilgen.routing_assembly import assemble_routing_parts, validate_routing_assembly


def _source_parts(*, separation=0.09, levels=(0.00013, 0.00029)):
    parts = []
    for index, shift in enumerate((-separation / 2, separation / 2)):
        x, y = np.meshgrid(np.linspace(-0.03, 0.03, 7), np.linspace(-0.03, 0.03, 7))
        vertices = np.column_stack((x.ravel() + shift, y.ravel(), np.zeros(x.size)))
        faces = []
        for row in range(6):
            for column in range(6):
                start = row * 7 + column
                faces.extend(((start, start + 1, start + 8), (start, start + 8, start + 7)))
        mesh = normalize_mesh(Mesh(vertices, np.asarray(faces), metadata={"source": index}))
        sign = 1 if index == 0 else -1
        stream = sign * (x.ravel() ** 2 + y.ravel() ** 2)
        mesh.stream_function = stream.copy()
        part = CoilPart(
            mesh=mesh,
            stream_function=stream,
            potential_level_list=sign * np.asarray(levels),
            contour_step=1.0,
            current_density=np.full((mesh.n_faces, 3), float(sign)),
        )
        parts.append(part)
    config = CoilGenConfig(min_point_loop_number=4, progress_flag=False)
    return calculate_group_centers(topological_loop_grouping(extract_contours(parts, config), config=config)), config


def test_assembly_preserves_surface_geometry_provenance_and_original_primary():
    parts, _config = _source_parts()
    original = pickle.dumps(parts)
    result = assemble_routing_parts(parts, primary_surface_index=1)[0]
    mapping = result.topology_diagnostics["routing_assembly"]
    assert mapping["original_primary_surface_index"] == 1
    assert mapping["group_source_surface_indices"] == [0, 1]
    assert mapping["contour_source_surface_indices"] == [0, 0, 1, 1]
    np.testing.assert_array_equal(result.potential_level_list, parts[1].potential_level_list)
    np.testing.assert_array_equal(result.stream_function, np.concatenate([part.stream_function for part in parts]))
    np.testing.assert_array_equal(result.current_density, np.concatenate([part.current_density for part in parts]))
    for source, part in zip(mapping["sources"], parts):
        va, vb = source["vertex_range"]
        fa, fb = source["face_range"]
        ca, cb = source["contour_range"]
        np.testing.assert_array_equal(result.mesh.vertices[va:vb], part.mesh.vertices)
        np.testing.assert_array_equal(result.mesh.faces[fa:fb] - va, part.mesh.faces)
        for actual, expected in zip(result.contour_lines[ca:cb], part.contour_lines):
            np.testing.assert_array_equal(actual.v, expected.v)
            np.testing.assert_array_equal(actual.segment_faces - fa, expected.segment_faces)
            np.testing.assert_array_equal(actual.surface_edge_vertices - va, expected.surface_edge_vertices)
            reconstructed = np.einsum(
                "nki,nij->nkj",
                actual.segment_barycentric,
                result.mesh.vertices[result.mesh.faces[actual.segment_faces]],
            )
            np.testing.assert_allclose(reconstructed[:, 0], actual.v[:-1], atol=1e-15)
            np.testing.assert_allclose(reconstructed[:, 1], actual.v[1:], atol=1e-15)
            assert actual.current_orientation == expected.current_orientation
    assert pickle.dumps(parts) == original
    result.contour_lines[0].v[0, 0] += 1
    assert pickle.dumps(parts) == original


@pytest.mark.parametrize("levels", [(0.00013,), (0.00013, 0.00029)])
def test_global_proposal_visits_every_source_once_with_its_directed_intervals(levels):
    from collections import Counter

    parts, config = _source_parts(levels=levels)
    merged = assemble_routing_parts(parts)[0]
    interconnect_within_groups([merged], config)
    context = prepare_joint_group_context(merged, config)
    proposed = propose_joint_group_routes(
        context, config, limits=GroupRoutingLimits(return_fractions=(0.0,)), transition_evaluator=lambda *_args: 0.0
    )
    assert proposed.candidates
    for candidate in proposed.candidates:
        assert sorted(candidate.group_sequence) == [0, 1]
        np.testing.assert_array_equal(candidate.path.v[0], candidate.path.v[-1])
        counts = Counter(identity for identity in candidate.segment_ids if identity is not None)
        assert counts == Counter({identity: 1 for identity in candidate.protected_segments})
        for segment, identity in enumerate(candidate.segment_ids):
            if identity is not None:
                np.testing.assert_array_equal(
                    candidate.path.v[segment : segment + 2], candidate.protected_segments[identity]
                )
    assert not merged.topology_diagnostics["routing_assembly"]["physical_feasibility_proven"]


def test_disjoint_union_does_not_invent_support_across_the_gap():
    parts, _config = _source_parts()
    merged = assemble_routing_parts(parts)[0]
    points = np.array([[-0.025, 0.0, 0.0], [0.025, 0.0, 0.0]])
    reasons = _validate_support(
        _SurfaceProjector(merged.mesh),
        points[0],
        points[None, :, :],
        np.array([0.0, 0.0, 1.0]),
        0.01,
        1e-8,
        {},
    )
    assert reasons
    assert any(
        reason in {"lift_limit_violation", "support_domain_violation", "support_boundary_crossing"}
        for reason in reasons
    )


@pytest.mark.parametrize("current", [None, 0.0, np.nan, np.inf, 2.0, -1.0])
def test_unequal_or_invalid_series_current_rejects_without_mutation(current):
    parts, _config = _source_parts()
    parts[1].contour_step = current
    original = pickle.dumps(parts)
    with pytest.raises(ValueError, match="contour_step"):
        assemble_routing_parts(parts)
    assert pickle.dumps(parts) == original


def test_source_nested_hierarchy_and_membership_are_namespaced():
    parts, _config = _source_parts()
    for part in parts:
        group = part.groups[0]
        part.groups = [
            replace(deepcopy(group), loops=[loop], loop_indices=[i], outer_loop_indices=(i,))
            for i, loop in enumerate(part.contour_lines)
        ]
        part.group_levels = [[0], [1]]
        part.level_positions = [[], [0]]
        part.group_centers_v = None
    result = assemble_routing_parts(parts)[0]
    assert result.group_levels == [[0, 2], [1], [3]]
    assert result.level_positions == [[], [0], [2]]
    assert [group.loop_indices for group in result.groups] == [[0], [1], [2], [3]]
    assert [group.outer_loop_indices for group in result.groups] == [(0,), (1,), (2,), (3,)]
    for group in result.groups:
        assert group.loops[0] is result.contour_lines[group.loop_indices[0]]
    np.testing.assert_array_equal(result.loop_in_loop_matrix[:2, 2:], False)
    np.testing.assert_array_equal(result.loop_in_loop_matrix[2:, :2], False)


def test_assembly_preserves_sparsification_source_namespace():
    parts, _config = _source_parts()
    parts[1].contour_sparsification = ContourSparsificationResult(
        source_contours=deepcopy(parts[1].contour_lines),
        final_source_contour_indices=[(0,), (1,)],
        diagnostics={"source_test": True},
    )
    result = assemble_routing_parts(parts)[0]
    provenance = result.contour_sparsification
    assert provenance.final_source_contour_indices == [(0,), (1,), (2,), (3,)]
    assert len(provenance.source_contours) == 4
    np.testing.assert_array_equal(provenance.source_contours[2].v, parts[1].contour_lines[0].v)
    assert result.topology_diagnostics["routing_assembly"]["sources"][1]["sparsification_diagnostics"] == {
        "source_test": True
    }


def test_single_part_is_unchanged_and_already_routed_sources_reject():
    parts, config = _source_parts()
    single = parts[:1]
    assert assemble_routing_parts(single) is single
    interconnect_within_groups(parts, config)
    with pytest.raises(ValueError, match="before routing"):
        assemble_routing_parts(parts)


def test_assembly_rejects_missing_or_duplicated_group_coverage():
    parts, _config = _source_parts()
    parts[1].groups[0].loop_indices = [0, 0]
    with pytest.raises(ValueError, match="exactly one group"):
        assemble_routing_parts(parts)


def _completed_assembly(**kwargs):
    from test_group_route_support import _swept_circle

    part, config = _swept_circle(**kwargs)
    part.wire_path.v[-1] = part.wire_path.v[0]
    part.topology_diagnostics["routing_assembly"] = {"source_part_count": 2}
    return part, config


def test_final_assembly_gate_audits_the_actual_body_and_support():
    part, config = _completed_assembly()
    assert part.manufacturing_diagnostics["manufacturing_ok"]
    assert "exact_collision_completed" not in part.manufacturing_diagnostics
    parts = [part]
    assert validate_routing_assembly(parts, config) is parts
    evidence = part.route_diagnostics["routing_assembly_validation"]
    assert evidence["single_circuit_certified"]
    assert evidence["manufacturing"]["exact_collision_completed"]
    assert evidence["manufacturing"]["exact_intersection_count"] == 0
    assert not evidence["field_acceptance_checked"]


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("split", "multiple independent"),
        ("open", "not_closed"),
        ("no_sweep", "missing_closed_fabrication_path"),
        ("manufacturing", "manufacturing_not_certified"),
        ("stale_body", "body_section_frame_mismatch"),
        ("field", "fixed_current_acceptance_evidence"),
    ],
)
def test_final_assembly_gate_rejects_incomplete_or_uncertified_routes(defect, reason):
    part, config = _completed_assembly()
    parts = [part]
    if defect == "split":
        parts.append(deepcopy(part))
    elif defect == "open":
        part.wire_path.v[-1, 0] += 0.001
    elif defect == "no_sweep":
        part.fabrication_path = None
        part.swept_body = None
    elif defect == "manufacturing":
        part.manufacturing_diagnostics = {"manufacturing_ok": False, "mesh_solid_ok": True}
    elif defect == "stale_body":
        part.swept_body.vertices[0, 2] += 0.001
    else:
        part.joint_route_diagnostics = {"status": "accepted"}
    with pytest.raises(ValueError, match=reason):
        validate_routing_assembly(parts, config)


def test_final_assembly_gate_retains_lift_limits_despite_legacy_manufacturing_pass():
    part, config = _completed_assembly(height=0.003, lift=0.002)
    assert part.manufacturing_diagnostics["manufacturing_ok"]
    with pytest.raises(ValueError, match="reference_support_side_or_lift_violation"):
        validate_routing_assembly([part], config)


def test_final_gate_is_inactive_for_legacy_independent_parts():
    parts, config = _source_parts()
    assert validate_routing_assembly(parts, config) is parts
