"""Regular traversal preserves directed turns and intrinsic grouping."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.contours import extract_contours, reverse_contour_line
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from coilgen.layout import calculate_group_centers, interconnect_within_groups, topological_loop_grouping
from coilgen.layout import _coherent_axial_ring_group_indices, _inter_group_cut_alignment_state
from coilgen.models import CoilPart
from coilgen.regular_rings import closed_curve_centroid, regular_ring_order, ring_support_axis


def _config(policy="regular"):
    config = CoilGenConfig(interconnection_cut_width=0.003, topological_grouping_method="auto")
    config.joint_group_routing_selection_policy = policy
    return config


def _part(samples=24, axial_samples=8):
    mesh = normalize_mesh(build_cylinder_mesh(0.4, 0.2, samples, axial_samples, 1, 0, 0, 0))
    part = CoilPart(mesh=mesh, stream_function=mesh.vertices[:, 2].copy())
    part.potential_level_list = np.linspace(-0.13, 0.13, 6)
    part.contour_step = float(np.diff(part.potential_level_list)[0])
    part = extract_contours([part], CoilGenConfig(min_point_loop_number=8))[0]
    # Reproduce extraction/potential order unrelated to axial adjacency.
    part.contour_lines = [part.contour_lines[index] for index in [0, 5, 1, 4, 2, 3]]
    for index, loop in enumerate(part.contour_lines):
        loop.potential = float(index)
    return part


def _positions(part):
    return [float(closed_curve_centroid(loop.v)[2]) for loop in part.groups[0].loops]


def _roll_loop(loop, shift):
    for name in ("v", "surface_faces", "surface_barycentric", "surface_edge_vertices", "surface_edge_t"):
        value = getattr(loop, name, None)
        if value is None:
            continue
        rolled = np.roll(value[:-1], shift, axis=0)
        setattr(loop, name, np.concatenate((rolled, rolled[:1]), axis=0))
    for name in ("segment_faces", "segment_barycentric"):
        value = getattr(loop, name)
        if value is not None:
            setattr(loop, name, np.roll(value, shift, axis=0))


def test_regular_ring_traversal_removes_axial_backtracking_without_changing_directed_contours():
    part = _part()
    original = [deepcopy(loop) for loop in part.contour_lines]
    physical = topological_loop_grouping([deepcopy(part)], _config("physical"))[0]
    regular = topological_loop_grouping([part], _config())[0]
    assert len(physical.groups) == len(regular.groups) == 1
    assert set(physical.groups[0].loop_indices) == set(regular.groups[0].loop_indices)
    positions = np.asarray(_positions(regular))
    assert np.all(np.diff(positions) > 0) or np.all(np.diff(positions) < 0)
    record = regular.topology_diagnostics["regular_ring_traversal"][0]
    assert record["original_axial_travel_m"] > 2 * record["ordered_axial_travel_m"]
    for source, contour in zip(original, regular.contour_lines):
        np.testing.assert_array_equal(source.v, contour.v)
        np.testing.assert_array_equal(source.segment_faces, contour.segment_faces)
        np.testing.assert_array_equal(source.surface_barycentric, contour.surface_barycentric)
        assert source.current_orientation == contour.current_orientation
        assert source.potential == contour.potential
    for routed, config in ((physical, _config("physical")), (regular, _config())):
        calculate_group_centers([routed])
        interconnect_within_groups([routed], config)
    old_travel = np.abs(np.diff(physical.connected_groups[0].return_path_v[:, 2])).sum()
    new_travel = np.abs(np.diff(regular.connected_groups[0].return_path_v[:, 2])).sum()
    assert new_travel < 0.5 * old_travel


@pytest.mark.parametrize("samples,axial_samples", [(24, 8), (48, 16)])
def test_ring_order_is_invariant_to_input_shuffle_cyclic_starts_rigid_rotation_and_mesh_density(samples, axial_samples):
    part = _part(samples, axial_samples)
    angle = 0.74
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    shift = np.array([0.21, -0.36, 0.4])
    part.mesh.vertices = part.mesh.vertices @ rotation.T + shift
    for name in ("vertex_normals", "face_normals"):
        normals = getattr(part.mesh, name)
        if normals is not None:
            setattr(part.mesh, name, normals @ rotation.T)
    for index, loop in enumerate(part.contour_lines):
        loop.v = loop.v @ rotation.T + shift
        _roll_loop(loop, 3 + index)
    part.contour_lines = [part.contour_lines[index] for index in [5, 1, 3, 0, 2, 4]]
    config = _config()
    config.b_0_direction = tuple(rotation[:, 2])
    grouped = topological_loop_grouping([part], config)[0]
    positions = [float((closed_curve_centroid(loop.v) - shift) @ rotation[:, 2]) for loop in grouped.groups[0].loops]
    reference = topological_loop_grouping([_part()], _config())[0]
    np.testing.assert_allclose(positions, _positions(reference), atol=2e-12, rtol=0)


def test_boundary_axis_is_invariant_to_collinear_boundary_resampling():
    part = _part()
    baseline = ring_support_axis(part.mesh)
    first, second = part.mesh.boundaries[0][:2]
    inserted = part.mesh.vertices[first] + np.linspace(0.1, 0.9, 9)[:, None] * (
        part.mesh.vertices[second] - part.mesh.vertices[first]
    )
    new_ids = np.arange(len(part.mesh.vertices), len(part.mesh.vertices) + len(inserted))
    part.mesh.vertices = np.vstack((part.mesh.vertices, inserted))
    part.mesh.boundaries[0] = np.insert(part.mesh.boundaries[0], 1, new_ids)
    actual = ring_support_axis(part.mesh)
    for left, right in zip(actual, baseline):
        np.testing.assert_allclose(left, right, atol=2e-14, rtol=0)


@pytest.mark.parametrize(
    "failure", ["ambiguous_topology", "nested", "mixed_current", "tilted", "multiple_turns", "overlap", "missing_axis"]
)
def test_unverified_or_nonparallel_groups_preserve_source_enumeration(failure):
    part = _part()
    indices = list(range(len(part.contour_lines)))
    signatures = [
        SimpleNamespace(loop_index=index, boundary_associated=True, ambiguous=False, boundary_signature=((0,), (1,)))
        for index in indices
    ]
    contains = np.zeros((len(indices), len(indices)), dtype=bool)
    axis = ring_support_axis(part.mesh)
    if failure == "ambiguous_topology":
        signatures[0].ambiguous = True
    elif failure == "nested":
        contains[0, 1] = True
    elif failure == "mixed_current":
        part.contour_lines[0] = reverse_contour_line(part.contour_lines[0])
    elif failure == "tilted":
        part.contour_lines[0].v[:, 2] += 0.3 * part.contour_lines[0].v[:, 0]
    elif failure == "multiple_turns":
        loop = part.contour_lines[0]
        loop.v = np.vstack((loop.v[:-1], loop.v))
    elif failure == "overlap":
        part.contour_lines[1].v[:, 2] = part.contour_lines[0].v[:, 2].mean()
    else:
        axis = None
    ordered, diagnostics = regular_ring_order(part, indices, signatures, contains, axis)
    assert ordered == indices
    assert diagnostics["method"] == "unchanged"


def test_physical_policy_preserves_historical_group_enumeration():
    part = topological_loop_grouping([_part()], _config("physical"))[0]
    assert part.groups[0].loop_indices == list(range(6))
    assert "regular_ring_traversal" not in part.topology_diagnostics


def _conical_part():
    mesh = normalize_mesh(build_cylinder_mesh(0.4, 0.2, 32, 12, 1, 0, 0, 0))
    mesh.vertices[:, :2] *= (1 + 0.8 * mesh.vertices[:, 2])[:, None]
    mesh = normalize_mesh(mesh)
    part = CoilPart(mesh=mesh, stream_function=mesh.vertices[:, 2].copy())
    part.potential_level_list = np.linspace(-0.13, 0.13, 6)
    part.contour_step = float(np.diff(part.potential_level_list)[0])
    return extract_contours([part], CoilGenConfig(min_point_loop_number=8))[0]


def test_verified_conical_ring_groups_use_one_automatic_opening_lane():
    config = _config()
    config.surface_is_cylinder_flag = False
    part = topological_loop_grouping([_conical_part()], config)[0]
    assert _coherent_axial_ring_group_indices(part, config) == (0,)
    state = _inter_group_cut_alignment_state(part, config)
    assert state is not None
    assert state.axial_ring_alignment
    calculate_group_centers([part])
    original = [loop.v.copy() for loop in part.contour_lines]
    interconnect_within_groups([part], config)
    openings = part.groups[0].opening_cuts_v
    assert len(openings) == 6
    # The taper changes radius, but every opening remains on one meridian.
    radial = np.array([points.mean(axis=0)[:2] for points in openings])
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    np.testing.assert_allclose(radial, np.broadcast_to(radial[0], radial.shape), atol=1e-12, rtol=0)
    for loop, before in zip(part.contour_lines, original):
        np.testing.assert_array_equal(loop.v, before)


@pytest.mark.parametrize("policy,mode", [("physical", "auto"), ("regular", "off"), ("regular", "disabled")])
def test_conical_seam_alignment_preserves_explicit_opt_out_and_physical_policy(policy, mode):
    config = _config(policy)
    config.surface_is_cylinder_flag = False
    config.inter_group_cut_alignment = mode
    part = topological_loop_grouping([_conical_part()], config)[0]
    assert _inter_group_cut_alignment_state(part, config) is None
    assert _coherent_axial_ring_group_indices(part, config) == ()
