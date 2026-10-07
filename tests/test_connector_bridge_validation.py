from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.connector_bridge_validation import _near_pairs, validate_connector_bridge


def _rectangle(x0, x1, y0=-0.1, y1=0.1):
    a, b, c, d = [x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]
    return np.asarray([[a, b, c], [a, c, d]])


def _plane(*patches):
    return SimpleNamespace(triangles=np.concatenate(patches or (_rectangle(-0.1, 0.1),)))


def _validate(original, start, end, candidate, **kwargs):
    options = dict(
        support_projector=_plane(),
        max_lift_m=0.003,
        support_normal=np.array([0.0, 0.0, 1.0]),
        physical_clearance_m=0.0026,
        target_clearance_m=0.006,
    )
    options.update(kwargs)
    return validate_connector_bridge(np.asarray(original), start, end, np.asarray(candidate), **options)


def _straight_route():
    return np.column_stack((np.linspace(-0.02, 0.02, 5), np.zeros((5, 2))))


def test_bridge_crosses_coplanar_triangle_boundary_and_preserves_inputs():
    original = _straight_route()
    candidate = np.array([original[1], [0, 0, 0.003], original[3]])
    before, before_candidate = original.copy(), candidate.copy()
    result = _validate(original, 1, 3, candidate, source_fractions=np.array([0, 0.5, 1]))
    assert result.accepted, result.reasons
    assert result.diagnostics["coplanar_support_triangle_count"] == 2
    assert result.diagnostics["unsupported_chord_count"] == 0
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(candidate, before_candidate)


def test_whole_chord_domain_check_catches_narrow_hole_between_valid_endpoints():
    original = np.array([[-0.02, 0, 0], [0.02, 0, 0]])
    # Both endpoints and the midpoint are supported. A sampling-only check can
    # miss this narrow, off-center gap; triangle-union coverage must detect it.
    support = _plane(_rectangle(-0.03, 0.004), _rectangle(0.00401, 0.03))
    result = _validate(original, 0, 1, original.copy(), support_projector=support)
    assert not result.accepted
    assert "support_domain_violation" in result.reasons


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_signed_height_is_bounded_on_entire_linear_chord(sign):
    original = _straight_route()
    candidate = np.array([original[1], [0, 0, sign * 0.003], original[3]])
    assert _validate(original, 1, 3, candidate, max_lift_m=sign * 0.003).accepted
    candidate[1, 2] *= 1.1
    assert "lift_limit_violation" in _validate(original, 1, 3, candidate, max_lift_m=sign * 0.003).reasons
    candidate[1, 2] *= -1
    assert "lift_limit_violation" in _validate(original, 1, 3, candidate, max_lift_m=sign * 0.003).reasons


def test_nonplanar_support_is_explicitly_unsupported():
    original = _straight_route()
    result = _validate(original, 1, 3, original[1:4], support_normal=np.array([0, 1, 1]))
    assert not result.accepted
    assert "unsupported_planar_patch" in result.reasons


def test_chord_cannot_pierce_another_support_face_above_valid_patch():
    original = _straight_route()
    candidate = np.array([original[1], [0, 0, 0.003], original[3]])
    other_face = np.array([[[0, -0.01, 0.001], [0, 0.01, 0.001], [0, 0, 0.004]]])
    result = _validate(original, 1, 3, candidate, support_projector=_plane(_rectangle(-0.1, 0.1), other_face))
    assert "support_surface_intersection" in result.reasons


def _route_with_existing_gap():
    return np.array(
        [
            [-0.03, -0.01, 0],
            [-0.03, 0.004, 0],
            [0.03, 0.004, 0],
            [0.03, 0, 0],
            [0.01, 0, 0],
            [-0.01, 0, 0],
            [-0.03, 0, 0],
        ]
    )


def test_existing_target_clearance_failure_does_not_reject_improving_bridge():
    original = _route_with_existing_gap()
    candidate = np.array([original[4], [0, 0, 0.003], original[5]])
    result = _validate(original, 4, 5, candidate)
    assert result.accepted, result.reasons
    assert result.diagnostics["baseline_below_clearance_count"] > 0
    assert result.diagnostics["candidate_below_clearance_count"] > 0
    assert result.diagnostics["clearance_regression_count"] == 0


def test_new_smaller_gap_to_any_surrounding_segment_is_rejected():
    original = _route_with_existing_gap()
    candidate = np.array([original[4], [0, 0.002, 0.001], original[5]])
    result = _validate(original, 4, 5, candidate)
    assert "clearance_regression" in result.reasons
    assert result.diagnostics["first_regressed_route_segment"] == 1
    assert result.diagnostics["maximum_gap_regression_m"] > 0.001


def test_clearance_comparison_is_capped_at_design_target():
    original = _route_with_existing_gap()
    original[1:3, 1] = 0.012
    candidate = np.array([original[4], [0, 0.004, 0.001], original[5]])
    result = _validate(original, 4, 5, candidate)
    assert result.accepted, result.reasons
    assert result.diagnostics["clearance_regression_count"] == 0


def _route_with_old_crossing():
    return np.array(
        [
            [-0.02, -0.01, 0],
            [0, -0.01, 0],
            [0, 0.01, 0],
            [-0.03, 0.03, 0],
            [-0.01, -0.005, 0],
            [0.01, -0.005, 0],
            [0.03, -0.005, 0],
        ]
    )


def test_existing_zero_gap_does_not_mask_new_crossing_elsewhere_on_same_segment():
    original = _route_with_old_crossing()
    candidate = np.array([original[4], [0, 0.005, 0], original[5]])
    result = _validate(original, 4, 5, candidate)
    assert "new_route_intersection" in result.reasons
    assert result.diagnostics["new_route_intersection_count"] > 0


def test_unchanged_existing_intersection_is_reported_as_inherited():
    original = _route_with_old_crossing()
    result = _validate(original, 4, 5, original[4:6].copy())
    assert result.accepted, result.reasons
    assert result.diagnostics["inherited_route_contact_count"] > 0
    assert result.diagnostics["new_route_intersection_count"] == 0


def test_existing_point_contact_does_not_mask_new_collinear_overlap():
    original = np.array(
        [[-0.02, -0.02, 0], [-0.02, 0, 0], [0.02, 0, 0], [0.02, -0.02, 0], [0, -0.01, 0], [0, 0.01, 0], [0.02, 0.02, 0]]
    )
    candidate = np.array([original[4], [0, 0, 0], [0.005, 0, 0], original[5]])
    result = _validate(original, 4, 5, candidate)
    assert "new_route_intersection" in result.reasons


def test_existing_collinear_overlap_is_distinguished_from_new_overlap():
    original = np.array(
        [[-0.02, -0.02, 0], [-0.02, 0, 0], [0.02, 0, 0], [0.02, -0.02, 0], [-0.01, 0, 0], [0.01, 0, 0], [0.02, 0.02, 0]]
    )
    result = _validate(original, 4, 5, original[4:6].copy())
    assert result.accepted, result.reasons
    assert result.diagnostics["inherited_route_contact_count"] > 0


def test_nonadjacent_candidate_crossing_is_checked_without_local_length_exclusion():
    original = np.array([[-0.002, 0, 0], [0.002, 0, 0]])
    candidate = np.array([original[0], [0.001, 0.001, 0], [-0.001, 0.001, 0], original[-1]])
    result = _validate(original, 0, 1, candidate)
    assert "candidate_self_intersection" in result.reasons


def test_adjacent_collinear_backtracking_is_not_a_permitted_shared_endpoint():
    original = np.array([[-0.02, 0, 0], [0.02, 0, 0]])
    candidate = np.array([original[0], [0, 0, 0], [-0.01, 0, 0], original[-1]])
    assert "candidate_self_intersection" in _validate(original, 0, 1, candidate).reasons


def test_candidate_may_not_backtrack_along_unchanged_adjacent_route():
    original = _straight_route()
    candidate = np.array([original[1], [-0.015, 0, 0], [0, 0, 0.003], original[3]])
    assert "new_route_intersection" in _validate(original, 1, 3, candidate).reasons


def test_collinear_resampling_preserves_gap_nonregression_decision():
    original = _route_with_existing_gap()
    candidate = np.array([original[4], [0, 0, 0.003], original[5]])
    refined = np.concatenate(
        [np.linspace(a, b, 9)[:-1] for a, b in zip(candidate[:-1], candidate[1:])] + [candidate[-1:]]
    )
    coarse, fine = _validate(original, 4, 5, candidate), _validate(original, 4, 5, refined)
    assert coarse.accepted and fine.accepted
    assert coarse.diagnostics["clearance_regression_count"] == fine.diagnostics["clearance_regression_count"]


def test_closed_route_includes_implicit_closing_segment():
    original = np.array([[-0.03, 0.003, 0], [-0.02, 0, 0], [0.02, 0, 0], [0.03, 0.003, 0]])
    candidate = np.array([original[1], [0, 0.003, 0], original[2]])
    result = _validate(original, 1, 2, candidate, closed=True)
    assert "new_route_intersection" in result.reasons


def test_bounded_broad_phase_returns_exact_segment_distances():
    first = np.array([[[0, 0, 0], [2, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=float)
    second = np.array([[[1, -1, 0], [1, 1, 0]], [[0, 1, 0], [2, 1, 0]]], dtype=float)
    found = {
        (int(i), int(j)): float(d) for ii, jj, dd in _near_pairs(first, second, 2.0) for i, j, d in zip(ii, jj, dd)
    }
    assert found == pytest.approx({(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1})


def test_endpoint_and_fraction_validation():
    original = _straight_route()
    candidate = original[1:4].copy()
    candidate[0, 0] += 0.001
    assert _validate(original, 1, 3, candidate).reasons == ("endpoint_mismatch",)
    with pytest.raises(ValueError, match="Source fractions"):
        _validate(original, 1, 3, original[1:4], source_fractions=np.array([0, 1.2, 1]))


def _creased_projector(*, hole=False):
    from coilgen.geometry import normalize_mesh
    from coilgen.manufacturing import _SurfaceProjector
    from coilgen.models import Mesh

    xs = [-0.03, 0.0, 0.004, 0.00401, 0.03] if hole else [-0.03, 0.0, 0.03]
    vertices = np.array([[x, y, -0.2 * abs(x)] for x in xs for y in [-0.03, 0.03]])
    faces = []
    for index in range(len(xs) - 1):
        if hole and index == 2:
            continue
        a = 2 * index
        faces.extend([[a, a + 2, a + 3], [a, a + 3, a + 1]])
    return _SurfaceProjector(normalize_mesh(Mesh(vertices, np.asarray(faces))))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_spatial_bridge_crosses_actual_crease_with_signed_support_distance(sign):
    support = _creased_projector()
    original = np.array([[-0.02, 0, -0.004], [0, 0, 0], [0.02, 0, -0.004]])
    candidate = original.copy()
    candidate[1, 2] = sign * 0.002
    result = _validate(original, 0, 2, candidate, support_projector=support, max_lift_m=sign * 0.003)
    assert result.accepted, result.reasons
    assert result.diagnostics["support_certificate"].startswith("adaptive_distance")
    assert result.diagnostics["maximum_support_distance_m"] <= 0.003 + 1e-8
    assert result.diagnostics["minimum_signed_height_m"] >= -1e-8
    assert result.diagnostics["support_evaluation_count"] < 100


def test_spatial_bridge_cannot_hide_excess_height_or_wrong_side_at_a_crease():
    support = _creased_projector()
    original = np.array([[-0.02, 0, -0.004], [0, 0, 0], [0.02, 0, -0.004]])
    candidate = original.copy()
    candidate[1, 2] = 0.0031
    assert "lift_limit_violation" in _validate(original, 0, 2, candidate, support_projector=support).reasons
    candidate[1, 2] = -0.002
    assert "lift_limit_violation" in _validate(original, 0, 2, candidate, support_projector=support).reasons


def test_spatial_boundary_certificate_detects_narrow_hole_between_samples():
    original = np.array([[-0.02, 0, -0.004], [0, 0, 0], [0.02, 0, -0.004]])
    candidate = original.copy()
    candidate[1, 2] = 0.002
    result = _validate(original, 0, 2, candidate, support_projector=_creased_projector(hole=True))
    assert not result.accepted
    assert "support_domain_violation" in result.reasons


def test_spatial_chord_triangle_intersection_cannot_be_missed_between_samples():
    support = _creased_projector()
    original = np.array([[-0.02, 0, -0.004], [0, 0, 0], [0.02, 0, -0.004]])
    # A single chord starts above the left facet and ends below the right one.
    # The exact intersection check is independent of adaptive sample placement.
    candidate = np.array([original[0], [-0.01, 0, 0.001], [0.01, 0, -0.003], original[-1]])
    result = _validate(original, 0, 2, candidate, support_projector=support)
    assert "support_surface_intersection" in result.reasons


def test_spatial_support_refinement_budget_never_silently_accepts():
    from coilgen.connector_bridge_validation import _SpatialSupport

    support = _SpatialSupport(_creased_projector(), 0.003, 1.0, 1e-8)
    segments = np.array([[[-0.02, 0, -0.004], [0, 0, 0.002]], [[0, 0, 0.002], [0.02, 0, -0.004]]])
    assert support.validate(segments, maximum_evaluations=1) == ["support_certificate_budget_exceeded"]


def test_spatial_support_acceptance_does_not_bypass_new_wire_collision_check():
    original = np.array(
        [
            [0, -0.01, 0.002],
            [0, 0.01, 0.002],
            [-0.03, 0.03, 0],
            [-0.02, 0, -0.004],
            [0, 0, 0],
            [0.02, 0, -0.004],
            [0.03, 0, -0.006],
        ]
    )
    candidate = np.array([original[3], [0, 0, 0.002], original[5]])
    result = _validate(original, 3, 5, candidate, support_projector=_creased_projector())
    assert "new_route_intersection" in result.reasons
    assert result.diagnostics["unsupported_chord_count"] == 0
    assert result.diagnostics["new_route_intersection_count"] > 0


def test_spatial_distance_bound_verifies_projector_nearest_candidate_heuristic():
    from coilgen.connector_bridge_validation import _SpatialSupport
    from coilgen.manufacturing import _SurfaceProjector
    from coilgen.models import Mesh

    # A large triangle is geometrically nearest but has distant vertices and
    # centroid. Nearby small triangles fill the projector's finite tree query.
    triangles = [[[-1.0, -1.0, 0], [3.0, -1.0, 0], [-1.0, 3.0, 0]]]
    for x in np.linspace(-0.001, 0.001, 12):
        triangles.append([[x, 0, 0.006], [x + 1e-5, 0, 0.006], [x, 1e-5, 0.006]])
    vertices = np.asarray(triangles).reshape(-1, 3)
    projector = _SurfaceProjector(Mesh(vertices, np.arange(len(vertices)).reshape(-1, 3)))
    point = np.array([0, 0, 0.002])
    proposed = projector.project_with_details(point[None])[0][0]
    assert np.linalg.norm(point - proposed) > 0.003
    exact = _SpatialSupport(projector, 0.003, 1.0, 1e-8).sample(point)
    assert exact[0] == pytest.approx(0.002)
    assert exact[2] == pytest.approx(0.002)
    assert exact[3] == 0


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("endpoint_depth", [0.0, 5e-9, 5e-7])
def test_bridge_attachment_at_neighboring_facet_obeys_signed_distance_tolerance(sign, endpoint_depth):
    support = _creased_projector()
    original = np.array([[-0.02, 0, -0.004], [-0.01, 0, -0.002], [0, 0, -sign * endpoint_depth]])
    candidate = original.copy()
    candidate[1, 2] += sign * 0.001
    result = _validate(
        original,
        0,
        2,
        candidate,
        support_projector=support,
        support_normal=support.face_normals[0],
        max_lift_m=sign * 0.003,
    )
    assert result.accepted == (endpoint_depth <= 1e-8), result.reasons
    if result.accepted:
        assert result.diagnostics["support_certificate"].startswith("adaptive_distance")
        assert result.diagnostics["minimum_signed_height_m"] >= -1e-8
