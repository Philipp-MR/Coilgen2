"""Fresh physical slices preserve geometry, current, owned targets and gaps."""

from dataclasses import replace
from hashlib import sha256
from itertools import product

import numpy as np
import pytest

from coilgen.application.field_analysis import FieldDataset, source_catalog
from coilgen.application.field_sampling import sample_slices, _Interpolator, _region_level
from coilgen.application.gradient_studies import RigidTransform
from coilgen.application.service import RunRecord
from coilgen.physics import biot_savart_wire_path
from coilgen.progress import DesignCancelled
from coilgen.target_regions import TargetRegion


def cube_points(size=0.1):
    return np.asarray(list(product((-size, size), repeat=3)))


def wire_record(tmp_path, current=-2.0):
    path = np.asarray([[-0.1, 0, 0], [0.1, 0, 0]])
    points = cube_points(0.08)
    actual = current * biot_savart_wire_path(path, points)
    record = RunRecord(
        tmp_path,
        {
            "wire_fields": {
                "routed": {
                    "available": True,
                    "current_A": 99,
                    "at_stream_current": {"available": True, "current_A": current},
                }
            }
        },
        {"evaluated_winding_0_points_m": path, "target_points_m": points, "routed_field_at_stream_current_T": actual},
    )
    return record, next(source for source in source_catalog(record) if source.key == "routed_at_stream_current")


@pytest.mark.parametrize("current", [-2.0, 3.0])
def test_finite_wire_slice_matches_analytic_signed_current_and_masks_wire(tmp_path, current):
    record, source = wire_record(tmp_path, current)
    result = sample_slices(record, source, representation="direct", resolution=5, size_m=(0.2, 0.1))
    slice_ = result.slices[0]
    assert result.representation == "direct_wire"
    assert not result.provenance["analysis_current_refitted"]
    assert slice_.available_mask.sum() == 20
    assert not np.any(slice_.available_mask[2])
    assert np.all(np.isnan(slice_.field_T[2]))
    # Finite straight x-directed segment: Bz(0,y,0) = mu0 I/(4pi)
    # * 2a / (y sqrt(a²+y²)). Its sign includes the stored signed current.
    y, a = 0.05, 0.1
    expected = 1e-7 * current * 2 * a / (y * np.sqrt(a * a + y * y))
    assert slice_.field_T[-1, 2, 2] == pytest.approx(expected, rel=2e-15)
    assert np.all(slice_.field_T[-1, 2, :2] == 0)
    assert not np.any(slice_.target_valid_mask)
    assert result.target_representation == "unavailable"
    # Fresh physics is available outside the original sampling hull.
    extended = sample_slices(
        record, source, representation="direct", resolution=3, centre_m=(0.3, 0, 0), offsets_m=(0.1,), size_m=(0.1, 0.1)
    ).slices[0]
    assert extended.available_mask.all() and not extended.roi_mask.any()
    with pytest.raises(ValueError):
        slice_.field_T.setflags(write=True)


def test_saved_sheet_replays_existing_kernel_and_masks_surface(tmp_path):
    from coilgen.config import CoilGenConfig
    from coilgen.models import TargetField
    from coilgen.physics import calculate_sensitivity_matrix
    from coilgen.application.field_sampling import _sheet_source

    vertices = np.asarray([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0, 0.1, 0]])
    sf = np.asarray([0.0, 1.0, -0.5])
    points = cube_points(0.05)
    base = RunRecord(tmp_path, {}, {"vertices_m": vertices, "faces": np.asarray([[0, 1, 2]]), "stream_function_A": sf})
    config = CoilGenConfig(progress_flag=False)
    part, stream = _sheet_source(base, config, None)
    target = TargetField(
        points, np.zeros_like(points), np.ones(len(points)), np.zeros(len(points), bool), np.zeros(len(points), int)
    )
    calculate_sensitivity_matrix([part], target, config)
    fields = np.einsum("ctv,v->tc", part.sensitivity_matrix, stream)
    source = FieldDataset("sheet", "Saved sheet", points, fields, provenance={"physical_representation": "sheet"})
    result = sample_slices(
        base, source, representation="direct", offsets_m=(0.0, 0.05), size_m=(0.2, 0.2), resolution=5
    )
    assert result.representation == "direct_sheet"
    assert not result.slices[0].available_mask[2, 2]
    assert np.all(np.isnan(result.slices[0].field_T[2, 2]))
    assert result.slices[0].support_outlines_m
    positive = result.slices[1]
    assert positive.available_mask.all()
    selected = np.asarray([positive.coordinates_m[2, 2]])
    target = TargetField(selected, np.zeros_like(selected), np.ones(1), np.zeros(1, bool), np.zeros(1, int))
    calculate_sensitivity_matrix([part], target, config)
    np.testing.assert_allclose(
        positive.field_T[2, 2], np.einsum("ctv,v->tc", part.sensitivity_matrix, sf)[0], rtol=1e-14
    )


def test_interpolation_never_extrapolates_or_fills_off_plane(tmp_path):
    points = np.asarray([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0]])
    field = np.column_stack((points[:, 0], points[:, 1], points[:, 0] + points[:, 1]))
    source = FieldDataset("map", "Imported samples", points, field, field * 2)
    result = sample_slices(
        RunRecord(tmp_path, {}, {}),
        source,
        representation="interpolation",
        offsets_m=(0, 0.01),
        size_m=(0.4, 0.4),
        resolution=5,
    )
    assert result.representation == "display_interpolation"
    assert result.target_representation == "display_interpolation"
    assert result.roi_representation == "sampling_hull"
    assert result.slices[0].available_mask.sum() == 9
    assert not result.slices[1].available_mask.any()
    assert not result.slices[1].target_valid_mask.any()
    assert np.isnan(result.slices[0].field_T[0, 0]).all()
    np.testing.assert_allclose(result.slices[0].field_T[2, 3], [0.1, 0, 0.1], atol=1e-17)
    assert not result.provenance["independent_validation_claim"]
    assert not result.provenance["interpolation_extrapolation"]


def test_gap_screen_keeps_exact_observations_but_not_large_unsampled_bridge():
    points = np.asarray([[0, 0, 0], [0.01, 0, 0], [0.5, 0, 0], [0.51, 0, 0]])
    source = _Interpolator(points, points * 2)
    values, valid = source.sample(np.asarray([[0.005, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.005, 0.001, 0]]))
    np.testing.assert_array_equal(valid, [True, False, True, False])
    np.testing.assert_allclose(values[0], [0.01, 0, 0])
    assert np.isnan(values[1]).all()
    duplicate = _Interpolator(np.vstack((points, points)), np.vstack((points * 2, points * 2)))
    assert duplicate.rank == 1
    with pytest.raises(ValueError, match="Duplicate"):
        _Interpolator(np.vstack((points, points)), np.vstack((points * 2, points * 3)))


@pytest.mark.parametrize("plane,normal", [("xy", 2), ("xz", 1), ("yz", 0)])
def test_slice_offsets_are_absolute_world_coordinates(tmp_path, plane, normal):
    points = cube_points(1)
    source = FieldDataset("map", "Map", points, points)
    result = sample_slices(
        RunRecord(tmp_path, {}, {}),
        source,
        plane=plane,
        centre_m=(0.2, 0.3, 0.4),
        offsets_m=(0.5,),
        size_m=(0.2, 0.3),
        resolution=3,
    )
    np.testing.assert_array_equal(result.slices[0].coordinates_m[:, :, normal], np.full((3, 3), 0.5))


def test_rotated_registered_roi_uses_exact_owned_geometry_and_outline(tmp_path):
    region = TargetRegion("box", size_m=(0.2, 0.1, 0.3), rotation_angle_rad=np.pi / 4, rotation_axis=(0, 0, 1))
    geometry = {
        "region": region.to_dict(),
        "registration": RigidTransform(translation_m=(0.2, 0.1, 0)).to_dict(),
        "sample_transform": RigidTransform(translation_m=(0, 0.1, 0)).to_dict(),
    }
    points = cube_points(1)
    source = FieldDataset("map", "Map", points, points, points, target_geometry=geometry)
    result = sample_slices(RunRecord(tmp_path, {}, {}), source, centre_m=(0.2, 0.2, 0), size_m=(0.4, 0.4), resolution=9)
    selected = result.slices[0]
    assert result.roi_representation == "owned_geometry"
    level = _region_level(selected.coordinates_m.reshape(-1, 3), geometry).reshape(9, 9)
    np.testing.assert_array_equal(selected.roi_mask, level <= 64 * np.finfo(float).eps)
    assert selected.roi_mask[4, 4] and not selected.roi_mask[0, 0]
    assert selected.outlines_m
    assert all(np.max(np.abs(_region_level(line, geometry))) < 1e-12 for line in selected.outlines_m)
    np.testing.assert_array_equal(selected.target_valid_mask, selected.roi_mask)


def test_native_target_replays_saved_normalization_and_excludes_suppressed_points(tmp_path):
    points = cube_points(0.05)
    target = np.column_stack((np.zeros((8, 2)), (points[:, 0] + 2 * points[:, 2]) * 0.002))
    suppressed_points = np.asarray([[0.5, 0, 0]])
    arrays = {
        "target_points_m": np.vstack((points, suppressed_points)),
        "target_field_T": np.vstack((target, np.zeros((1, 3)))),
        "target_suppressed": np.r_[np.zeros(8, bool), True],
    }
    metadata = {
        "target_recipe": {
            "field_shape_function": "x+2*z",
            "target_field_units": "T",
            "normalization_scale": 0.002,
            "field_component": "Bz",
            "coordinate_frame": "world",
        },
        "config": {"target_region_radius": 0.1},
    }
    record = RunRecord(tmp_path, metadata, arrays)
    result = sample_slices(
        record, "target", representation="direct", offsets_m=(0.03,), resolution=5, size_m=(0.3, 0.3)
    )
    selected = result.slices[0]
    assert result.representation == "owned_analytic_target"
    assert result.target_representation == "owned_analytic"
    assert selected.field_T[2, 2, 2] == pytest.approx(0.00012)
    assert not selected.target_valid_mask[0, 0]  # Out-of-ROI suppression cannot become analytic desired truth.
    assert selected.available_mask.all()
    wrong = RunRecord(
        tmp_path, {**metadata, "target_recipe": {**metadata["target_recipe"], "normalization_scale": 0.003}}, arrays
    )
    with pytest.raises(ValueError, match="primary desired"):
        sample_slices(wrong, "target", resolution=3)


def test_unknown_old_native_normalization_stays_display_interpolation(tmp_path):
    points = cube_points(0.05)
    fields = np.column_stack((np.zeros((8, 2)), points[:, 0] * 0.02))
    record = RunRecord(
        tmp_path, {"config": {"target_region_radius": 0.1}}, {"target_points_m": points, "target_field_T": fields}
    )
    result = sample_slices(
        record,
        "target",
        request={"config": {"field_shape_function": "x", "target_gradient_normalization": "extrema"}},
        resolution=3,
    )
    assert result.representation == result.target_representation == "display_interpolation"
    with pytest.raises(ValueError, match="physical reevaluation"):
        sample_slices(record, "target", representation="direct", resolution=3)


def test_named_partial_path_pose_and_vector_target_registration_are_replayed(tmp_path):
    from coilgen.application.research_requirements import (
        DesiredDataset,
        PhysicalScenario,
        RegionalRequirement,
        ResearchRequirements,
    )

    rz = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    rx = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
    desired = DesiredDataset.analytic(TargetRegion("box", size_m=(0.1, 0.1, 0.1)), ("x", "2*y", "3*z"), resolution=2)
    desired = replace(desired, registration=RigidTransform(rz, (0, 0.2, 0)))
    pose = PhysicalScenario(
        "Moved",
        "validation",
        support_transform=RigidTransform(rz, (0.1, 0, 0)),
        sample_transform=RigidTransform(rx, (0.05, 0, 0.1)),
        support_scale_xyz=(1.2, 0.8, 1),
    )
    spec = ResearchRequirements((RegionalRequirement("ROI", desired, rms_limit_T=1),), (pose,))
    points, values = desired.world_values()
    points = points @ np.asarray(rx).T + pose.sample_transform.translation_m
    values = values @ np.asarray(rx).T
    path = np.asarray([[-0.1, 0, 0], [0.1, 0, 0]])
    moved = (path * np.asarray(pose.support_scale_xyz)) @ np.asarray(rz).T + pose.support_transform.translation_m
    current = -3.0
    actual = current * biot_savart_wire_path(moved, points)
    row = {
        "available": True,
        "array_prefix": "requirements_0000",
        "region": "ROI",
        "scenario": "Moved",
        "sampling": "design",
        "scope": "partial_electrical_paths",
        "currents_A": [current],
        "dataset_sha256": desired.digest,
        "path_sha256": [sha256(np.asarray(path, dtype="<f8").tobytes()).hexdigest()],
    }
    record = RunRecord(
        tmp_path,
        {"requirements_review": {"rows": [row]}},
        {
            "review_electrical_0_points_m": path,
            "requirements_0000_points_m": points,
            "requirements_0000_field_T": actual,
            "requirements_0000_target_T": values,
        },
    )
    request = {"requirements": spec.to_dict()}
    source = source_catalog(record, request=request)[0]
    result = sample_slices(
        record,
        source,
        request=request,
        representation="direct",
        centre_m=(0.05, 0, 0.3),
        offsets_m=(0.3,),
        size_m=(0.08, 0.08),
        resolution=3,
    )
    selected = result.slices[0]
    assert result.representation == "direct_wire"
    np.testing.assert_allclose(
        selected.field_T.reshape(-1, 3),
        current * biot_savart_wire_path(moved, selected.coordinates_m.reshape(-1, 3)),
        rtol=1e-14,
    )
    inverse_sample = (selected.coordinates_m.reshape(-1, 3) - pose.sample_transform.translation_m) @ np.asarray(rx)
    original = (inverse_sample - desired.registration.translation_m) @ np.asarray(rz)
    expected = (
        np.column_stack((original[:, 0], 2 * original[:, 1], 3 * original[:, 2])) @ np.asarray(rz).T @ np.asarray(rx).T
    )
    np.testing.assert_allclose(selected.target_T.reshape(-1, 3), expected, atol=1e-16)
    assert selected.target_valid_mask.all() and result.target_representation == "owned_analytic"
    bad_source = replace(
        source,
        provenance={**dict(source.provenance), "requirements_row": {**row, "scenario": "Wrong"}},
        target_geometry=source.to_dict()["target_geometry"],
    )
    with pytest.raises(ValueError, match="unknown original"):
        sample_slices(record, bad_source, request=request, resolution=3)


def test_mismatched_saved_map_has_explicit_auto_fallback_and_direct_refusal(tmp_path):
    record, source = wire_record(tmp_path)
    source = replace(source, field_T=source.field_T * 2)
    result = sample_slices(record, source, resolution=3)
    assert result.representation == "display_interpolation"
    assert "does not reproduce" in result.provenance["fallback_reason"]
    with pytest.raises(ValueError, match="does not reproduce"):
        sample_slices(record, source, representation="direct", resolution=3)


def test_cancellation_progress_and_explicit_work_budgets(tmp_path):
    record, source = wire_record(tmp_path)
    calls = []
    result = sample_slices(
        record,
        source,
        resolution=17,
        offsets_m=(0.01, 0.02),
        progress=lambda percent, message: calls.append((percent, message)),
    )
    assert result.representation == "direct_wire"
    assert calls[0][0] == 0 and calls[-1][0] == 100
    determinate = [percent for percent, message in calls if percent is not None]
    assert len(determinate) > 5 and determinate == sorted(determinate)
    assert any(percent is None and "Reconstruct saved" in message for percent, message in calls)
    with pytest.raises(DesignCancelled):
        sample_slices(record, source, cancel_check=lambda: True)
    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if checks == 7:
            raise DesignCancelled("Stopped during slice work")

    with pytest.raises(DesignCancelled, match="during slice"):
        sample_slices(record, source, resolution=64, cancel_check=check)
    with pytest.raises(ValueError, match="524288"):
        sample_slices(record, source, resolution=256, offsets_m=np.arange(9) / 100)
    with pytest.raises(ValueError, match="distinct"):
        sample_slices(record, source, offsets_m=(0, 0))


def test_sheet_slice_exposes_kernel_progress_before_target_chunk_finishes(monkeypatch):
    from types import SimpleNamespace
    from coilgen.application import field_sampling
    from coilgen.config import CoilGenConfig
    from coilgen.progress import ProgressReporter
    import coilgen.physics as physics

    points = np.zeros((16, 3))
    stream = np.arange(100, dtype=float)
    part = SimpleNamespace(sensitivity_matrix=None)
    calls = []

    def sensitivity(parts, target, config):
        task = ProgressReporter(enabled=False).task("Sheet nodes", len(stream))
        for index in range(len(stream)):
            task.advance()
            if 10 < index < len(stream) - 1:
                assert any(value is not None and 0 < value < 1 for value in calls)
        parts[0].sensitivity_matrix = np.ones((3, len(target.coords), len(stream)))

    monkeypatch.setattr(field_sampling, "_sheet_mask", lambda points, part, check: np.ones(len(points), bool))
    monkeypatch.setattr(physics, "calculate_sensitivity_matrix", sensitivity)
    values, valid = field_sampling._physical(
        points, "sheet", (part, stream), CoilGenConfig(progress_flag=False), None, calls.append
    )
    np.testing.assert_array_equal(values, np.full((len(points), 3), stream.sum()))
    assert valid.all() and part.sensitivity_matrix is None
    determinate = [value for value in calls if value is not None]
    assert len(determinate) > 10 and determinate == sorted(determinate)
    assert determinate[-1] == 1


def test_artifact_integrity_failure_is_not_display_interpolation(tmp_path, monkeypatch):
    from coilgen.application.evidence import EvidenceIntegrityError
    import coilgen.application.evidence as evidence

    record, source = wire_record(tmp_path)
    source = replace(source, key="contour_at_stream_current")

    def changed(*args):
        raise EvidenceIntegrityError("Changed saved checkpoint")

    monkeypatch.setattr(evidence, "read_verified_artifact", changed)
    with pytest.raises(EvidenceIntegrityError, match="Changed saved checkpoint"):
        sample_slices(record, source, resolution=3)


def test_singular_original_sample_does_not_discard_physics_or_fill_false_zero(tmp_path):
    record, source = wire_record(tmp_path)
    points = np.vstack((source.points_m, [0, 0, 0]))
    values = np.vstack(
        (source.field_T, [0, 0, 0])
    )  # Existing native wire kernel regularizes its singular value to zero.
    source = replace(source, points_m=points, field_T=values)
    physical = sample_slices(record, source, resolution=5, size_m=(0.2, 0.1))
    assert physical.representation == "direct_wire"
    assert not physical.slices[0].available_mask[2, 2]
    assert np.isnan(physical.slices[0].field_T[2, 2]).all()
    interpolated = sample_slices(record, source, resolution=5, representation="interpolation", size_m=(0.2, 0.1))
    assert not interpolated.slices[0].available_mask[2, 2]
    assert np.isnan(interpolated.slices[0].field_T[2, 2]).all()
    mismatch = replace(source, field_T=source.field_T * 2)
    fallback = sample_slices(record, mismatch, resolution=5, size_m=(0.2, 0.1))
    assert fallback.representation == "display_interpolation"
    assert not fallback.slices[0].available_mask[2, 2]


def test_real_completed_checkpoint_direct_sheet_contour_and_owned_request(tmp_path):
    from coilgen.application import DesignService, ProjectDocument
    from coilgen.config import CoilGenConfig

    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 8, 4, 1, 0, 0, 0),
        field_shape_function="x",
        target_region_geometry={"shape": "box", "size_m": [0.02, 0.02, 0.02]},
        target_region_resolution=2,
        symmetry_reduction="off",
        gauss_order=3,
        progress_flag=False,
        levels=4,
        min_point_loop_number=3,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
    )
    DesignService.run(ProjectDocument.from_config(config), tmp_path / "run", validation_resolution=2)
    record = DesignService.load_run(tmp_path / "run")
    before = (record.directory / "result.json").read_bytes()
    sources = {source.key: source for source in source_catalog(record)}
    for key, representation in (("sheet", "direct_sheet"), ("contour_at_stream_current", "direct_wire")):
        assert key in sources
        sampled = sample_slices(record, sources[key], representation="direct", resolution=5, size_m=(0.02, 0.02))
        assert sampled.representation == representation and sampled.target_representation == "owned_analytic"
        assert sampled.slices[0].available_mask.all()
        assert sampled.slices[0].target_valid_mask.all()
    assert (record.directory / "result.json").read_bytes() == before
    with pytest.raises(ValueError, match="original request"):
        sample_slices(record, sources["sheet"], request={"config": {"field_shape_function": "z"}}, resolution=3)


def test_real_owned_enclosure_silhouette_and_containment(tmp_path):
    from coilgen.application import DesignService, ProjectDocument
    from coilgen.config import CoilGenConfig
    from coilgen.models import Mesh

    mesh = Mesh(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * 0.04,
        np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.2, 0.12, 8, 4, 1, 0, 0, 0),
        target_region_geometry={"shape": "enclosing_mesh"},
        target_region_resolution=4,
        field_shape_function="y",
        target_gradient_strength=0.02,
        target_gradient_normalization="linear_fit",
        target_gradient_axis="y",
        gauss_order=1,
        symmetry_reduction="off",
        progress_flag=False,
    )
    project = ProjectDocument.new(config=config).with_mesh(mesh)
    DesignService.run(project, tmp_path / "enclosure", pipeline_stage="sheet", validation_resolution=4)
    record = DesignService.load_run(tmp_path / "enclosure")
    result = sample_slices(
        record,
        "sheet",
        representation="direct",
        offsets_m=(0.01,),
        centre_m=(0.02, 0.02, 0.01),
        size_m=(0.05, 0.05),
        resolution=11,
    )
    selected = result.slices[0]
    assert result.roi_representation == "owned_geometry"
    assert selected.roi_mask.any() and not selected.roi_mask.all()
    assert len(selected.outlines_m) == 3
    assert selected.roi_mask[3, 3] and not selected.roi_mask[-1, -1]
    np.testing.assert_array_equal(selected.target_valid_mask, selected.roi_mask)
    boundary = np.vstack(selected.outlines_m)
    assert np.max(np.abs(boundary[:, 2] - 0.01)) < 1e-16
    assert np.max(boundary[:, 0]) == pytest.approx(0.03)


def test_surface_primitive_has_labelled_expression_continuation_but_no_acceptance(tmp_path):
    points = cube_points(0.1)
    target = np.column_stack((np.zeros((8, 2)), points[:, 0]))
    record = RunRecord(
        tmp_path,
        {
            "config": {"target_region_geometry": {"shape": "box", "size_m": [0.2] * 3, "sampling": "surface"}},
            "target_recipe": {
                "field_shape_function": "x",
                "target_field_units": "T",
                "normalization_scale": 1,
                "field_component": "Bz",
                "coordinate_frame": "world",
            },
        },
        {"target_points_m": points, "target_field_T": target},
    )
    result = sample_slices(record, "target", representation="direct", size_m=(0.1, 0.1), resolution=3)
    assert result.slices[0].target_valid_mask.all()
    assert "original constraints sampled on surface" in result.provenance["target_domain_scope"]
    assert not result.provenance["independent_validation_claim"]


def test_open_surface_mesh_never_claims_an_enclosed_volume(tmp_path, monkeypatch):
    import coilgen.application.field_sampling as sampling
    from coilgen.models import Mesh

    points = cube_points(0.1)
    values = np.column_stack((np.zeros((8, 2)), points[:, 0]))
    geometry = {"region": {"shape": "enclosing_mesh", "sampling": "surface"}}
    source = FieldDataset("target", "Surface target", points, values, values, target_geometry=geometry)
    mesh = Mesh(np.asarray([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0, 0.1, 0]]), np.asarray([[0, 1, 2]]))
    monkeypatch.setattr(sampling, "_owned_enclosure", lambda *args: mesh)

    def impossible(*args):
        pytest.fail("An open authored surface must not invoke closed-volume containment.")

    monkeypatch.setattr(sampling, "_enclosure_mask", impossible)
    result = sample_slices(
        RunRecord(tmp_path, {}, {}),
        source,
        representation="direct",
        resolution=5,
        size_m=(0.2, 0.2),
        request={"config": {"field_shape_function": "x", "target_gradient_normalization": "none"}},
    )
    assert result.provenance["target_domain_scope"] == "owned boundary samples only"
    assert result.slices[0].roi_mask[2, 2] and not result.slices[0].roi_mask[-1, -1]
    assert result.slices[0].outlines_m


def test_independent_desired_pair_is_checked_at_its_own_points(tmp_path):
    points = cube_points(0.1)
    target = np.column_stack((np.zeros((8, 2)), points[:, 0]))
    metadata = {
        "target_recipe": {
            "field_shape_function": "x",
            "target_field_units": "T",
            "normalization_scale": 1,
            "field_component": "Bz",
            "coordinate_frame": "world",
        }
    }
    record = RunRecord(tmp_path, metadata, {"target_points_m": points, "target_field_T": target})
    validation = FieldDataset(
        "validation", "Independent", points * 0.8, target, target, provenance={"points_key": "validation_points_m"}
    )
    with pytest.raises(ValueError, match="primary desired samples"):
        sample_slices(record, validation, resolution=3)
