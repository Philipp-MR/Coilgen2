"""Saved-source pairing and numerical diagnostics for the analysis application layer."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen.application.field_analysis import (
    FieldDataset,
    analyze_harmonics,
    field_analysis_available,
    metric_map,
    source_catalog,
)
from coilgen.application.service import RunRecord
from coilgen.spherical_harmonics import fit_spherical_harmonics_many, solid_spherical_harmonic_basis


def dataset(actual=(0.0, 2.0, 4.0), desired=(0.0, 1.0, 2.0), weights=None):
    points = np.column_stack((np.arange(len(actual)), np.zeros((len(actual), 2))))
    return FieldDataset(
        "example",
        "Saved test field",
        points,
        np.column_stack((np.zeros((len(actual), 2)), actual)),
        np.column_stack((np.zeros((len(actual), 2)), desired)),
        weights,
    )


def test_owned_finite_vector_dataset_and_json_exports():
    points, actual = np.zeros((2, 3)), np.ones((2, 3))
    source = FieldDataset("field", "Computed", points, actual, provenance={"nested": {"flag": True}})
    actual[:] = 20
    assert np.all(source.field_T == 1)
    with pytest.raises(ValueError):
        source.field_T[0, 0] = 5
    with pytest.raises(TypeError):
        source.provenance["nested"]["flag"] = False
    assert source.to_dict()["provenance"]["nested"]["flag"]
    assert metric_map(source).to_dict()["summary"]["rms"] == 1
    for bad in (actual.astype(complex), np.full((2, 3), np.nan), actual[:, :2]):
        with pytest.raises(ValueError):
            FieldDataset("field", "Computed", points, bad)
    with pytest.raises(ValueError, match="same saved sampling"):
        FieldDataset("field", "Computed", points, actual, np.zeros((3, 3)))
    geometry = {"region": {"shape": "box", "size_m": [0.1, 0.2, 0.3]}}
    owned = replace(source, target_geometry=geometry)
    copied = replace(owned, label="Copied owned snapshot")
    assert copied.provenance["nested"]["flag"]
    assert copied.target_geometry["region"]["size_m"] == (0.1, 0.2, 0.3)


def test_saved_catalog_pairs_validation_with_its_own_target_and_excludes_unavailable_maps(tmp_path):
    design, independent = np.zeros((3, 3)), np.full((4, 3), 0.1)
    metadata = {
        "config": {"target_region_radius": 0.2, "b_0_direction": [1, 0, 0]},
        "validation": {"available": True},
        "wire_fields": {
            "contour": {"available": True, "current_A": 3, "at_stream_current": {"available": True, "current_A": 2}},
            "routed": {"available": False},
        },
    }
    arrays = {
        "target_points_m": design,
        "target_field_T": np.ones_like(design),
        "sheet_field_T": np.ones_like(design) * 2,
        "target_weights": np.arange(3.0),
        "validation_points_m": independent,
        "validation_field_T": np.ones_like(independent) * 3,
        "validation_target_T": np.ones_like(independent) * 4,
        "contour_field_T": np.ones_like(design) * 5,
        "contour_field_at_stream_current_T": np.ones_like(design) * 6,
        "routed_field_T": np.ones_like(design) * 99,
    }
    record = RunRecord(tmp_path, metadata, arrays)
    sources = {item.key: item for item in source_catalog(record)}
    assert field_analysis_available(record)
    assert set(sources) == {"target", "sheet", "validation", "contour", "contour_at_stream_current"}
    np.testing.assert_array_equal(sources["validation"].target_T, arrays["validation_target_T"])
    assert sources["validation"].weights is None
    assert sources["contour_at_stream_current"].provenance["current_policy"] == "original"
    assert sources["contour"].provenance["current_A"] == 3
    assert sources["sheet"].provenance["b0_direction"] == (1, 0, 0)
    assert sources["sheet"].target_geometry["region"]["radius_m"] == 0.2
    assert not field_analysis_available(RunRecord(tmp_path, {}, {"target_points_m": design, "target_field_T": design}))
    assert not field_analysis_available(
        RunRecord(tmp_path, {}, {"target_points_m": design, "sheet_field_T": np.ones((3, 2))})
    )
    from types import SimpleNamespace

    assert not field_analysis_available(SimpleNamespace(metadata={}, arrays=arrays))


def test_missing_desired_and_mismatched_fabricated_points_never_reuse_design_target(tmp_path):
    points, field = np.zeros((3, 3)), np.ones((3, 3))
    record = RunRecord(
        tmp_path,
        {},
        {
            "target_points_m": points,
            "target_field_T": field,
            "fabricated_points_m": points + 1,
            "fabricated_field_T": field,
        },
    )
    realized = next(item for item in source_catalog(record) if item.key == "fabricated")
    assert realized.target_T is None
    assert metric_map(realized, metric="field").summary["rms"] == 1
    with pytest.raises(ValueError, match="no paired desired"):
        metric_map(realized, metric="signed_residual")


def test_named_pose_catalog_retains_source_scope_and_authored_volume(tmp_path):
    from coilgen.application.gradient_studies import RigidTransform
    from coilgen.application.research_requirements import (
        DesiredDataset,
        PhysicalScenario,
        RegionalRequirement,
        ResearchRequirements,
    )
    from coilgen.target_regions import TargetRegion

    desired = DesiredDataset.analytic(TargetRegion("box", size_m=(0.01, 0.02, 0.03)), ("0", "0", "x"), resolution=2)
    desired = replace(desired, registration=RigidTransform(translation_m=(1, 0, 0)))
    pose = PhysicalScenario("Held out", "validation", sample_transform=RigidTransform(translation_m=(0, 0.1, 0)))
    spec = ResearchRequirements((RegionalRequirement("ROI", desired, rms_limit_T=1e-4),), (pose,))
    points, target = desired.world_values()
    row = {
        "available": True,
        "array_prefix": "requirements_0000",
        "region": "ROI",
        "scenario": "Held out",
        "sampling": "design",
        "scope": "partial_electrical_paths",
        "currents_A": [2.0, -1.0],
        "passed": None,
    }
    record = RunRecord(
        tmp_path,
        {"requirements_review": {"rows": [row, {**row, "available": False, "array_prefix": "requirements_0001"}]}},
        {
            "requirements_0000_points_m": points,
            "requirements_0000_field_T": target * 2,
            "requirements_0000_target_T": target,
        },
    )
    sources = source_catalog(record, request={"requirements": spec.to_dict()})
    assert len(sources) == 1 and field_analysis_available(record)
    source = sources[0]
    assert source.scope == "partial_electrical_paths"
    assert source.provenance["requirements_row"]["passed"] is None
    assert not source.provenance["acceptance_claim"]
    assert source.target_geometry["registration"]["translation_m"] == (1, 0, 0)
    assert source.target_geometry["sample_transform"]["translation_m"] == (0, 0.1, 0)


def test_imported_desired_maps_do_not_inherit_unused_analytic_roi(tmp_path):
    arrays = {"target_points_m": np.zeros((3, 3)), "target_field_T": np.ones((3, 3))}
    record = RunRecord(
        tmp_path,
        {
            "config": {
                "target_field_definition_file": "map.npz",
                "target_region_geometry": {"shape": "sphere", "radius_m": 0.1},
            }
        },
        arrays,
    )
    assert source_catalog(record)[0].target_geometry is None


def test_native_scope_counts_include_suppression_without_changing_map_statistics(tmp_path):
    source = dataset(actual=(1, 2, 10), desired=(0, 0, 0))
    record = RunRecord(
        tmp_path,
        {},
        {
            "target_points_m": source.points_m,
            "target_field_T": source.target_T,
            "sheet_field_T": source.field_T,
            "target_suppressed": np.array([False, False, True]),
        },
    )
    saved = next(item for item in source_catalog(record) if item.key == "sheet")
    assert saved.provenance["primary_sample_count"] == 2
    assert saved.provenance["suppression_sample_count"] == 1
    assert saved.provenance["includes_suppression"]
    mapped = metric_map(saved, metric="absolute_error")
    assert mapped.summary["rms"] == pytest.approx(np.sqrt(105 / 3))
    assert mapped.summary["sample_scope"] == "All saved primary and suppression samples"
    harmonic = analyze_harmonics(saved, max_order=0)
    assert harmonic["sample_count"] == 3 and harmonic["suppression_sample_count"] == 1
    assert any("span current-carrying material" in message for message in harmonic["diagnostics"])
    np.testing.assert_array_equal(saved.points_m, source.points_m)


def test_global_normalization_is_finite_at_individual_target_zeros_and_statistics_are_explicit():
    source = dataset()
    metric = metric_map(source, metric="signed_residual", normalization="target_rms")
    np.testing.assert_allclose(metric.values, np.array([0, 1, 2]) / np.sqrt(5 / 3) * 100)
    assert np.all(np.isfinite(metric.values)) and metric.unit == "%"
    assert metric.summary["error_rms_T"] == pytest.approx(np.sqrt(5 / 3))
    assert metric.summary["rms"] == pytest.approx(100)
    assert metric.summary["relative_error_rms"] == pytest.approx(1)
    assert metric.summary["p95"] == pytest.approx(1.9 / np.sqrt(5 / 3) * 100)
    assert "not a volume integral" in metric.sampling_convention
    assert not metric.summary["acceptance_claim"]
    zero = dataset(desired=(0, 0, 0))
    assert metric_map(zero, metric="absolute_error").summary["relative_error_rms"] is None
    with pytest.raises(ValueError, match="zero desired field"):
        metric_map(zero, metric="absolute_error", normalization="target_rms")


def test_true_sample_weighting_changes_rms_and_excludes_zero_weight_witness():
    source = dataset(actual=(1, 2, 1000), desired=(0, 0, 0), weights=(1, 3, 0))
    equal = metric_map(source, metric="signed_residual")
    weighted = metric_map(source, metric="signed_residual", sampling="weighted")
    assert equal.summary["peak"] == 1000
    assert weighted.summary["peak"] == 2
    assert weighted.summary["rms"] == pytest.approx(np.sqrt(0.25 + 3))
    assert weighted.summary["mean"] == pytest.approx(1.75)
    assert weighted.summary["std"] == pytest.approx(np.sqrt(0.25 * 0.75**2 + 0.75 * 0.25**2))
    assert weighted.summary["sample_count"] == 2
    assert "saved sample weights" in weighted.sampling_convention
    with pytest.raises(ValueError, match="weights are unavailable"):
        metric_map(dataset(), sampling="weighted")


def test_parallel_projection_and_magnitude_difference_are_distinct_from_vector_error():
    source = FieldDataset("field", "Field", [[0, 0, 0]], [[3, 4, 0]], [[-3, -4, 0]])
    assert metric_map(source, "Bparallel", b0_direction=(3, 4, 0)).values[0] == pytest.approx(5)
    assert metric_map(source, "magnitude", "signed_residual").values[0] == 0
    assert metric_map(source, "Bz", "vector_error").values[0] == 10
    with pytest.raises(ValueError, match="B0 direction"):
        metric_map(source, "Bparallel", b0_direction=(0, 0, 0))
    large = dataset(actual=(1e300, 1e300, 1e300), desired=(0, 0, 0))
    assert metric_map(large, metric="absolute_error").summary["rms"] == pytest.approx(1e300)


def test_harmonic_fit_preserves_exact_basis_coefficients_shared_target_and_residual():
    rng = np.random.default_rng(42)
    center, radius = np.array([0.01, -0.02, 0.03]), 0.08
    points = center + rng.uniform(-0.04, 0.04, (150, 3))
    basis, _, _, _ = solid_spherical_harmonic_basis(points, 3, center=center, radius=radius)
    desired_coefficients = rng.normal(size=16) * 1e-6
    error_coefficients = rng.normal(size=16) * 1e-9
    target = np.column_stack((np.zeros((150, 2)), basis @ desired_coefficients))
    actual = target + np.column_stack((np.zeros((150, 2)), basis @ error_coefficients))
    source = FieldDataset("sheet", "Sheet", points, actual, target)
    progress = []
    report = analyze_harmonics(
        source,
        max_order=3,
        center_m=center,
        radius_m=radius,
        progress=lambda percent, message: progress.append(percent),
    )
    np.testing.assert_allclose(report["fits"]["target"].coefficients, desired_coefficients, rtol=1e-11, atol=1e-20)
    np.testing.assert_allclose(report["fits"]["residual"].coefficients, error_coefficients, rtol=1e-10, atol=1e-21)
    assert report["rank"] == report["term_count"] == 16
    assert report["condition"] == pytest.approx(np.linalg.cond(basis))
    assert len(report["coefficient_rows"]["residual"]) == 16
    assert report["fits"]["field"].residual_rms < 1e-20
    assert not report["source_free_verified"] and not report["acceptance_claim"]
    assert "source-free homogeneous domain" in report["sourcefree_caveat"]
    assert progress == [10, 90, 100]


@pytest.mark.parametrize("thickness", [0.0, 1e-9, 0.03])
def test_existing_harmonic_solver_retains_lstsq_rank_coefficients_and_singular_diagnostics(thickness):
    rng = np.random.default_rng(51)
    points = rng.uniform(-0.1, 0.1, (45, 3))
    points[:, 2] *= thickness
    values = np.column_stack((points[:, 0] + 2 * points[:, 1], points[:, 2] * 5))
    basis, _, _, _ = solid_spherical_harmonic_basis(points, 2, center=np.zeros(3), radius=0.2)
    expected, _, rank, singular = np.linalg.lstsq(basis, values, rcond=None)
    fits = fit_spherical_harmonics_many(
        points, {"first": values[:, 0], "second": values[:, 1]}, max_order=2, center=np.zeros(3), radius=0.2
    )
    for index, fit in enumerate(fits.values()):
        np.testing.assert_array_equal(fit.coefficients, expected[:, index])
        np.testing.assert_array_equal(fit.singular_values, singular)
        assert fit.rank == rank


def test_harmonic_rank_deficiency_and_cancellation_never_claim_unique_coefficients():
    source = dataset()
    report = analyze_harmonics(source, max_order=2, center_m=(0, 0, 0), radius_m=2)
    assert not report["full_rank"] and report["condition"] == np.inf
    assert report["diagnostics"]
    with pytest.raises(ValueError, match="not a scalar harmonic"):
        analyze_harmonics(source, component="magnitude")
    with pytest.raises(ValueError, match="positive"):
        analyze_harmonics(source, radius_m=0)
    with pytest.raises(ValueError, match="integer"):
        analyze_harmonics(source, max_order=3.5)
    with pytest.raises(InterruptedError, match="cancelled"):
        analyze_harmonics(source, cancel_check=lambda: True)


def test_real_saved_sheet_source_analysis_uses_saved_pairs_and_preserves_run(tmp_path):
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
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
    )
    project = ProjectDocument.from_config(config)
    DesignService.run(project, tmp_path / "run", pipeline_stage="sheet", validation_resolution=2)
    record = DesignService.load_run(tmp_path / "run")
    before = (record.directory / "result.json").read_bytes()
    sources = {item.key: item for item in source_catalog(record)}
    np.testing.assert_array_equal(sources["sheet"].field_T, record.array("sheet_field_T"))
    np.testing.assert_array_equal(sources["sheet"].target_T, record.array("target_field_T"))
    assert metric_map(sources["sheet"], metric="absolute_error").summary["rms"] >= 0
    assert analyze_harmonics(sources["sheet"], max_order=1)["sample_count"] == 8
    assert (record.directory / "result.json").read_bytes() == before
