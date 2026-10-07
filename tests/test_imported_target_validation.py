"""Physical target-file invariants apply before normalization and optimization."""

import json

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.pipeline import CoilGenPipeline
from coilgen.target import load_target_field_definition


def data():
    return {
        "coords": np.asarray([[0.0, 0.0, 0.01], [0.0, 0.01, 0.02]]),
        "b": np.asarray([[1e-5, 2e-5, 3e-5], [2e-5, 0.0, -1e-5]]),
        "weights": np.asarray([0.0, 1.0]),
        "suppressed": np.asarray([False, True]),
        "target_field_group_inds": np.asarray([-2, 0], dtype=np.int64),
        "target_gradient_dbdxyz": np.asarray([[0.001, 0.002, 0.003], [0.0, 0.0, 0.0]]),
        "suppression_normals": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        "suppression_area_weights": np.asarray([0.0, 0.002]),
    }


def write(path, values, kind):
    if kind == "json":
        path = path.with_suffix(".json")
        path.write_text(json.dumps({key: value.tolist() for key, value in values.items()}), encoding="utf-8")
    else:
        path = path.with_suffix(".npz")
        if kind == "owned":
            values = {**values, "schema": np.asarray("coilgen-target-v1"), "target_diagnostics_json": np.asarray("{}")}
        np.savez(path, **values)
    return path


def load(path):
    return load_target_field_definition(path, None, CoilGenConfig(target_gradient_normalization="off"))


@pytest.mark.parametrize("kind", ["npz", "json", "owned"])
@pytest.mark.parametrize(
    "field",
    [
        "coords",
        "b",
        "weights",
        "target_gradient_dbdxyz",
        "suppression_normals",
        "suppression_area_weights",
        "suppressed",
        "target_field_group_inds",
    ],
)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_target_samples_and_metadata_are_rejected(tmp_path, kind, field, bad):
    values = data()
    invalid = values[field].astype(float)
    invalid.flat[0] = bad
    values[field] = invalid
    with pytest.raises(ValueError, match="finite"):
        load(write(tmp_path / "target", values, kind))


@pytest.mark.parametrize("kind", ["npz", "json", "owned"])
@pytest.mark.parametrize("field", ["weights", "suppression_area_weights"])
def test_negative_weight_is_not_silently_disabled(tmp_path, kind, field):
    values = data()
    values[field][0] = -0.01
    with pytest.raises(ValueError, match="nonnegative"):
        load(write(tmp_path / "target", values, kind))


@pytest.mark.parametrize("kind", ["npz", "json", "owned"])
def test_zero_weights_and_surface_metadata_and_signed_group_labels_remain_valid(tmp_path, kind):
    values = data()
    loaded = load(write(tmp_path / "target", values, kind))
    for field, expected in values.items():
        np.testing.assert_array_equal(getattr(loaded, field), expected)


@pytest.mark.parametrize("kind", ["npz", "json", "owned"])
def test_suppression_mask_requires_boolean_or_binary_numeric_flags(tmp_path, kind):
    values = data()
    values["suppressed"] = np.asarray([0.0, 2.0])
    with pytest.raises(ValueError, match="booleans or numeric 0/1"):
        load(write(tmp_path / "target", values, kind))
    values["suppressed"] = np.asarray([0.0, 1.0])
    np.testing.assert_array_equal(load(write(tmp_path / "target", values, kind)).suppressed, [False, True])


@pytest.mark.parametrize(
    "groups", [np.asarray([1.5, 2.0]), np.asarray([0, 2**63], dtype=np.uint64), np.asarray([0.0, float(2**63)])]
)
def test_group_labels_are_integral_and_int64_representable(tmp_path, groups):
    values = data()
    values["target_field_group_inds"] = groups
    with pytest.raises(ValueError, match="integer"):
        load(write(tmp_path / "target", values, "npz"))


def test_int64_group_label_extremes_are_preserved(tmp_path):
    values = data()
    values["target_field_group_inds"] = np.asarray([np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64)
    loaded = load(write(tmp_path / "target", values, "npz"))
    np.testing.assert_array_equal(loaded.target_field_group_inds, values["target_field_group_inds"])


@pytest.mark.parametrize("column,bad", [("x", "nan"), ("bz", "inf"), ("weight", "-1"), ("dbdy", "-inf")])
@pytest.mark.parametrize("suffix,separator", [(".csv", ","), (".tsv", "\t")])
def test_csv_tsv_apply_same_physical_checks(tmp_path, column, bad, suffix, separator):
    row = {"x": "0", "y": "0", "z": ".01", "bz": "1e-5", "weight": "1", "dbdx": "0", "dbdy": "0", "dbdz": "0"}
    row[column] = bad
    path = tmp_path / f"target{suffix}"
    path.write_text(separator.join(row) + "\n" + separator.join(row.values()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite|nonnegative"):
        load(path)


def test_invalid_import_fails_before_operator_assembly(tmp_path, monkeypatch):
    values = data()
    values["b"][0, 2] = np.nan
    path = write(tmp_path / "target", values, "npz")

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid target reached numerical basis/operator assembly.")

    monkeypatch.setattr("coilgen.pipeline.calculate_one_ring_by_mesh", unexpected)
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 8, 4, 1, 0, 0, 0),
        target_field_definition_file=str(path),
        target_gradient_normalization="off",
        progress_flag=False,
    )
    with pytest.raises(ValueError, match="finite"):
        CoilGenPipeline(config).run_stage("prepared")


def test_complex_npz_field_is_not_silently_projected_to_real(tmp_path):
    values = data()
    values["b"] = values["b"].astype(complex) + 1j
    with pytest.raises(ValueError, match="real values"):
        load(write(tmp_path / "target", values, "npz"))


@pytest.mark.parametrize("field", ["b", "target_gradient_dbdxyz"])
def test_normalization_overflow_cannot_create_a_nonfinite_solver_target(tmp_path, field):
    values = data()
    values[field].flat[0] = 1e200
    path = write(tmp_path / "target", values, "npz")
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="finite"):
        load_target_field_definition(
            path, None, CoilGenConfig(target_gradient_normalization="scale", target_gradient_strength=1e200)
        )
