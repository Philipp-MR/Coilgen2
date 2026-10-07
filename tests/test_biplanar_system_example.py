"""Physical basis, world frames and portable independent system examples."""

from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

from coilgen.application.assembly import ProjectAssembly
from coilgen.application.bundles import RunArchive
from coilgen.application.service import DesignService
from coilgen.application.system_presets import (
    BIPLANAR_GRADIENT_T_PER_M,
    BIPLANAR_ROI_RADIUS_M,
    BIPLANAR_SECOND_ORDER_PEAK_T,
    BIPLANAR_Z0_FIELD_T,
    SYSTEM_PRESET_NAMES,
    system_assembly,
    system_description,
)
from coilgen.application.targets import recognize_solid_harmonic_target
from coilgen.geometry import load_mesh_from_config
from coilgen.models import Mesh
from coilgen.target import define_target_field


@pytest.fixture(scope="module")
def assembly():
    return system_assembly()


def _cartesian_fields(points):
    x, y, z = points.T
    amplitude = BIPLANAR_SECOND_ORDER_PEAK_T / BIPLANAR_ROI_RADIUS_M**2
    gradient = BIPLANAR_GRADIENT_T_PER_M
    return {
        "gx": gradient * x,
        "gy": gradient * y,
        "gz": gradient * z,
        "z0": np.full(len(points), BIPLANAR_Z0_FIELD_T),
        "z2": amplitude * (2 * z**2 - x**2 - y**2) / 2,
        "xz": amplitude * 2 * x * z,
        "yz": amplitude * 2 * y * z,
        "x2-y2": amplitude * (x**2 - y**2),
        "xy": amplitude * 2 * x * y,
    }


def test_nine_owned_full_studies_have_realistic_nominal_layer_geometry(assembly):
    assert len(assembly.channels) == 9
    assert len({channel.project.id for channel in assembly.channels}) == 9
    assert len({channel.project.fingerprint for channel in assembly.channels}) == 9
    assert {channel.id for channel in assembly.channels} == set(_cartesian_fields(np.zeros((1, 3))))
    for index, channel in enumerate(assembly.channels):
        config = channel.project.config
        assert DesignService.validate(channel.project).ok
        assert channel.enabled
        assert channel.project.study["pipeline_stage"] == "evaluated"
        assert channel.project.study["fabricate"] is True
        assert channel.project.study["validation_resolution"] == 18
        assert config.target_error_tolerance_percent / channel.project.study["design_margin"] == pytest.approx(5)
        assert not config.skip_postprocessing and not config.skip_sweep and not config.contour_only_sweep_mode
        assert config.routing_circuit_scope == "per_part"
        assert config.field_evaluation_current_source == "stream_function"
        assert config.specific_conductivity_conductor == pytest.approx(18e-9)
        assert config.sweep_insulation_thickness == pytest.approx(0.0001)
        assert config.conductor_thickness == pytest.approx(0.001)
        assert not config.set_roi_into_mesh_center
        assert config.b_0_direction == [0, 0, 1] or config.b_0_direction == (0, 0, 1)
        assert config.b_0_strength_T == 3
        mesh = load_mesh_from_config(config, channel.project.project_root)
        half_gap = 0.05 + index * 0.006
        np.testing.assert_allclose(np.abs(mesh.vertices[:, 0]), half_gap, atol=1e-15)
        np.testing.assert_allclose(mesh.vertices.mean(axis=0), (0, 0, 0), atol=1e-15)
        np.testing.assert_allclose(np.ptp(mesh.vertices[:, 1:], axis=0), (0.30, 0.30), atol=1e-15)
        assert mesh.n_vertices == 450
        assert np.all(np.isfinite(mesh.vertices))
        triangles = mesh.vertices[mesh.faces]
        assert np.all(
            np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) > 0
        )
        width = 0.002 if channel.id in {"gx", "gy", "gz"} else 0.001
        assert config.conductor_cross_section_width == config.conductor_cross_section_height == width
        assert config.sweep_min_bend_radius == 3 * width


def test_target_basis_uses_identical_world_samples_and_is_well_conditioned(assembly):
    fields = []
    reference = None
    for channel in assembly.channels:
        target = define_target_field([], None, None, channel.project.config)
        if reference is None:
            reference = target.coords
        np.testing.assert_array_equal(target.coords, reference)
        np.testing.assert_allclose(target.b[:, 2], _cartesian_fields(reference)[channel.id], rtol=1e-12, atol=1e-18)
        np.testing.assert_array_equal(target.b[:, :2], np.zeros((len(reference), 2)))
        assert len(reference) == 257
        fields.append(target.b[:, 2])
    basis = np.column_stack(fields)
    assert np.linalg.matrix_rank(basis) == 9
    # Normalize columns before judging conditioning: gradients and shims have
    # intentionally different physical amplitudes and derivative units.
    normalized = basis / np.linalg.norm(basis, axis=0)
    assert np.linalg.cond(normalized) < 1.01
    np.testing.assert_allclose(normalized.T @ normalized, np.eye(9), atol=1e-13)


def test_quadratic_mode_phase_and_boundary_amplitudes_match_cartesian_convention(assembly):
    radius = BIPLANAR_ROI_RADIUS_M
    diagonal = radius / np.sqrt(2)
    points = np.array(
        [
            (0, 0, radius),
            (0, 0, -radius),
            (radius, 0, 0),
            (0, radius, 0),
            (diagonal, 0, diagonal),
            (diagonal, 0, -diagonal),
            (0, diagonal, diagonal),
            (0, diagonal, -diagonal),
            (diagonal, diagonal, 0),
            (diagonal, -diagonal, 0),
            (0, 0, 0),
        ]
    )
    region = Mesh(vertices=points, faces=np.zeros((0, 3), dtype=np.int64))
    expected = _cartesian_fields(points)
    for channel in assembly.channels[4:]:
        config = replace(channel.project.config, use_only_target_mesh_verts=True)
        recipe = recognize_solid_harmonic_target(config.field_shape_function)
        assert recipe.degree == 2 and recipe.reference_radius_m == radius and recipe.center_m == (0, 0, 0)
        assert config.target_gradient_normalization == "off" and config.target_field_units == "T"
        target = define_target_field([], region, None, config)
        np.testing.assert_allclose(target.b[:, 2], expected[channel.id], atol=1e-18)
        assert np.max(np.abs(target.b[:, 2])) == pytest.approx(BIPLANAR_SECOND_ORDER_PEAK_T)
    assert expected["z2"][0] == expected["z2"][1] == pytest.approx(10e-6)
    assert expected["z2"][2] == pytest.approx(-5e-6)
    assert expected["xz"][4] == pytest.approx(10e-6)
    assert expected["xz"][5] == pytest.approx(-10e-6)


def test_complete_system_json_is_portable_and_channels_remain_independent(assembly, tmp_path):
    path = assembly.save(tmp_path / "system.coilgen.json")
    portable = json.loads(path.read_text(encoding="utf-8"))
    assert portable["coordinate_frame"] == "world" and portable["units"] == "SI"
    restored = ProjectAssembly.from_dict(portable, project_root=tmp_path / "relocated")
    assert restored.fingerprint == assembly.fingerprint
    before = [channel.project.fingerprint for channel in restored.channels]
    edited = restored.channel("gy").project.update_config({"target_gradient_strength": 0.01})
    changed = restored.update_channel("gy", project=edited)
    assert [channel.project.fingerprint for channel in restored.channels] == before
    assert all(
        left.project.fingerprint == right.project.fingerprint
        for left, right in zip(restored.channels, changed.channels)
        if left.id != "gy"
    )
    assert "mutual inductance" not in system_description().lower() or "coupling" in system_description().lower()
    assert "construction release" in system_description().lower()


def _example_module():
    path = Path(__file__).parents[1] / "examples" / "biplanar_gradient_shim_workbench.py"
    spec = importlib.util.spec_from_file_location("biplanar_system_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_script_saves_collection_and_rejects_existing_output_before_changes(tmp_path):
    example = _example_module()
    directory = tmp_path / "new-system"
    assert example.main(["--output", str(directory)]) == 0
    path = directory / "biplanar_gradient_shim.coilgen.json"
    assert len(ProjectAssembly.load(path).channels) == 9
    previous = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        example.main(["--compute", "--output", str(directory)])
    assert exc.value.code == 2 and path.read_bytes() == previous
    invalid = tmp_path / "unknown-channel"
    with pytest.raises(SystemExit):
        example.main(["--compute", "--channels", "nonexistent", "--output", str(invalid)])
    assert not invalid.exists()


def test_script_retains_project_and_records_failed_channels_without_invented_results(tmp_path, monkeypatch):
    example = _example_module()

    def fail(_project, _directory, **_kwargs):
        raise RuntimeError("numerical smoke failure")

    monkeypatch.setattr(example.DesignService, "run", fail)
    directory = tmp_path / "failed-system"
    assert example.main(["--compute", "--channels", "gx", "z2", "--output", str(directory)]) == 2
    report = json.loads((directory / "channel_execution.json").read_text(encoding="utf-8"))
    assert set(report["failures"]) == {"gx", "z2"}
    restored = ProjectAssembly.load(directory / "biplanar_gradient_shim.coilgen.json")
    assert all(channel.project.saved_runs == () for channel in restored.channels)
    assert all(channel.project.study["pipeline_stage"] == "evaluated" for channel in restored.channels)


def test_unknown_system_kind_has_actionable_error():
    assert SYSTEM_PRESET_NAMES == ("biplanar_gradient_shim",)
    with pytest.raises(ValueError, match="choose biplanar_gradient_shim"):
        system_assembly("unknown")


def test_example_console_supports_legacy_windows_encoding(monkeypatch):
    example = _example_module()
    sink = io.BytesIO()
    stream = io.TextIOWrapper(sink, encoding="cp1252")
    monkeypatch.setattr(example.sys, "stdout", stream)
    example._emit("X²−Y²: completed")
    assert sink.getvalue().decode("cp1252").splitlines() == ["X²\\u2212Y²: completed"]


def test_completed_channel_results_survive_one_json_and_relocation(assembly, tmp_path):
    portable = assembly
    expected = {}
    for id_ in ("gy", "z2"):
        project = assembly.channel(id_).project.update_study({"pipeline_stage": "prepared", "fabricate": False})
        directory = tmp_path / "source-runs" / id_
        DesignService.run(project, directory)
        record = DesignService.load_run(directory)
        expected[id_] = record.arrays
        archive = RunArchive.capture(record, alias=f"{id_} prepared evidence")
        portable = portable.update_channel(id_, project=project.with_results((archive,)))
    path = portable.save(tmp_path / "single-system.coilgen.json")
    restored = ProjectAssembly.from_dict(json.loads(path.read_text(encoding="utf-8")), project_root=tmp_path / "moved")
    for id_, arrays in expected.items():
        channel = restored.channel(id_)
        assert len(channel.project.saved_runs) == 1
        record = channel.project.saved_runs[0].materialize(tmp_path / "moved-runs")
        assert record.metadata["project_id"] == channel.project.id
        assert record.metadata["pipeline_stage"] == "prepared"
        assert set(record.arrays) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_array_equal(record.arrays[key], value)
    assert all(not channel.project.saved_runs for channel in restored.channels if channel.id not in expected)
