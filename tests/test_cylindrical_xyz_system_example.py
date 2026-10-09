"""A complete XYZ insert retains physical layers and independent world targets."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from coilgen.application import DesignService, ProjectAssembly
from coilgen.application.system_presets import (
    CYLINDRICAL_GRADIENT_T_PER_M,
    CYLINDRICAL_LENGTH_M,
    CYLINDRICAL_ROI_RADIUS_M,
    CYLINDRICAL_SUPPORT_RADII_M,
    system_assembly,
    system_description,
)
from coilgen.application.templates import template_assembly, template_spec
from coilgen.geometry import load_mesh_from_config
from coilgen.target import define_target_field


KIND = "cylindrical_xyz_gradient"


def test_three_ready_studies_have_separate_coaxial_supports_and_real_conductor_settings(tmp_path):
    assembly = template_assembly(KIND, project_root=tmp_path)
    assert template_spec(KIND).category == "Coil systems"
    assert tuple(channel.id for channel in assembly.channels) == ("gx", "gy", "gz")
    assert tuple(channel.name for channel in assembly.channels) == ("Gx", "Gy", "Gz")
    assert len({channel.project.id for channel in assembly.channels}) == 3
    assert len({channel.project.fingerprint for channel in assembly.channels}) == 3
    for channel, radius in zip(assembly.channels, CYLINDRICAL_SUPPORT_RADII_M):
        project, config = channel.project, channel.project.config
        DesignService.validate(project).raise_for_errors()
        assert channel.enabled and not project.saved_runs
        assert project.study["pipeline_stage"] == "evaluated" and project.study["fabricate"]
        assert project.study["validation_resolution"] == 18
        assert config.target_error_tolerance_percent / project.study["design_margin"] == pytest.approx(5)
        assert not config.skip_postprocessing and not config.skip_sweep
        assert not config.contour_only_sweep_mode and not config.skip_inductance_calculation
        assert config.constrained_quadratic_fallback_policy == "strict"
        assert config.research_physics["objective"] == "power"
        assert config.field_evaluation_current_source == "stream_function"
        assert config.conductor_cross_section_width == config.conductor_cross_section_height == 0.002
        assert config.sweep_min_bend_radius == 0.006
        assert config.sweep_insulation_thickness == 0.0001
        assert config.specific_conductivity_conductor == 18e-9
        assert config.levels == 16 and config.conductor_thickness == 0.001
        assert config.b_0_strength_T == 3 and tuple(config.b_0_direction) == (0, 0, 1)
        assert not config.set_roi_into_mesh_center
        mesh = load_mesh_from_config(config, tmp_path)
        assert mesh.n_vertices == 544 and len(mesh.boundaries) == 2
        np.testing.assert_allclose(np.linalg.norm(mesh.vertices[:, :2], axis=1), radius, atol=1e-14)
        np.testing.assert_allclose(np.ptp(mesh.vertices[:, 2]), CYLINDRICAL_LENGTH_M, atol=1e-14)
        np.testing.assert_allclose(mesh.vertices.mean(axis=0), 0, atol=1e-14)
        triangles = mesh.vertices[mesh.faces]
        assert np.all(
            np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) > 0
        )
        assert project.session["system_example"]["nominal_support_radius_m"] == radius
    # Nominal support pitch leaves room for insulated 2 mm conductor bodies;
    # routes and feeds still need a separately computed assembly-clearance check.
    assert np.min(np.diff(CYLINDRICAL_SUPPORT_RADII_M)) > 0.002 + 2 * 0.0001


def test_xyz_targets_share_world_samples_and_produce_the_correct_bz_gradient_signs():
    fields, reference = [], None
    for index, channel in enumerate(system_assembly(KIND).channels):
        config = channel.project.config
        assert config.target_gradient_axis == "xyz"[index]
        assert config.target_region_radius == CYLINDRICAL_ROI_RADIUS_M
        target = define_target_field([], None, None, config)
        if reference is None:
            reference = target.coords
        np.testing.assert_array_equal(target.coords, reference)
        np.testing.assert_allclose(target.b[:, 2], CYLINDRICAL_GRADIENT_T_PER_M * reference[:, index], atol=1e-18)
        np.testing.assert_array_equal(target.b[:, :2], 0)
        assert len(reference) == 257
        assert np.max(target.b[:, 2]) == pytest.approx(0.0004)
        assert np.min(target.b[:, 2]) == pytest.approx(-0.0004)
        fields.append(target.b[:, 2])
    sampled = np.column_stack(fields)
    assert np.linalg.matrix_rank(sampled) == 3
    normalized = sampled / np.linalg.norm(sampled, axis=0)
    np.testing.assert_allclose(normalized.T @ normalized, np.eye(3), atol=1e-13)


def test_one_portable_json_keeps_channels_independent_and_makes_qualification_scope_explicit(tmp_path):
    assembly = system_assembly(KIND, project_root=tmp_path)
    path = assembly.save(tmp_path / "xyz.coilgen.json")
    restored = ProjectAssembly.from_dict(json.loads(path.read_text(encoding="utf-8")), project_root=tmp_path / "moved")
    assert restored.fingerprint == assembly.fingerprint
    project = restored.channel("gy").project
    changed = restored.update_channel("gy", project=project.update_config({"target_gradient_strength": 0.01}))
    assert changed.channel("gy").project.fingerprint != restored.channel("gy").project.fingerprint
    for id_ in ("gx", "gz"):
        assert changed.channel(id_).project is restored.channel(id_).project
    assert "unshielded" in system_description(KIND).lower()
    assert "construction release" in system_description(KIND).lower()
    assert "mutual inductance" in system_description(KIND).lower()


def test_shared_example_runner_writes_three_channels_and_checks_fresh_output(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "biplanar_gradient_shim_workbench.py"
    spec = importlib.util.spec_from_file_location("system_example_runner", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / "xyz"
    assert module.main(["--output", str(directory)], kind=KIND) == 0
    path = directory / f"{KIND}.coilgen.json"
    assert len(ProjectAssembly.load(path).channels) == 3
    saved = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        module.main(["--compute", "--output", str(directory)], kind=KIND)
    assert exc.value.code == 2 and path.read_bytes() == saved
