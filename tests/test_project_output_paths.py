from __future__ import annotations

from pathlib import Path

import pytest

from projects._shared.paths import (
    PROJECT_DATA_ROOT,
    REPOSITORY_ROOT,
    begin_project_run,
    latest_project_output,
    project_data_dir,
    project_output_dir,
    resolve_project_path,
)
from projects.halbach_cylindrical_3axis_gradient.paths import (
    HALBACH_DATA_ROOT,
    HALBACH_OUTPUT_ROOT,
    resolve_halbach_path,
)


def test_halbach_project_data_root_is_centralized() -> None:
    assert HALBACH_DATA_ROOT == PROJECT_DATA_ROOT / "halbach_cylindrical_3axis_gradient"
    assert resolve_halbach_path("inputs/coil_surface.stl") == HALBACH_DATA_ROOT / "coil_surface.stl"


def test_halbach_output_helper_does_not_create_an_empty_run_until_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from projects._shared import paths
    from projects.halbach_cylindrical_3axis_gradient.paths import halbach_output_dir

    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(paths, "_PROJECT_RUNS", {})

    lazy_root = halbach_output_dir()

    assert lazy_root.parent == tmp_path / "outputs" / "halbach_cylindrical_3axis_gradient"
    assert not lazy_root.exists()
    assert halbach_output_dir(create=True) == lazy_root
    assert lazy_root.is_dir()


def test_halbach_execution_modules_use_central_output_root() -> None:
    from projects.halbach_cylindrical_3axis_gradient import gradient_project
    from projects.halbach_cylindrical_3axis_gradient import run_halbach_cylindrical_3axis_gradient as run_script

    assert gradient_project.OUTPUT_ROOT == HALBACH_OUTPUT_ROOT
    assert run_script.OUTPUT_ROOT == HALBACH_OUTPUT_ROOT
    assert run_script.PROJECT.output_dir == HALBACH_OUTPUT_ROOT
    config = run_script.GRADIENT_TEMPLATE.to_config()
    assert run_script.SURFACE_MESH_FILES == {
        "x": "Octagonal Gradients - Gx Surface_remeshed.stl",
        "y": "Octagonal Gradients - Gy Surface_remeshed.stl",
        "z": "Octagonal Gradients - Gz Surface_remeshed.stl",
    }
    assert run_script.SURFACE_MESH_LENGTH_SCALE_TO_M == pytest.approx(1.0e-3)
    assert config.coil_mesh_file == run_script.SURFACE_MESH_FILES["x"]
    assert config.geometry_source_path == HALBACH_DATA_ROOT
    assert config.imported_mesh_length_scale_to_m == pytest.approx(1.0e-3)
    assert config.surface_is_cylinder_flag
    resolved_designs = run_script.PROJECT.resolved_designs()
    for axis in "xyz":
        axis_config = resolved_designs[f"G{axis.upper()}"].to_config()
        assert axis_config.coil_mesh_file == run_script.SURFACE_MESH_FILES[axis]
        assert axis_config.imported_mesh_length_scale_to_m == pytest.approx(1.0e-3)
    assert run_script.DSV_DIAMETER_M == pytest.approx(0.050)
    assert run_script.WIRE_DIAMETER_M == pytest.approx(0.0014)
    assert run_script.SWEEP_CROSS_SECTION == "trapezoid"
    assert run_script.RECTANGLE_WIDTH_M == pytest.approx(0.0017)
    assert run_script.RECTANGLE_HEIGHT_M == pytest.approx(0.0017)
    assert run_script.TRAPEZOID_BOTTOM_WIDTH_M == pytest.approx(0.00196)
    assert run_script.TRAPEZOID_TOP_WIDTH_M == pytest.approx(0.00160)
    assert run_script.TRAPEZOID_HEIGHT_M == pytest.approx(0.00180)
    assert config.conductor_cross_section_width == pytest.approx(0.00196)
    assert config.conductor_cross_section_height == pytest.approx(0.00180)
    expected_trapezoid = (
        (-0.00098, -0.00090),
        (+0.00098, -0.00090),
        (+0.00080, +0.00090),
        (-0.00080, +0.00090),
    )
    assert len(config.cross_sectional_points) == len(expected_trapezoid)
    for actual, expected in zip(config.cross_sectional_points, expected_trapezoid):
        assert actual == pytest.approx(expected)
    assert config.contour_path_regularization_mode == "parameterized_bspline"
    assert config.contour_smoothing_patch_rings == run_script.CONTOUR_SMOOTHING_PATCH_RINGS
    assert config.contour_resample_target_segment_length == pytest.approx(
        run_script.CONTOUR_SMOOTHING_TARGET_SEGMENT_LENGTH_M
    )
    assert config.contour_smoothing_factor == pytest.approx(run_script.CONTOUR_SMOOTHING_FACTOR)
    assert config.contour_smoothing_scale_mode == run_script.CONTOUR_SMOOTHING_SCALE_MODE
    assert config.contour_smoothing_mesh_gain == pytest.approx(run_script.CONTOUR_SMOOTHING_MESH_GAIN)
    assert (
        config.contour_smoothing_work_samples_per_mesh_length
        == run_script.CONTOUR_SMOOTHING_WORK_SAMPLES_PER_MESH_LENGTH
    )
    assert (
        config.contour_smoothing_output_samples_per_mesh_length
        == run_script.CONTOUR_SMOOTHING_OUTPUT_SAMPLES_PER_MESH_LENGTH
    )
    assert config.contour_smoothing_max_displacement == pytest.approx(
        run_script.CONTOUR_SMOOTHING_MAX_DISPLACEMENT_M if run_script.ENABLE_CONTOUR_SMOOTHING_DISPLACEMENT_CAP else 0.0
    )
    assert config.inter_group_connection_path_method == "direct_3d"
    assert config.skip_normal_shift is not run_script.ENABLE_NORMAL_SHIFT
    assert config.normal_shift_length == pytest.approx(run_script.NORMAL_SHIFT_LENGTH_M)
    assert config.normal_sweep_smooth_factor == pytest.approx(run_script.NORMAL_SHIFT_TRANSITION_FACTOR)


def test_projects_have_no_legacy_local_output_directories() -> None:
    project_directories = (path for path in (REPOSITORY_ROOT / "projects").iterdir() if path.is_dir())

    assert not [
        legacy
        for project in project_directories
        for legacy in (project / "output", project / "outputs")
        if legacy.exists()
    ]


def test_common_output_tree_is_ignored_as_one_repository_root() -> None:
    ignore_lines = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/outputs/" in ignore_lines


def test_common_project_data_tree_is_ignored_as_one_repository_root() -> None:
    ignore_lines = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/project_data/" in ignore_lines


def test_explicit_run_boundaries_never_reuse_an_output_directory(tmp_path: Path, monkeypatch) -> None:
    from projects._shared import paths

    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(paths, "_PROJECT_RUNS", {})

    first = begin_project_run("example")
    first_file = project_output_dir("example", "case") / "result.txt"
    first_file.write_text("first", encoding="utf-8")
    second = begin_project_run("example")
    second_file = project_output_dir("example", "case") / "result.txt"
    second_file.write_text("second", encoding="utf-8")

    assert first != second
    assert first_file.read_text(encoding="utf-8") == "first"
    assert second_file.read_text(encoding="utf-8") == "second"


def test_reexecuting_a_project_script_automatically_starts_a_new_run(tmp_path: Path, monkeypatch) -> None:
    from projects._shared import paths

    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(paths, "_PROJECT_RUNS", {})
    namespace = {
        "__name__": "__main__",
        "__file__": str(REPOSITORY_ROOT / "projects" / "example" / "run_example.py"),
    }
    code = compile(
        "from projects._shared.paths import project_output_dir\nrun_root = project_output_dir('example')\n",
        namespace["__file__"],
        "exec",
    )

    exec(code, namespace)
    first = namespace["run_root"]
    exec(code, namespace)
    second = namespace["run_root"]

    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_latest_output_resolves_the_newest_matching_prior_run(tmp_path: Path, monkeypatch) -> None:
    from projects._shared import paths

    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(paths, "_PROJECT_RUNS", {})

    first = begin_project_run("example")
    first_file = project_output_dir("example", "case") / "result.txt"
    first_file.write_text("first", encoding="utf-8")
    second = begin_project_run("example")
    second_file = project_output_dir("example", "case") / "result.txt"
    second_file.write_text("second", encoding="utf-8")
    begin_project_run("example")

    assert latest_project_output("example", "case", "result.txt") == second_file
    assert first != second


def test_output_prefix_accepts_optional_redundant_project_name(tmp_path: Path, monkeypatch) -> None:
    from projects._shared import paths

    monkeypatch.setattr(paths, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(paths, "_PROJECT_RUNS", {})
    run_root = begin_project_run("example")

    resolved = resolve_project_path(
        REPOSITORY_ROOT / "projects" / "example",
        "example",
        "outputs/example/case/result.csv",
    )

    assert resolved == run_root / "case" / "result.csv"


def test_project_inputs_are_redirected_to_the_common_data_root() -> None:
    project_root = REPOSITORY_ROOT / "projects" / "example"

    assert (
        resolve_project_path(project_root, "example", "inputs/mesh.stl") == PROJECT_DATA_ROOT / "example" / "mesh.stl"
    )
    assert (
        resolve_project_path(project_root, "example", "data/field.nii") == PROJECT_DATA_ROOT / "example" / "field.nii"
    )


def test_explicit_repository_project_data_path_is_resolved_from_repository_root() -> None:
    project_root = REPOSITORY_ROOT / "projects" / "example"

    assert (
        resolve_project_path(project_root, "example", "project_data/example/mesh.stl")
        == PROJECT_DATA_ROOT / "example" / "mesh.stl"
    )


@pytest.mark.parametrize("unsafe", ("../outside", Path("..") / "outside", "C:/outside"))
def test_output_helper_rejects_paths_outside_the_common_root(unsafe: str | Path) -> None:
    with pytest.raises(ValueError):
        project_output_dir("example", unsafe, create=False)


@pytest.mark.parametrize("unsafe", ("../outside", Path("..") / "outside", "C:/outside"))
def test_data_helper_rejects_paths_outside_the_common_root(unsafe: str | Path) -> None:
    with pytest.raises(ValueError):
        project_data_dir("example", unsafe, create=False)
