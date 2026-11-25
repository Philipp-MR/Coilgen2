"""Editable exports and direct scripts retain numerical inputs without a GUI."""

from dataclasses import replace
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.application import ProjectDocument
from _coilgen_test_support.application_backend import _config, _mesh, _target


ROOT = Path(__file__).resolve().parents[1]
BLOCK_GUI = """
import importlib.abc, sys
class BlockGUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6', 'pyvista', 'pyvistaqt', 'vtk', 'vtkmodules', 'trame'}:
            raise AssertionError('Headless script imported ' + fullname)
sys.meta_path.insert(0, BlockGUI())
"""


def headless(code, *args, cwd):
    completed = subprocess.run(
        [sys.executable, "-c", BLOCK_GUI + code, *map(str, args)],
        cwd=cwd,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_standalone_settings_are_editable_and_import_does_not_run(tmp_path):
    project = ProjectDocument.from_config(_config(), name="Gy 'quoted' \"name\" µT\\coil\nsecond line")
    project = project.update_study({"pipeline_stage": "sheet", "validation_resolution": 7, "design_margin": 0.8})
    script = project.export_python(tmp_path / "editable.py", standalone=True)
    namespace = runpy.run_path(str(script))
    assert list(tmp_path.iterdir()) == [script]
    exported = namespace["build_project"]()
    assert exported.to_dict() == project.inputs_only().to_dict()
    assert exported.fingerprint == project.fingerprint
    namespace["CONFIG"].levels = 13
    namespace["STUDY"]["validation_resolution"] = 9
    edited = namespace["build_project"]()
    assert edited.config.levels == 13 and edited.study["validation_resolution"] == 9
    assert edited.fingerprint != project.fingerprint
    assert project.config.levels != 13 and project.study["validation_resolution"] == 7


def test_embedded_inputs_survive_relocation_and_source_deletion(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = ProjectDocument.from_config(_config())
    # Exercise real file provenance as well as process-local mesh/target capture.
    config = original._resolved_config(source)
    project = ProjectDocument.from_config(config)
    script = project.export_python(tmp_path / "design.py", standalone=True)
    source_files = tuple(source.iterdir())
    assert source_files
    for path in source_files:
        path.unlink()
    assert all(not path.exists() for path in source_files)
    # The source files must be absent; removing their empty parent adds only
    # incidental Windows directory-lock behavior to this portability test.
    moved = tmp_path / "moved"
    moved.mkdir()
    script = script.replace(moved / "renamed.py")
    elsewhere = tmp_path / "empty"
    elsewhere.mkdir()
    headless(
        """
import runpy
from pathlib import Path
from coilgen.geometry import load_mesh_from_config
from coilgen.target import load_target_field_definition
import numpy as np
ns = runpy.run_path(sys.argv[1])
project = ns['build_project']()
assert project.fingerprint == sys.argv[2]
assert not list(Path.cwd().iterdir())
config = project.config
mesh = load_mesh_from_config(config, project.project_root)
target = load_target_field_definition(config.target_field_definition_file, None, config)
np.savez(sys.argv[3], vertices=mesh.vertices, faces=mesh.faces, coords=target.coords,
         b=target.b, weights=target.weights, suppressed=target.suppressed)
""",
        script,
        project.fingerprint,
        tmp_path / "inputs.npz",
        cwd=elsewhere,
    )
    with np.load(tmp_path / "inputs.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["vertices"], _mesh().vertices)
        np.testing.assert_array_equal(data["faces"], _mesh().faces)
        for key in ("coords", "b", "weights", "suppressed"):
            np.testing.assert_array_equal(data[key], getattr(_target(), key))


@pytest.mark.parametrize("options", [{"compact": True}, {"include_results": True}])
def test_standalone_rejects_companion_options_before_writing(tmp_path, options):
    with pytest.raises(ValueError, match="Standalone Python"):
        ProjectDocument.new().export_python(tmp_path / "bad.py", standalone=True, **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("geometry", ["cylinder", "biplanar"])
def test_exported_script_matches_direct_numerical_pipeline_without_gui(tmp_path, geometry):
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 12, 6, 1, 0, 0, 0),
        surface_is_cylinder_flag=True,
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_radius=0.035,
        target_region_resolution=5,
        levels=4,
        min_point_loop_number=3,
        regularization_selection_method="fixed",
        tikonov_reg_factor=10,
        skip_sweep=True,
        skip_inductance_calculation=True,
        parallel_worker_count=1,
        plot_flag=False,
        save_stl_flag=False,
        progress_flag=False,
    )
    if geometry == "biplanar":
        config = replace(
            config,
            coil_mesh_file="create bi-planary mesh",
            surface_is_cylinder_flag=False,
            biplanar_mesh_parameter_list=(0.3, 0.3, 6, 6, 0, 0, 1, 0, 0, 0, 0.2),
            field_shape_function="z",
            skip_postprocessing=True,
        )
    project = ProjectDocument.from_config(config).update_study({"validation_resolution": 4})
    script = project.export_python(tmp_path / "design.py", standalone=True)
    headless(
        """
import runpy
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from coilgen import run_coilgen
ns = runpy.run_path(sys.argv[1])
with threadpool_limits(1):
    direct = run_coilgen(ns['CONFIG'], project_root=Path(sys.argv[1]).parent)
assert direct.coil_parts and any(part.contour_lines for part in direct.coil_parts)
if not ns['CONFIG'].skip_postprocessing:
    assert any(part.wire_path is not None for part in direct.coil_parts)
assert np.isfinite(direct.needed_current_layout) and direct.needed_current_layout != 0
output = Path(sys.argv[1]).with_suffix('.results')
sys.argv = [sys.argv[1], '--output', str(output), '--threads', '1']
runpy.run_path(sys.argv[0], run_name='__main__')
with np.load(output / 'scientific_result.npz', allow_pickle=False) as data:
    for key, expected in [('stream_function_A', direct.combined_mesh.stream_function),
                          ('sheet_field_T', direct.b_field_opt_sf),
                          ('target_field_T', direct.target_field.b)]:
        np.testing.assert_allclose(data[key], expected, rtol=1e-11, atol=1e-14)
    for index, part in enumerate(direct.coil_parts):
        for contour_index, contour in enumerate(part.contour_lines):
            np.testing.assert_allclose(data[f'contour_{index}_{contour_index}_points_m'], contour.v,
                                       rtol=1e-11, atol=1e-14)
        if part.wire_path is not None:
            np.testing.assert_allclose(data[f'route_{index}_points_m'], part.wire_path.v, rtol=1e-11, atol=1e-14)
""",
        script,
        cwd=tmp_path,
    )


def test_cli_can_export_a_single_editable_file(tmp_path):
    from coilgen.cli import main

    project = ProjectDocument.new(config=CoilGenConfig(coil_mesh_file="create cylinder mesh"))
    saved = project.save(tmp_path / "input.json")
    script = tmp_path / "cli.py"
    assert main(["export-script", str(saved), str(script), "--standalone"]) == 0
    assert runpy.run_path(str(script))["build_project"]().fingerprint == project.fingerprint
    assert not script.with_suffix(".coilgen.json").exists()


def test_object_design_api_runs_in_a_plain_python_process(tmp_path):
    headless(
        """
import numpy as np
from threadpoolctl import threadpool_limits
from coilgen import CylinderCoilDesign, CoilGenPipeline
design = CylinderCoilDesign(
    target_bz_expression='y', target_scale=0.01, radius_m=0.12, length_m=0.3,
    angular_divisions=12, length_divisions=6, target_region_radius_m=0.035,
    target_region_resolution=5, contour_levels=4, min_contour_points=3,
    regularization_method='fixed', regularization_factor=10, cpu_workers=1,
    progress=False, skip_sweep=True, skip_inductance=True,
)
with threadpool_limits(1):
    result = design.run(print_validation_warnings=False)
    pipeline_result = CoilGenPipeline(design.to_config()).run()
assert any(part.wire_path is not None for part in result.coil_parts)
assert np.isfinite(result.needed_current_layout) and result.needed_current_layout != 0
np.testing.assert_array_equal(result.combined_mesh.stream_function, pipeline_result.combined_mesh.stream_function)
np.testing.assert_array_equal(result.b_field_opt_sf, pipeline_result.b_field_opt_sf)
""",
        cwd=tmp_path,
    )
