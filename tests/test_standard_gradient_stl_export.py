from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.manufacturing import (
    SweptBodyExportError,
    mesh_manufacturing_diagnostics,
    save_swept_bodies_as_step,
    save_swept_bodies_as_stl,
)
from coilgen.models import CoilPart, Mesh
from projects.standard_cylindrical_3axis_gradient import run_standard_cylindrical_3axis_gradient as example


def _part(*, rejected=False):
    mesh = Mesh(
        vertices=0.01 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
    )
    diagnostics = mesh_manufacturing_diagnostics(mesh)
    if rejected:
        diagnostics.update(
            sweep_route_clearance_violation=True,
            sweep_capsule_clearance_violation=True,
            sweep_capsule_self_intersection=True,
            sweep_curvature_envelope_violation=True,
            sweep_triangle_quality_violation=True,
        )
    return CoilPart(mesh=mesh, swept_body=mesh, manufacturing_diagnostics=diagnostics)


def _channel(name, *parts, solved=True):
    return SimpleNamespace(
        result=SimpleNamespace(coil_parts=list(parts)) if solved else None,
        channel=SimpleNamespace(name=name, safe_name=name.lower()),
        design=SimpleNamespace(name=f"fixture_{name.lower()}"),
        files=(),
    )


def _run(tmp_path, *channels):
    return SimpleNamespace(channel_runs=channels, output_dir=tmp_path, files=(), figures=[object()])


@pytest.mark.parametrize("exporter", [save_swept_bodies_as_stl, save_swept_bodies_as_step])
def test_readiness_rejection_has_distinct_value_error_type_and_writes_no_mesh(tmp_path, exporter):
    result = SimpleNamespace(coil_parts=[_part(), _part(rejected=True)])
    with pytest.raises(SweptBodyExportError, match="Part 2.*route_clearance_violation") as caught:
        exporter(result, tmp_path, fail_on_violations=True)
    assert isinstance(caught.value, ValueError)
    assert not list(tmp_path.iterdir())


def test_stl_default_exports_mesh_with_manufacturing_violations(tmp_path):
    part = _part(rejected=True)
    files = save_swept_bodies_as_stl(SimpleNamespace(coil_parts=[part]), tmp_path)
    assert len(files) == 2
    assert all(path.exists() for path in files)
    assert part.manufacturing_diagnostics["manufacturing_ok"] is False
    assert "sweep_self_intersection" in part.manufacturing_diagnostics["manufacturing_failure_reasons"]


def test_failing_channel_exports_stls_with_diagnostics_and_allows_later_channel(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(example, "EXPORT_STL", True)
    failing = _channel("GX", _part(), _part(rejected=True))
    valid = _channel("GY", _part())
    unavailable = _channel("GZ", solved=False)
    run = _run(tmp_path, failing, valid, unavailable)
    saved = example.export_project_stls(run)

    assert len(list((tmp_path / "gx").rglob("*.stl"))) == 4
    assert len(list((tmp_path / "gy").rglob("*.stl"))) == 2
    diagnostics_path = next(path for path in failing.files if path.suffix == ".json")
    rows = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["manufacturing_ok"] is True
    assert rows[1]["watertight"] is True
    assert rows[1]["manufacturing_ok"] is False
    assert rows[1]["manufacturing_failure_reasons"].split(";") == [
        "route_clearance_violation",
        "sweep_clearance_violation",
        "sweep_self_intersection",
        "curvature_envelope_violation",
        "poor_sweep_triangle_quality",
    ]
    assert any(path.suffix == ".csv" for path in failing.files)
    report = json.loads((tmp_path / "stl_export_report.json").read_text(encoding="utf-8"))
    assert [entry["status"] for entry in report] == ["exported", "exported", "unavailable"]
    assert report[0]["manufacturing_ready"] is False
    assert report[0]["manufacturing_warnings"][0]["part_index"] == 2
    assert "route_clearance_violation" in report[0]["manufacturing_warnings"][0]["reasons"]
    assert report[1]["manufacturing_ready"] is True
    for entry in report:
        assert all((tmp_path / path).exists() for path in entry["files"])
    assert "STL exported for GX with manufacturing warnings" in capsys.readouterr().out
    assert set(saved) == set(run.files)
    assert all(path.exists() for path in saved)
    assert not unavailable.files

    # Re-exporting a solved run does not multiply entries in its file inventory.
    assert example.export_project_stls(run) == saved
    assert len(run.files) == len(set(run.files))
    assert len(valid.files) == len(set(valid.files))


def test_run_returns_solved_results_and_plotbook_after_export_with_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(example, "EXPORT_STL", True)
    run = _run(tmp_path, _channel("GX", _part(rejected=True)))
    figures = run.figures
    project = SimpleNamespace(describe=lambda: "fixture", open_plot_book=False, solve=lambda **_kwargs: run)
    assert example.main(project) is run
    assert run.figures is figures
    assert run.channel_runs[0].result is not None
    assert (tmp_path / "stl_export_report.json").exists()
    assert len(list(tmp_path.rglob("*.stl"))) == 2


def test_partial_sweep_exports_available_body_and_reports_missing_part(tmp_path, monkeypatch):
    monkeypatch.setattr(example, "EXPORT_STL", True)
    missing = _part()
    missing.swept_body = None
    run = _run(tmp_path, _channel("GX", _part(), missing))

    example.export_project_stls(run)

    assert len(list(tmp_path.rglob("*.stl"))) == 2
    report = json.loads((tmp_path / "stl_export_report.json").read_text(encoding="utf-8"))
    assert report[0]["status"] == "exported"
    assert report[0]["manufacturing_ready"] is False
    assert report[0]["manufacturing_warnings"][0]["part_index"] == 2
    assert report[0]["manufacturing_warnings"][0]["reasons"] == "no_swept_body"


@pytest.mark.parametrize("error", [OSError("disk full"), ValueError("unexpected mesh format")])
def test_unexpected_export_errors_are_not_silenced(tmp_path, monkeypatch, error):
    monkeypatch.setattr(example, "EXPORT_STL", True)

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(example, "save_swept_bodies_as_stl", fail)
    with pytest.raises(type(error), match=str(error)):
        example.export_project_stls(_run(tmp_path, _channel("GX", _part())))
    assert list(tmp_path.rglob("*_manufacturing_diagnostics.json"))


def test_missing_sweep_has_explicit_report_status(tmp_path, monkeypatch):
    monkeypatch.setattr(example, "EXPORT_STL", True)
    part = _part()
    part.swept_body = None
    run = _run(tmp_path, _channel("GX", part))
    example.export_project_stls(run)
    report = json.loads((tmp_path / "stl_export_report.json").read_text(encoding="utf-8"))
    assert report[0]["status"] == "unavailable"
    assert "No swept bodies" in report[0]["reason"]
    assert not list(tmp_path.rglob("*.stl"))


def test_disabled_export_does_not_create_files(tmp_path, monkeypatch):
    monkeypatch.setattr(example, "EXPORT_STL", False)
    run = _run(tmp_path / "unused", _channel("GX", _part()))
    assert example.export_project_stls(run) == ()
    assert not run.files
    assert not list(tmp_path.iterdir())
