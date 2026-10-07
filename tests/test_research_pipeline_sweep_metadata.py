"""Native sweep evidence stays distinct from sheet and fabrication acceptance."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.application.service import DesignService
from coilgen.models import CoilPart, Mesh, SweptBody
from coilgen.research.application import _pipeline_sweep_evidence, _scientific_arrays, run_design
from coilgen.research.requests import ResearchRequest
from coilgen.research.storage import atomic_json, result_metadata_digest


def _mesh(offset=0.0):
    return Mesh(
        np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]) + offset,
        np.asarray([[0, 1, 2]]),
    )


def _config():
    return CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 12, 6, 1, 0, 0, 0),
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_radius=0.045,
        target_region_resolution=8,
        sf_opt_method="constrained_quadratic",
        research_physics={"objective": "power"},
        constrained_quadratic_fallback_policy="strict",
        target_error_tolerance_percent=5,
        levels=6,
        min_point_loop_number=3,
        skip_postprocessing=False,
        skip_sweep=False,
        skip_inductance_calculation=True,
        progress_flag=False,
        plot_flag=False,
    )


@pytest.mark.parametrize(
    "statuses,expected,checked",
    [
        ([], None, 0),
        ([True], True, 1),
        ([False], False, 1),
        ([None], None, 0),
        ([True, None], None, 1),
        ([None, False], False, 1),
    ],
)
def test_only_recorded_body_verdicts_contribute_to_tristate(statuses, expected, checked):
    parts = [
        CoilPart(mesh=_mesh(), swept_bodies=[SweptBody(_mesh(), diagnostics={"manufacturing_ok": value})])
        for value in statuses
    ]

    evidence = _pipeline_sweep_evidence(parts, CoilGenConfig())

    assert evidence["scope"] == "pipeline_sweep_geometry"
    assert evidence["body_count"] == len(statuses)
    assert evidence["checked_body_count"] == checked
    assert evidence["checks_passed"] is expected


def test_skipped_sweep_and_unrecorded_legacy_checks_are_unverified():
    skipped = CoilPart(mesh=_mesh(), manufacturing_diagnostics={"manufacturing_ok": True, "swept_body_created": False})
    old = CoilPart(mesh=_mesh(), swept_body=_mesh(), manufacturing_diagnostics={"watertight": True})
    ambiguous = CoilPart(mesh=_mesh(), swept_body=_mesh(), manufacturing_diagnostics={"manufacturing_ok": 1})

    assert _pipeline_sweep_evidence([skipped], CoilGenConfig(skip_sweep=True))["checks_passed"] is None
    for part in (old, ambiguous):
        evidence = _pipeline_sweep_evidence([part], CoilGenConfig())
        assert evidence["body_count"] == 1 and evidence["checked_body_count"] == 0
        assert evidence["checks_passed"] is None
        assert evidence["bodies"][0]["mesh_solid_ok"] is None


def test_body_identity_matches_export_without_duplicate_aggregate_or_diagnostic_mutation():
    diagnostics = {
        "manufacturing_ok": np.bool_(False),
        "mesh_solid_ok": True,
        "manufacturing_failure_reasons": "folded_sweep_quad",
    }
    route = {
        "route_quality_status": "fail",
        "route_quality_failure_reasons": "connector_bridge_unresolved",
        "nested": {"indices": [4, 7]},
    }
    part = CoilPart(
        mesh=_mesh(),
        swept_body=_mesh(9.0),
        swept_bodies=[SweptBody(_mesh(0.1), contour_index=3, turn_multiplicity=2, diagnostics=diagnostics)],
        manufacturing_diagnostics={"manufacturing_ok": True},
        route_diagnostics=route,
    )
    parts = [CoilPart(mesh=_mesh()), part]
    target = SimpleNamespace(coords=np.zeros((1, 3)), b=np.zeros((1, 3)), suppressed=[False], weights=[1.0])

    evidence = _pipeline_sweep_evidence(parts, CoilGenConfig())
    arrays = _scientific_arrays(parts, target)
    body = evidence["bodies"][0]

    assert evidence["body_count"] == 1 and evidence["checks_passed"] is False
    assert body["body_id"] == "pipeline_part002_body0001"
    assert (body["part_index"], body["body_index"], body["contour_index"]) == (2, 1, 3)
    assert body["source_kind"] == "contour" and body["turn_multiplicity"] == 2
    np.testing.assert_array_equal(arrays[body["vertices_key"]], part.swept_bodies[0].mesh.vertices)
    np.testing.assert_array_equal(arrays[body["faces_key"]], part.swept_bodies[0].mesh.faces)
    assert body["diagnostics"]["manufacturing_failure_reasons"] == "folded_sweep_quad"
    assert body["route_diagnostics"]["route_quality_failure_reasons"] == "connector_bridge_unresolved"
    body["diagnostics"]["manufacturing_ok"] = True
    body["route_diagnostics"]["nested"]["indices"].append(9)
    assert part.swept_bodies[0].diagnostics["manufacturing_ok"] == np.bool_(False)
    assert part.route_diagnostics["nested"]["indices"] == [4, 7]


def test_legacy_actual_body_preserves_recorded_diagnostics_and_export_keys():
    part = CoilPart(
        mesh=_mesh(),
        swept_body=_mesh(0.3),
        manufacturing_diagnostics={
            "manufacturing_ok": False,
            "manufacturing_failure_reasons": "sweep_self_intersection",
        },
    )

    body = _pipeline_sweep_evidence([part], CoilGenConfig())["bodies"][0]

    assert body["manufacturing_ok"] is False
    assert body["diagnostics"] == part.manufacturing_diagnostics
    assert body["vertices_key"] == "swept_0_0_vertices_m" and body["faces_key"] == "swept_0_0_faces"


def test_saved_full_run_retains_native_failure_without_fabrication_review_or_new_gate(tmp_path):
    result = run_design(ResearchRequest.from_config(_config(), fabricate=False, validation_resolution=8), tmp_path)
    record = DesignService.load_run(tmp_path)

    assert record.metadata["pipeline_sweeps"] == result["pipeline_sweeps"]
    evidence = result["pipeline_sweeps"]
    assert evidence["body_count"] > 0 and evidence["checks_passed"] is False
    assert result["summary"]["pipeline_sweep_checks_passed"] is False
    assert result["summary"]["fabrication_review_performed"] is False
    assert result["fabrication"] is None and result["summary"]["scope"] == "continuous_sheet"
    assert result["summary"]["sheet_accepted"] is True
    assert result["summary"]["accepted"] is result["summary"]["sheet_accepted"]
    assert all(row["scope"] == "continuous_sheet" for row in result["constraints"])
    assert any(body["diagnostics"].get("manufacturing_failure_reasons") for body in evidence["bodies"])
    for body in evidence["bodies"]:
        assert body["vertices_key"] in record.arrays and body["faces_key"] in record.arrays
    assert result["summary"]["construction_released"] is False


def test_prepared_result_and_older_saved_results_do_not_acquire_sweep_approval(tmp_path):
    request = ResearchRequest.from_config(replace(_config(), skip_sweep=True), pipeline_stage="prepared")
    result = run_design(request, tmp_path)
    assert result["pipeline_sweeps"]["body_count"] == 0
    assert result["pipeline_sweeps"]["checks_passed"] is None
    assert result["summary"]["pipeline_sweep_checks_passed"] is None
    assert result["summary"]["fabrication_review_performed"] is False

    old = json.loads((tmp_path / "result.json").read_text())
    old.pop("pipeline_sweeps")
    old["summary"].pop("pipeline_sweep_checks_passed")
    old["summary"].pop("fabrication_review_performed")
    old["metadata_sha256"] = result_metadata_digest(old)
    atomic_json(tmp_path / "result.json", old)

    loaded = DesignService.load_run(tmp_path)
    assert loaded.metadata.get("pipeline_sweeps") is None
    assert loaded.metadata["summary"].get("pipeline_sweep_checks_passed") is None
