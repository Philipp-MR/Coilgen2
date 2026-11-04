"""Live boundary visualization cannot become accepted final result evidence."""

from dataclasses import replace
from hashlib import sha256
import io
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from coilgen.application.presets import preset_config
from coilgen.application.stage_preview import (
    MANIFEST_NAME,
    load_stage_preview,
    publish_evaluation_preview,
    publish_feature_preview,
    publish_stage_preview,
    read_stage_preview_manifest,
)
from coilgen.models import CoilPart, ContourLine, Mesh, TargetField, WirePath
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, PipelineState
from coilgen.progress import DesignCancelled, ProgressReporter, progress_context
from coilgen.research import application as research
from coilgen.research.jobs import _record_completed_boundary
from coilgen.research.requests import ResearchRequest


DIGEST = "a" * 64


def request(stage="routed"):
    return ResearchRequest.from_config(
        preset_config(),
        name="Owned study",
        pipeline_stage=stage,
        fabricate=False,
        project_id="coil-channel",
        project_revision=3,
        project_fingerprint="owned-model-study",
        project_model_fingerprint="owned-model",
    )


def checkpoint(tmp_path, stage="prepared"):
    vertices = np.array([[0.00125, 0, 0], [0, 0.0025, 0], [0, 0, 0.00375]])
    mesh = Mesh(vertices, np.array([[0, 1, 2]]))
    target = TargetField(
        np.array([[0, 0, 0], [0, 0.01, 0]]),
        np.array([[0, 0, 0], [0, 0, 2e-4]]),
        np.array([1, 0.5]),
        np.array([False, True]),
        np.array([0, 1]),
    )
    part = CoilPart(mesh)
    state = PipelineState(parts=[part], target_field=target)
    selected = PipelineStage(stage)
    if selected.index >= 1:
        mesh.stream_function = np.array([0.25, -0.5, 0.75])
        part.stream_function = mesh.stream_function.copy()
        state.combined_mesh, state.sf_b_field = mesh, target.b * 0.99
    if selected.index >= 2:
        part.contour_lines = [ContourLine(vertices.copy(), 0.25)]
    if selected.index >= 3:
        part.wire_path = WirePath(vertices.copy())
        part.swept_body = Mesh(vertices * 0.9, mesh.faces.copy())
    return PipelineCheckpoint(selected, state, preset_config(), tmp_path, fingerprints={selected.value: "f" * 64})


def identity(value=None):
    value = value or request()
    return dict(
        expected_request_sha256=DIGEST,
        expected_project_id=value.project_id,
        expected_project_fingerprint=value.project_fingerprint,
        expected_project_revision=value.project_revision,
    )


def manifest(tmp_path):
    return json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))


def rewrite(tmp_path, value):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("stage", [item.value for item in PipelineStage])
def test_snapshot_contains_exact_recorded_si_arrays_and_no_acceptance(tmp_path, stage):
    original = checkpoint(tmp_path, stage)
    publish_stage_preview(tmp_path, original, request("evaluated"), DIGEST)
    loaded = load_stage_preview(tmp_path, **identity())
    expected = original.scientific_arrays()
    assert loaded.metadata["pipeline_stage"] == stage
    assert loaded.metadata["source"] == "live_stage_preview"
    assert loaded.metadata["acceptance_claim"] is False
    assert "accepted" not in loaded.metadata and "artifacts" not in loaded.metadata
    assert loaded.arrays.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(loaded.arrays[key], expected[key])
        assert not loaded.arrays[key].flags.writeable
    loaded.metadata["completed_stages"].clear()
    assert loaded.metadata["completed_stages"]
    assert not (tmp_path / "result.json").exists()


def test_scientific_array_accessor_does_not_copy_operators_or_share_mutable_data(tmp_path):
    class NeverCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("The display accessor copied a numerical operator.")

    original = checkpoint(tmp_path, "routed")
    original._state.parts[0].sensitivity_matrix = NeverCopy()
    detached = original.scientific_arrays()
    detached["target_field_T"][:] = 123
    detached["route_0_points_m"][:] = 456
    assert not np.all(original.scientific_arrays()["target_field_T"] == 123)
    assert not np.all(original.scientific_arrays()["route_0_points_m"] == 456)
    with pytest.raises(ValueError, match="permitted size"):
        original.scientific_arrays(max_bytes=1)


def test_feature_snapshots_publish_cumulative_geometry_before_routed_checkpoint(tmp_path):
    original = checkpoint(tmp_path, "routed")
    state = original.state
    part = state.parts[0]
    part.fabrication_path = SimpleNamespace(wire_points_m=part.wire_path.v + 1e-4)

    # The live writer must only visit scientific arrays, even for large operators.
    class NeverCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("Unexpected solver state copy")

    part.sensitivity_matrix = NeverCopy()
    keys = ("route_0_points_m", "fabrication_0_points_m", "swept_0_0_vertices_m")
    expected = (part.wire_path.v, part.fabrication_path.wire_points_m, part.swept_body.vertices)
    publish_stage_preview(tmp_path, checkpoint(tmp_path, "contours"), request(), DIGEST)
    for index, feature in enumerate(("interconnections", "realized_centerlines", "sweep")):
        publish_feature_preview(tmp_path, state, feature, request(), DIGEST)
        loaded = load_stage_preview(tmp_path, **identity())
        assert loaded.metadata["preview_feature"] == feature
        assert loaded.metadata["scope"] == "completed_feature_preview"
        assert loaded.metadata["pipeline_stage"] == "contours"
        assert loaded.metadata["completed_stages"] == ["prepared", "sheet", "contours"]
        assert loaded.metadata["acceptance_claim"] is False
        assert loaded.metadata["array_file"] == f"stage_previews/{feature}.npz"
        for key, value in zip(keys[: index + 1], expected):
            np.testing.assert_array_equal(loaded.arrays[key], value)
        assert not set(keys[index + 1 :]).intersection(loaded.arrays)
        assert "contour_0_0_points_m" in loaded.arrays and "stream_function_A" in loaded.arrays
    publish_stage_preview(tmp_path, original, request(), DIGEST)
    assert "preview_feature" not in load_stage_preview(tmp_path, **identity()).metadata


def test_missing_or_malformed_feature_cannot_advance_announcement(tmp_path):
    state = checkpoint(tmp_path, "contours").state
    publish_stage_preview(tmp_path, checkpoint(tmp_path, "contours"), request(), DIGEST)
    with pytest.raises(ValueError, match="no recorded geometry"):
        publish_feature_preview(tmp_path, state, "interconnections", request(), DIGEST)
    assert "preview_feature" not in manifest(tmp_path)
    state.parts[0].wire_path = WirePath(state.parts[0].mesh.vertices.copy())
    publish_feature_preview(tmp_path, state, "interconnections", request(), DIGEST)
    arrays = dict(load_stage_preview(tmp_path, **identity()).arrays)
    arrays["fabrication_0_points_m"] = arrays["route_0_points_m"]
    _replace_arrays(tmp_path, arrays)
    with pytest.raises(ValueError, match="later feature"):
        load_stage_preview(tmp_path, **identity())
    changed = manifest(tmp_path)
    changed["pipeline_stage"] = "routed"
    rewrite(tmp_path, changed)
    with pytest.raises(ValueError, match="feature or scientific scope"):
        read_stage_preview_manifest(tmp_path, **identity())


def evaluated_arrays(tmp_path, *, routed=True):
    completed = checkpoint(tmp_path, "routed")
    state = completed.state
    state.parts[0].contour_step = 2.5
    completed.config.field_evaluation_current_source = "target_fit"
    result = SimpleNamespace(
        coil_parts=state.parts,
        target_field=state.target_field,
        input_data=completed.config,
        needed_current_layout=3.25,
        field_by_unconnected_loops=state.target_field.b * 0.98,
        field_by_layout=state.target_field.b * 0.96,
        field_loops_per1Amp=state.target_field.b * 0.30,
        field_layout_per1Amp=state.target_field.b * 0.29,
    )
    result.input_data.research_physics = None
    result.input_data.skip_postprocessing = not routed
    wire, evidence = research._wire_field_evidence(result)
    return {**completed.scientific_arrays(), **wire}, evidence


def test_evaluation_features_preserve_exact_sampling_currents_and_cumulative_plot_data(tmp_path):
    arrays, wire = evaluated_arrays(tmp_path)
    # These result-only arrays are deliberately outside the plotting contract.
    arrays["review_conductor_0_owner_kind"] = np.array(["connector"])
    coordinates = arrays["target_points_m"]
    independent = np.vstack([coordinates, [0.011, 0.002, 0]])
    features = {
        "evaluated_winding": {},
        "independent_field": dict(
            validation_points_m=independent, validation_field_T=independent * 0.2, validation_target_T=independent * 0.3
        ),
        "review_conductor": dict(
            review_conductor_0_points_m=coordinates * 1.2, review_electrical_0_points_m=coordinates * 1.3
        ),
        "fabricated_field": dict(fabricated_points_m=coordinates.copy(), fabricated_field_T=coordinates * 0.1),
        "winding_validation": dict(
            winding_validation_points_m=independent[::-1],
            winding_validation_field_T=independent * 0.4,
            winding_validation_target_T=independent * 0.32,
        ),
    }
    for feature, additions in features.items():
        arrays.update(additions)
        policy = "original_contour_current_no_refit" if "fabricated_field_T" in arrays else None
        publish_evaluation_preview(
            tmp_path, arrays, feature, request("evaluated"), DIGEST, wire_fields=wire, fabricated_current_source=policy
        )
        preview = load_stage_preview(tmp_path, **identity())
        assert preview.metadata["completed_stages"] == [stage.value for stage in PipelineStage]
        assert preview.metadata["pipeline_stage"] == "evaluated"
        assert preview.metadata["preview_feature"] == feature and preview.metadata["acceptance_claim"] is False
        assert preview.metadata["wire_fields"] == wire
        assert preview.metadata.get("fabricated_physics") == ({"current_source": policy} if policy else None)
        assert preview.arrays.keys() == arrays.keys() - {"review_conductor_0_owner_kind"}
        for key, value in preview.arrays.items():
            np.testing.assert_array_equal(value, arrays[key])
            assert not value.flags.writeable


def test_contour_only_winding_preview_does_not_advertise_a_connected_field(tmp_path):
    arrays, wire = evaluated_arrays(tmp_path, routed=False)
    publish_evaluation_preview(tmp_path, arrays, "evaluated_winding", request("evaluated"), DIGEST, wire_fields=wire)
    preview = load_stage_preview(tmp_path, **identity())
    assert "contour_field_T" in preview.arrays and "routed_field_T" not in preview.arrays
    assert not preview.metadata["wire_fields"]["routed"]["available"]


@pytest.mark.parametrize("invalid", ["current", "refit", "grid", "geometry", "scope", "future", "unreported"])
def test_loader_rejects_mismatched_wire_evidence_and_sampling(tmp_path, invalid):
    arrays, wire = evaluated_arrays(tmp_path)
    publish_evaluation_preview(tmp_path, arrays, "evaluated_winding", request("evaluated"), DIGEST, wire_fields=wire)
    info = manifest(tmp_path)
    if invalid == "current":
        info["wire_fields"]["contour"]["current_A"] = None
    elif invalid == "refit":
        info["wire_fields"]["routed"]["at_stream_current"]["current_refitted"] = True
    elif invalid == "scope":
        info["wire_fields"]["routed"]["acceptance_claim"] = True
    elif invalid == "grid":
        arrays["routed_field_T"] = arrays["routed_field_T"][:1]
    elif invalid == "geometry":
        arrays["evaluated_winding_0_points_m"] = arrays["evaluated_winding_0_points_m"] + 0.01
    elif invalid == "unreported":
        info.pop("wire_fields")
    else:
        arrays["review_electrical_0_points_m"] = arrays["target_points_m"]
    rewrite(tmp_path, info)
    _replace_arrays(tmp_path, arrays)
    with pytest.raises(ValueError):
        load_stage_preview(tmp_path, **identity())


@pytest.mark.parametrize("feature", ["independent_field", "fabricated_field", "winding_validation"])
def test_incomplete_field_or_unknown_current_preserves_last_preview(tmp_path, feature):
    original = checkpoint(tmp_path, "evaluated")
    publish_stage_preview(tmp_path, original, request("evaluated"), DIGEST)
    arrays = original.scientific_arrays()
    prefix = {"independent_field": "validation", "fabricated_field": "fabricated"}.get(feature, feature)
    arrays[prefix + "_field_T"] = arrays["sheet_field_T"]
    if feature == "fabricated_field":
        arrays["fabricated_points_m"] = arrays["target_points_m"] + 1e-5
    with pytest.raises(ValueError):
        publish_evaluation_preview(tmp_path, arrays, feature, request("evaluated"), DIGEST)
    if feature == "fabricated_field":
        with pytest.raises(ValueError):
            publish_evaluation_preview(
                tmp_path,
                arrays,
                feature,
                request("evaluated"),
                DIGEST,
                fabricated_current_source="original_contour_current_no_refit",
            )
    assert "preview_feature" not in load_stage_preview(tmp_path, **identity()).metadata


def test_no_publication_and_partially_written_npz_leave_previous_boundary_visible(tmp_path):
    assert load_stage_preview(tmp_path, **identity()) is None
    publish_stage_preview(tmp_path, checkpoint(tmp_path, "prepared"), request(), DIGEST)
    (tmp_path / "stage_previews" / ".sheet.npz.pending.tmp").write_bytes(b"incomplete")
    loaded = load_stage_preview(tmp_path, **identity())
    assert loaded.metadata["pipeline_stage"] == "prepared"
    assert "stream_function_A" not in loaded.arrays


def test_new_boundary_is_atomic_and_an_announced_stage_file_is_immutable(tmp_path, monkeypatch):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    first = (tmp_path / MANIFEST_NAME).read_bytes()
    import coilgen.application.stage_preview as snapshots

    real_replace = snapshots.os.replace

    def replacement(source, destination):
        if str(destination).endswith("sheet.npz"):
            assert (tmp_path / MANIFEST_NAME).read_bytes() == first
            assert load_stage_preview(tmp_path, **identity()).metadata["pipeline_stage"] == "prepared"
        return real_replace(source, destination)

    monkeypatch.setattr(snapshots.os, "replace", replacement)
    publish_stage_preview(tmp_path, checkpoint(tmp_path, "sheet"), request(), DIGEST)
    assert load_stage_preview(tmp_path, **identity()).metadata["pipeline_stage"] == "sheet"
    with pytest.raises(ValueError, match="already published"):
        publish_stage_preview(tmp_path, checkpoint(tmp_path, "sheet"), request(), DIGEST)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_request_sha256", "b" * 64),
        ("expected_project_id", "other-channel"),
        ("expected_project_fingerprint", "older-model"),
        ("expected_project_revision", 4),
    ],
)
def test_snapshot_from_another_submission_or_revision_is_rejected(tmp_path, field, value):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    expected = {**identity(), field: value}
    with pytest.raises(ValueError, match="submitted job identity"):
        load_stage_preview(tmp_path, **expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "coilgen.live-stage-preview/2"),
        ("acceptance_claim", True),
        ("coordinate_units", "mm"),
        ("pipeline_stage", "fabricated"),
        ("completed_stages", ["routed"]),
        ("array_file", "../scientific_result.npz"),
        ("array_file", "stage_previews/sheet.npz"),
    ],
)
def test_manifest_cannot_change_scientific_scope_or_escape_job(tmp_path, field, value):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    changed = manifest(tmp_path)
    changed[field] = value
    rewrite(tmp_path, changed)
    with pytest.raises(ValueError):
        read_stage_preview_manifest(tmp_path, **identity())


def test_hash_failure_does_not_load_changed_arrays(tmp_path):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    path = tmp_path / manifest(tmp_path)["array_file"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity"):
        load_stage_preview(tmp_path, **identity())


@pytest.mark.parametrize("component", ["Bx", "By", "Bz", "BdotB0", "vector"])
def test_requested_target_component_is_retained_for_presentation(tmp_path, component):
    value = request()
    value = replace(value, config={**value.config, "target_field_component": component})
    publish_stage_preview(tmp_path, checkpoint(tmp_path), value, DIGEST)
    assert load_stage_preview(tmp_path, **identity(value)).metadata["target_field_component"] == component


def test_legacy_snapshot_without_component_uses_bz_and_unknown_component_fails(tmp_path):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    value = manifest(tmp_path)
    value.pop("target_field_component")
    rewrite(tmp_path, value)
    assert load_stage_preview(tmp_path, **identity()).metadata["target_field_component"] == "Bz"
    value["target_field_component"] = "unknown_projection"
    rewrite(tmp_path, value)
    with pytest.raises(ValueError, match="Unknown field component"):
        load_stage_preview(tmp_path, **identity())


def test_hash_consistent_but_malformed_npz_reports_value_error_to_background_loader(tmp_path):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    value = manifest(tmp_path)
    path = tmp_path / value["array_file"]
    path.write_bytes(b"not a zip archive")
    value["array_sha256"] = sha256(path.read_bytes()).hexdigest()
    rewrite(tmp_path, value)
    with pytest.raises(ValueError, match="Malformed.*archive"):
        load_stage_preview(tmp_path, **identity())


def test_nonfinite_or_opaque_arrays_are_rejected_before_publication(tmp_path):
    original = checkpoint(tmp_path)
    original._state.target_field.b[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite real"):
        publish_stage_preview(tmp_path, original, request(), DIGEST)
    assert not (tmp_path / MANIFEST_NAME).exists()


def test_archive_and_manifest_size_are_bounded(tmp_path):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    with pytest.raises(ValueError, match="permitted size"):
        load_stage_preview(tmp_path, **identity(), max_bytes=8)
    with pytest.raises(ValueError, match="byte limit"):
        load_stage_preview(tmp_path, **identity(), max_bytes=True)
    (tmp_path / MANIFEST_NAME).write_text(" " * 65537)
    with pytest.raises(ValueError, match="manifest exceeds"):
        read_stage_preview_manifest(tmp_path, **identity())


def _replace_arrays(tmp_path, arrays):
    changed = manifest(tmp_path)
    path = tmp_path / changed["array_file"]
    np.savez_compressed(path, **arrays)
    changed["array_sha256"] = sha256(path.read_bytes()).hexdigest()
    rewrite(tmp_path, changed)


def test_current_density_preview_preserves_recorded_face_vectors_in_si(tmp_path):
    original = checkpoint(tmp_path, "sheet")
    density = np.array([[3.0, -4.0, 0.0]])
    original._state.parts[0].current_density = density.copy()
    publish_stage_preview(tmp_path, original, request(), DIGEST)
    loaded = load_stage_preview(tmp_path, **identity())
    np.testing.assert_array_equal(loaded.arrays["sheet_current_density_A_per_m"], density)
    mesh = original._state.parts[0].mesh
    np.testing.assert_array_equal(
        loaded.arrays["sheet_current_density_points_m"], mesh.vertices[mesh.faces].mean(axis=1)
    )


@pytest.mark.parametrize(
    "invalid", ["missing_points", "missing_vectors", "count", "shape", "nonfinite", "boolean", "prepared"]
)
def test_current_density_preview_rejects_unpaired_or_incompatible_face_data(tmp_path, invalid):
    original = checkpoint(tmp_path, "prepared" if invalid == "prepared" else "sheet")
    publish_stage_preview(tmp_path, original, request(), DIGEST)
    arrays = original.scientific_arrays()
    arrays.update(sheet_current_density_points_m=np.zeros((1, 3)), sheet_current_density_A_per_m=np.ones((1, 3)))
    if invalid == "missing_points":
        del arrays["sheet_current_density_points_m"]
    elif invalid == "missing_vectors":
        del arrays["sheet_current_density_A_per_m"]
    elif invalid == "count":
        arrays["sheet_current_density_A_per_m"] = np.ones((2, 3))
    elif invalid == "shape":
        arrays["sheet_current_density_A_per_m"] = np.ones(3)
    elif invalid == "nonfinite":
        arrays["sheet_current_density_A_per_m"][0, 0] = np.nan
    elif invalid == "boolean":
        arrays["sheet_current_density_A_per_m"] = np.ones((1, 3), dtype=bool)
    _replace_arrays(tmp_path, arrays)
    with pytest.raises(ValueError):
        load_stage_preview(tmp_path, **identity())


@pytest.mark.parametrize(
    "key,values",
    [
        ("unknown_operator", np.zeros((3, 3))),
        ("faces", np.array([[0, 1, 999]])),
        ("target_field_T", np.zeros((7, 3))),
        ("route_0_points_m", np.zeros((3, 3))),
        ("target_field_T", np.array([[None] * 3] * 2, dtype=object)),
    ],
)
def test_loader_rejects_unknown_arrays_invalid_dimensions_and_later_geometry(tmp_path, key, values):
    original = checkpoint(tmp_path)
    publish_stage_preview(tmp_path, original, request(), DIGEST)
    arrays = original.scientific_arrays()
    arrays[key] = values
    _replace_arrays(tmp_path, arrays)
    with pytest.raises(ValueError):
        load_stage_preview(tmp_path, **identity())


def test_duplicate_numerical_archive_entries_are_rejected(tmp_path):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    info = manifest(tmp_path)
    path = tmp_path / info["array_file"]
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((1, 3)))
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("vertices_m.npy", buffer.getvalue())
            archive.writestr("vertices_m.npy", buffer.getvalue())
    info["array_sha256"] = sha256(path.read_bytes()).hexdigest()
    rewrite(tmp_path, info)
    with pytest.raises(ValueError, match="Malformed.*archive"):
        load_stage_preview(tmp_path, **identity())


def test_forged_array_shape_is_rejected_before_numpy_can_allocate(tmp_path, monkeypatch):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    info = manifest(tmp_path)
    path = tmp_path / info["array_file"]
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer, {"descr": "<f8", "fortran_order": False, "shape": (10**12, 3)})
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("vertices_m.npy", buffer.getvalue())
    info["array_sha256"] = sha256(path.read_bytes()).hexdigest()
    rewrite(tmp_path, info)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("Allocated an unchecked shape"))
    with pytest.raises(ValueError, match="recorded storage"):
        load_stage_preview(tmp_path, **identity())


def test_manifest_failure_leaves_previous_stage_usable(tmp_path, monkeypatch):
    publish_stage_preview(tmp_path, checkpoint(tmp_path), request(), DIGEST)
    import coilgen.application.stage_preview as snapshots

    def failure(*_args):
        raise OSError("Manifest publication unavailable")

    monkeypatch.setattr(snapshots, "atomic_json", failure)
    with pytest.raises(OSError, match="publication unavailable"):
        publish_stage_preview(tmp_path, checkpoint(tmp_path, "sheet"), request(), DIGEST)
    assert load_stage_preview(tmp_path, **identity()).metadata["pipeline_stage"] == "prepared"


def test_prepared_arrays_cannot_be_promoted_to_a_solved_stage(tmp_path):
    publish_stage_preview(tmp_path, checkpoint(tmp_path, "sheet"), request(), DIGEST)
    _replace_arrays(tmp_path, checkpoint(tmp_path).scientific_arrays())
    with pytest.raises(ValueError, match="missing its stream function"):
        load_stage_preview(tmp_path, **identity())


def test_boundary_events_are_structured_monotonic_and_independent_of_percent():
    events = []
    reporter = ProgressReporter(enabled=False)
    with progress_context(events.append):
        reporter.boundary("sheet", preview=MANIFEST_NAME)
        reporter.report(100, "Subtask finished")
        reporter.boundary("prepared")
        reporter.boundary("contours", preview_error="Preview filesystem unavailable")
    state = {"pipeline_stage": "evaluated", "progress": 0.3}
    for event in events:
        _record_completed_boundary(state, event)
    assert state["completed_stage"] == "contours"
    assert state["completed_stages"] == ["prepared", "sheet", "contours"]
    assert state["stage_preview_stage"] == "sheet"
    assert state["stage_preview_error"] == "Preview filesystem unavailable"
    assert state["progress"] == 0.3
    with pytest.raises(ValueError):
        reporter.boundary("fabricated")
    with progress_context(events.append, lambda: True):
        with pytest.raises(DesignCancelled):
            reporter.boundary("routed")


@pytest.mark.parametrize(
    "event",
    [
        {"label": "Routing", "completed_stage": "sheet", "completed_stages": ["prepared", "sheet"]},
        {"label": "CoilGen", "completed_stage": "evaluated", "completed_stages": ["evaluated"]},
        {
            "label": "CoilGen",
            "completed_stage": "routed",
            "completed_stages": ["prepared", "sheet", "contours", "routed"],
        },
        {"label": "CoilGen", "percent": 100, "message": "optimized stream function"},
    ],
)
def test_subtasks_or_beyond_requested_boundary_cannot_claim_completion(event):
    state = {"pipeline_stage": "sheet", "progress": 0.75}
    _record_completed_boundary(state, event)
    assert state == {"pipeline_stage": "sheet", "progress": 0.75}


@pytest.mark.parametrize("error_type", [PermissionError, MemoryError, OverflowError, AttributeError, Exception])
def test_optional_preview_failure_does_not_change_result_or_retained_checkpoints(tmp_path, monkeypatch, error_type):
    completed = checkpoint(tmp_path)
    callbacks = []

    class FakePipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_stage(self, stage, resumed, *, checkpoint_callback, feature_callback, **_callbacks):
            callbacks.append(checkpoint_callback)
            feature_callback("interconnections", completed.state)
            checkpoint_callback(completed)
            return completed

    def storage_failure(*_args, **_kwargs):
        raise error_type("Optional preview failed")

    result = {"scientific": "unchanged"}
    monkeypatch.setattr(research, "CoilGenPipeline", FakePipeline)
    monkeypatch.setattr(research, "publish_stage_preview", storage_failure)
    monkeypatch.setattr(research, "publish_feature_preview", storage_failure)
    monkeypatch.setattr(research, "_save_partial_result", lambda *_args: result)
    events = []
    with progress_context(events.append):
        actual = research.run_design(request("prepared"), tmp_path / "run")
    assert actual == result and len(callbacks) == 1
    assert (tmp_path / "run" / "checkpoints" / "prepared.cgstage").is_file()
    assert (tmp_path / "run" / "pipeline_checkpoint.cgstage").is_file()
    boundary = next(event for event in events if event.get("completed_stage") == "prepared")
    assert "stage_preview" not in boundary and error_type.__name__ in boundary["stage_preview_error"]
    assert any(
        f"Live interconnections preview unavailable: {error_type.__name__}" in event["message"] for event in events
    )


def test_same_stage_resume_announces_only_after_pipeline_compatibility_check(tmp_path, monkeypatch):
    completed = checkpoint(tmp_path)
    path = completed.save(tmp_path / "resume.cgstage")
    checked = []

    class FakePipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_stage(self, stage, resumed, *, resume_callback, **_callbacks):
            assert not (tmp_path / "run" / MANIFEST_NAME).exists()
            checked.append(True)
            resume_callback(resumed)
            return resumed

    monkeypatch.setattr(research, "CoilGenPipeline", FakePipeline)
    monkeypatch.setattr(research, "_save_partial_result", lambda *_args: {"resumed": True})
    research.run_design(replace(request("prepared"), resume_checkpoint=str(path)), tmp_path / "run")
    assert checked and manifest(tmp_path / "run")["pipeline_stage"] == "prepared"


def test_incompatible_resume_is_not_announced(tmp_path, monkeypatch):
    path = checkpoint(tmp_path).save(tmp_path / "resume.cgstage")

    class FakePipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_stage(self, *_args, **_kwargs):
            raise ValueError("Cannot reuse prepared checkpoint")

    monkeypatch.setattr(research, "CoilGenPipeline", FakePipeline)
    with pytest.raises(ValueError, match="Cannot reuse"):
        research.run_design(replace(request("prepared"), resume_checkpoint=str(path)), tmp_path / "run")
    assert not (tmp_path / "run" / MANIFEST_NAME).exists()
