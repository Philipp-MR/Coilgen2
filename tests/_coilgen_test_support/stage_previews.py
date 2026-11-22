"""Shared fixtures for stage previews."""

import numpy as np
from coilgen.application.presets import preset_config
from coilgen.models import CoilPart, ContourLine, Mesh, TargetField, WirePath
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, PipelineState
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
