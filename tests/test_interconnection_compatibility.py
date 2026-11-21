"""Existing designs remain readable when the optional winding editor is added."""

from hashlib import sha256
import json
import zipfile

import numpy as np
import pytest

from coilgen.application.project import ProjectDocument
from coilgen.pipeline import CoilGenPipeline
from coilgen.pipeline_stages import PipelineCheckpoint
from test_pipeline_stages import config


def test_pre_editor_project_preserves_identity_and_requires_other_settings():
    project = ProjectDocument.from_config(config())
    value = project.to_dict()
    value["config"].pop("routing_editor_plan")
    restored = ProjectDocument.from_dict(value)
    assert restored.config.routing_editor_plan is None
    assert restored.fingerprint == project.fingerprint
    assert restored.model_fingerprint == project.model_fingerprint
    value["config"].pop("levels")
    with pytest.raises(ValueError, match="every resolved"):
        ProjectDocument.from_dict(value)


@pytest.mark.parametrize("missing_unrelated", [False, True])
def test_pre_editor_checkpoint_migrates_without_rewriting_evidence(tmp_path, missing_unrelated):
    original = CoilGenPipeline(config()).run_stage("prepared")
    path = original.save(tmp_path / "pre-editor.cgstage")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        data = archive.read("arrays.npz")
    manifest["schema"] = "coilgen.pipeline-checkpoint/8"

    def remove_editor_fields(value):
        if isinstance(value, list):
            for item in value:
                remove_editor_fields(item)
        elif isinstance(value, dict):
            if value.get("$class") == "CoilGenConfig":
                value["fields"].pop("routing_editor_plan")
                if missing_unrelated:
                    value["fields"].pop("levels")
            if value.get("$class") == "CoilPart":
                value["fields"].pop("routing_editor_baseline")
            for item in value.values():
                remove_editor_fields(item)

    remove_editor_fields(manifest["payload"])
    manifest["fingerprints"] = {stage: "historical-engine" for stage in manifest["fingerprints"]}
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = sha256(
        json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", data)
    if missing_unrelated:
        with pytest.raises(ValueError, match="fields.*schema"):
            PipelineCheckpoint.load(path)
        return
    restored = PipelineCheckpoint.load(path)
    assert restored.config.routing_editor_plan is None
    assert restored.parts[0].routing_editor_baseline is None
    np.testing.assert_array_equal(restored.parts[0].mesh.vertices, original.parts[0].mesh.vertices)
    with pytest.raises(ValueError, match="Cannot reuse prepared"):
        CoilGenPipeline(restored.config).run_stage("sheet", restored)
