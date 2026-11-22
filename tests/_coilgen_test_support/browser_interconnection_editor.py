"""Shared fixtures for browser interconnection editor."""

from types import SimpleNamespace
import numpy as np
import pytest
from coilgen.application.interconnection_editor import WindingEditorSession
from coilgen.models import TargetField
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, PipelineState
from coilgen.web.interconnection_editor import BrowserWindingEditor
from _coilgen_test_support.interconnection_editor import routed as routed


class State(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def flush(self):
        pass


@pytest.fixture
def browser_editor(routed, tmp_path):
    coords = np.array([[0, 0, 0.025], [0.003, 0, 0.025]])
    target = TargetField(coords, np.full_like(coords, 1e-6), np.ones(2), np.zeros(2, bool), np.zeros(2, int))
    state = PipelineState(parts=[routed.part], target_field=target, combined_mesh=routed.part.mesh, sf_b_field=target.b)
    checkpoint = PipelineCheckpoint(PipelineStage.ROUTED, state, routed.config, tmp_path)
    model = SimpleNamespace(id="project", config=checkpoint.config)
    applied = []
    browser = SimpleNamespace(
        server=SimpleNamespace(state=State()),
        session=SimpleNamespace(
            workspace=tmp_path, project=model, active_channel_id="channel", apply_config=applied.append
        ),
        exports=tmp_path / "exports",
        _advanced_has_edits=lambda: False,
        record=None,
    )
    browser.exports.mkdir()
    adapter = BrowserWindingEditor(browser)
    adapter.session = WindingEditorSession(checkpoint)
    adapter.owner = adapter._owner()
    adapter.saved_identity = adapter.session.identity
    adapter.refresh()
    return adapter, applied


def command(adapter, kind, **kwargs):
    return adapter.action(dict(kind=kind, identity=adapter.session.identity, circuit=adapter.circuit, **kwargs))
