"""Presentation can borrow immutable evidence without copying unrelated rows."""

from collections.abc import Mapping

import numpy as np
import pytest

from coilgen.application import RunRecord
from coilgen.persistence import json_value


def test_borrowed_metadata_is_recursively_immutable_and_input_owned(tmp_path):
    supplied = {"summary": {"accepted": False}, "winding_steps": [{"points": [1.0, 2.0]}]}
    record = RunRecord(tmp_path, supplied, {})
    view = record.metadata_view
    assert isinstance(view, Mapping)
    assert view is record.metadata_view
    with pytest.raises(TypeError):
        view["summary"]["accepted"] = True
    with pytest.raises(TypeError):
        view["winding_steps"][0]["points"][0] = 99.0
    supplied["summary"]["accepted"] = True
    supplied["winding_steps"][0]["points"][0] = 99.0
    detached = record.metadata
    detached["summary"]["accepted"] = True
    detached["winding_steps"][0]["points"].append(3.0)
    assert view["summary"]["accepted"] is False
    assert view["winding_steps"][0]["points"] == (1.0, 2.0)
    assert record.metadata == {"summary": {"accepted": False}, "winding_steps": [{"points": [1.0, 2.0]}]}


def test_numpy_and_path_metadata_keep_serialized_values_without_mutable_leaves(tmp_path):
    samples = np.array([[1.0, np.nan], [np.inf, 4.0]])
    supplied = {"diagnostic": samples, "path": tmp_path / "run", "count": np.int64(2)}
    expected = json_value(supplied)
    record = RunRecord(tmp_path, supplied, {})
    samples[:] = 99.0
    assert record.metadata == expected
    assert json_value(record.metadata_view) == expected
    assert record.metadata_view["diagnostic"] == ((1.0, None), (None, 4.0))
    with pytest.raises(TypeError):
        record.metadata_view["diagnostic"][0][0] = 99.0


def test_borrowed_small_fields_do_not_traverse_large_saved_diagnostics(tmp_path, monkeypatch):
    record = RunRecord(
        tmp_path, {"summary": {"accepted": False}, "winding_steps": [{"point": i} for i in range(1000)]}, {}
    )

    def copying(_value):
        raise AssertionError("Read-only display attempted to copy full metadata")

    monkeypatch.setattr("coilgen.application.service.json_value", copying)
    assert record.metadata_view["summary"]["accepted"] is False
