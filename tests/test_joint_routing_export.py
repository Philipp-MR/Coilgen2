"""Joint-route decision exports coexist with manufacturing and lift artifacts."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.manufacturing import save_manufacturing_diagnostics
from coilgen.models import CoilPart, Mesh


def _part():
    return CoilPart(
        mesh=Mesh(
            np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]]),
            np.array([[0, 1, 2]]),
        ),
        manufacturing_diagnostics={"manufacturing_ok": False, "swept_body_created": False},
    )


@pytest.mark.parametrize("status", ["accepted", "baseline_retained", "unsupported_baseline"])
def test_joint_routing_export_preserves_status_reasons_and_part_number(tmp_path, status):
    first, second = _part(), _part()
    second.joint_route_diagnostics = {
        "status": status,
        "reason": "no_valid_improvement" if status != "accepted" else "validated_improvement",
        "candidate_attempts": [
            {"accepted": np.bool_(False), "reasons": ["support_penetration", "physical_field_budget"]}
        ],
        "candidate_count": np.int64(2),
        "field_change_t_per_a": np.array([0.0, 1e-9]),
    }
    original = deepcopy(second.joint_route_diagnostics)

    files = save_manufacturing_diagnostics(SimpleNamespace(coil_parts=[first, second]), tmp_path, prefix="case")

    assert [path.name for path in files] == [
        "case_manufacturing_diagnostics.json",
        "case_manufacturing_diagnostics.csv",
        "case_part002_joint_routing.json",
    ]
    saved = json.loads(files[-1].read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["part_index"] == 2
    assert saved["diagnostics"]["status"] == status
    assert saved["diagnostics"]["reason"] == original["reason"]
    assert saved["diagnostics"]["candidate_attempts"] == [
        {"accepted": False, "reasons": ["support_penetration", "physical_field_budget"]}
    ]
    assert saved["diagnostics"]["candidate_count"] == 2
    assert saved["diagnostics"]["field_change_t_per_a"] == [0.0, 1e-9]
    assert not (tmp_path / "case_part001_joint_routing.json").exists()
    np.testing.assert_array_equal(
        second.joint_route_diagnostics["field_change_t_per_a"], original["field_change_t_per_a"]
    )


def test_joint_routing_export_keeps_manufacturing_and_numeric_lift_outputs(tmp_path):
    part = _part()
    points = np.array([[0.0, 0, 0], [0.01, 0, 0]])
    part.normal_shift_provenance = {
        "schema_version": 2,
        "points_before_m": points,
        "points_after_m": points + [0, 0, 0.001],
        "output_source_segment_indices": np.array([0, 0]),
        "output_source_segment_fractions": np.array([0.0, 1.0]),
        "segment_owners": [{"kind": "inter_group_connector"}],
    }
    part.normal_shift_diagnostics = {"crossing_event_count": 1}
    result = SimpleNamespace(coil_parts=[part])
    baseline = save_manufacturing_diagnostics(result, tmp_path / "baseline")
    part.joint_route_diagnostics = {"status": "baseline_retained", "reason": "physical_collision"}

    files = save_manufacturing_diagnostics(result, tmp_path / "with_joint")

    assert [path.name for path in files] == [
        "coil_body_manufacturing_diagnostics.json",
        "coil_body_manufacturing_diagnostics.csv",
        "coil_body_part001_joint_routing.json",
        "coil_body_part001_normal_shift_provenance.npz",
    ]
    for before, after in zip(baseline[:2], files[:2]):
        assert before.read_bytes() == after.read_bytes()
    with np.load(baseline[-1], allow_pickle=False) as before, np.load(files[-1], allow_pickle=False) as after:
        assert before.files == after.files
        for name in before.files:
            assert after[name].dtype != object
            np.testing.assert_array_equal(after[name], before[name])
        metadata = json.loads(str(after["metadata_json"]))
        assert metadata["schema_version"] == 2
        assert metadata["length_unit"] == "m"
        assert metadata["diagnostics"]["crossing_event_count"] == 1
        assert metadata["segment_owners"] == [{"kind": "inter_group_connector"}]
