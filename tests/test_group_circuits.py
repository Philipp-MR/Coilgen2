"""Directed membership, independent circuits, original-current fields and exports."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import csv
import re

import numpy as np
import pytest

from _group_routing_cases import make_group_routing_case
from coilgen.application.group_editor import GroupEditor, draw_group_map
from coilgen.evaluation import evaluate_field_errors
from coilgen.group_circuits import (
    default_group_plan,
    group_snapshot,
    prepare_group_circuits,
    validate_directed_circuits,
    validate_group_plan,
)
from coilgen.inductance import calculate_inductance_by_coil_layout
from coilgen.layout import interconnect_among_groups, interconnect_within_groups
from coilgen.manufacturing import save_wire_trace_as_csv, save_wire_trace_as_step
from coilgen.models import RoutedCircuit, TargetField
from coilgen.physics import biot_savart_wire_path


def setup(group_count=2, mode="per_group"):
    case = make_group_routing_case(group_count=group_count)
    config = replace(
        case.config,
        joint_group_routing=False,
        skip_sweep=True,
        skip_inductance_calculation=False,
        field_evaluation_current_source="stream_function",
        conductor_cross_section_width=0.0003,
        conductor_cross_section_height=0.0003,
    )
    snapshot = group_snapshot([case.part])
    plan = default_group_plan(snapshot, config)
    plan["mode"] = mode
    return case, replace(config, routing_group_plan=plan)


def route(case, config):
    parts = prepare_group_circuits([deepcopy(case.part)], config)
    interconnect_within_groups(parts, config)
    interconnect_among_groups(parts, config)
    validate_directed_circuits(parts)
    return parts


def test_default_bypasses_partition_and_explicit_auto_plan_preserves_routes():
    case, config = setup(mode="existing")
    default = replace(config, routing_group_plan=None)
    original = [deepcopy(case.part)]
    assert prepare_group_circuits(original, default) is original
    interconnect_within_groups(original, default)
    interconnect_among_groups(original, default)
    actual = route(case, config)
    np.testing.assert_array_equal(actual[0].wire_path.v, original[0].wire_path.v)
    assert len(actual) == 1


def test_default_cut_assignments_retain_global_and_per_support_legacy_indices():
    case, config = setup(mode="existing")
    snapshot = group_snapshot([case.part, deepcopy(case.part)])
    config = replace(config, force_cut_selection=("high", "low", "low", "high"))
    plan = default_group_plan(snapshot, replace(config, routing_circuit_scope="single"))
    assert [row["cut"] for row in plan["groups"]] == ["high", "low", "low", "high"]
    plan = default_group_plan(snapshot, replace(config, routing_circuit_scope="per_part"))
    assert [row["cut"] for row in plan["groups"]] == ["high", "low", "high", "low"]


def test_independent_groups_cover_each_source_once_and_never_get_fictional_joins():
    case, config = setup(group_count=3)
    before = [loop.v.copy() for loop in case.part.contour_lines]
    parts = route(case, config)
    assert len(parts) == 3
    assert all(isinstance(part, RoutedCircuit) for part in parts)
    assert [part.source_group_keys for part in parts] == [("s0:g0",), ("s0:g1",), ("s0:g2",)]
    assert sum(len(part.contour_lines) for part in parts) == len(before)
    assert all(len(part.opening_cuts_among_groups) == 0 for part in parts)
    for actual, expected in zip(case.part.contour_lines, before):
        np.testing.assert_array_equal(actual.v, expected)


def test_selected_subset_does_not_drop_unselected_groups():
    case, config = setup(group_count=3, mode="custom")
    plan = deepcopy(config.routing_group_plan)
    for index, group in enumerate(plan["groups"]):
        group["circuit"] = "selected" if index in {0, 2} else "separate"
    parts = route(case, replace(config, routing_group_plan=plan))
    assert {part.circuit_id: part.source_group_keys for part in parts} == {
        "selected": ("s0:g0", "s0:g2"),
        "separate": ("s0:g1",),
    }
    assert sum(len(part.contour_lines) for part in parts) == 6


@pytest.mark.parametrize("order", ["spiral_in", "spiral_out"])
def test_manual_spiral_order_retains_source_circulation(order):
    from coilgen.group_routing import GroupRoutingLimits, propose_joint_group_routes
    from coilgen.layout import _group_route_variants

    case, config = setup(mode="existing")
    plan = deepcopy(config.routing_group_plan)
    for row in plan["groups"]:
        row["spiral"] = order
    config = replace(config, routing_group_plan=plan)
    parts = route(case, config)
    assert all(
        [variant.name for variant in _group_route_variants(group)] == [order] for group in parts[0].connected_groups
    )
    proposals = propose_joint_group_routes(parts[0], config, limits=GroupRoutingLimits(return_fractions=(0.0,)))
    assert proposals.candidates
    assert all(all(name.startswith(order) for name in candidate.variant_names) for candidate in proposals.candidates)


def test_plan_is_bound_to_exact_geometry_and_complete_unique_membership():
    case, config = setup()
    snapshot = group_snapshot([case.part])
    duplicate = deepcopy(config.routing_group_plan)
    duplicate["groups"].append(duplicate["groups"][0])
    with pytest.raises(ValueError, match="exactly one"):
        validate_group_plan(duplicate, snapshot)
    missing = deepcopy(config.routing_group_plan)
    missing["groups"].pop()
    with pytest.raises(ValueError, match="exactly once"):
        validate_group_plan(missing, snapshot)
    moved = deepcopy(case.part)
    moved.contour_lines[0].v[0, 0] += 1e-9
    with pytest.raises(ValueError, match="changed"):
        prepare_group_circuits([moved], config)


def test_group_identity_is_invariant_to_checkpoint_scalar_encoding():
    case, _config = setup()
    before = group_snapshot([case.part])
    case.part.contour_step = np.float64(case.part.contour_step)
    for loop in case.part.contour_lines:
        loop.potential = np.float64(loop.potential)
        loop.current_orientation = np.float64(loop.current_orientation)
    assert group_snapshot([case.part]) == before


def test_fields_sum_independent_circuits_at_one_original_current_and_inductance_is_a_matrix():
    case, config = setup()
    parts = route(case, config)
    coords = np.array([[0.0, 0.0, 0.03], [0.01, 0.0, 0.02], [-0.02, 0.01, 0.025]])
    expected = sum((biot_savart_wire_path(part.wire_path.v, coords) for part in parts), start=np.zeros((3, 3)))
    target = TargetField(coords, expected.copy(), np.ones(3), np.zeros(3, bool), np.ones(3, int))
    fields = evaluate_field_errors(parts, config, target, expected)
    np.testing.assert_allclose(fields[2], expected, rtol=1e-13, atol=1e-20)
    assert fields[5] == pytest.approx(case.part.contour_step)
    np.testing.assert_allclose(fields[0], expected * case.part.contour_step, rtol=1e-13, atol=1e-20)
    _, inductance, resistance, lengths, _ = calculate_inductance_by_coil_layout(parts, config)
    assert inductance.shape == (2, 2)
    np.testing.assert_allclose(inductance, inductance.T, rtol=0, atol=1e-18)
    assert abs(inductance[0, 1]) > 0
    assert np.linalg.eigvalsh(inductance).min() > 0
    assert np.all(resistance > 0) and np.all(lengths > 0)


def test_csv_and_step_export_distinct_circuits_without_appending_a_join(tmp_path):
    case, config = setup()
    parts = route(case, config)
    result = SimpleNamespace(coil_parts=parts, input_data=config)
    csv_path = save_wire_trace_as_csv(result, tmp_path)[0]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["circuit_id"] for row in rows} == {part.circuit_id for part in parts}
    for index, part in enumerate(parts, 1):
        saved = [row for row in rows if int(row["part_index"]) == index]
        points = np.array([[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in saved])
        np.testing.assert_array_equal(points, part.wire_path.v)
        assert {row["source_group_keys"] for row in saved} == {part.source_group_keys[0]}
    files = save_wire_trace_as_step(result, tmp_path)
    assert len(files) == 2
    for path, part in zip(files, parts):
        text = path.read_text()
        assert "ISO-10303-21" in text and "SI_UNIT($,.METRE.)" in text
        vertices = np.array(
            [
                [float(value) for value in row.split(",")]
                for row in re.findall(r"CARTESIAN_POINT\('',\(([^)]+)\)\)", text)
            ]
        )
        # Every route point survives STEP's declared 12-significant-digit format.
        for point in part.wire_path.v:
            assert np.min(np.linalg.norm(vertices - point, axis=1)) < 1e-12


def test_editor_is_transactional_and_group_map_shows_the_same_identity():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    case, config = setup(mode="existing")
    editor = GroupEditor.from_parts([case.part], replace(config, routing_group_plan=None))
    project = SimpleNamespace(config=replace(config, routing_group_plan=None))
    assert editor.apply_changes(project) == {}
    editor.selected = {"s0:g0"}
    editor.assign("separate")
    assert editor.dirty
    assert len({row["circuit"] for row in editor.rows()}) == 2
    figure = Figure(figsize=(6, 3), dpi=100)
    FigureCanvasAgg(figure)
    positions = draw_group_map(figure, editor, "XY")
    assert {row[0] for row in positions} == {row["key"] for row in editor.rows()}
    assert editor.apply_changes(project)["routing_group_plan"]["fingerprint"] == editor.snapshot["fingerprint"]


def test_stale_group_assignments_are_not_reapplied_and_can_be_discarded():
    _case, config = setup()
    new_case = make_group_routing_case(group_count=3)
    editor = GroupEditor.from_parts([new_case.part], config)
    assert editor.stale_plan and editor.dirty
    assert len(editor.rows()) == 3
    assert editor.plan["fingerprint"] != config.routing_group_plan["fingerprint"]
    editor.discard()
    assert not editor.dirty and len(editor.rows()) == 3


def test_editing_one_independent_circuit_keeps_the_other_assignments_independent():
    case, config = setup(group_count=3)
    editor = GroupEditor.from_parts([case.part], config)
    assert len({row["circuit"] for row in editor.rows()}) == 3
    editor.update("s0:g0", circuit="selected")
    assert editor.plan["mode"] == "custom"
    assert [row["circuit"] for row in editor.rows()] == ["selected", "s0_g1", "s0_g2"]


def test_manual_cut_choices_override_legacy_selection_per_circuit():
    case, config = setup()
    plan = deepcopy(config.routing_group_plan)
    for row, side in zip(plan["groups"], ("high", "low")):
        row["cut"] = side
    parts = route(case, replace(config, force_cut_selection=("low",), routing_group_plan=plan))
    assert [part.groups[0].within_group_cut_side for part in parts] == ["high", "low"]


def test_reversed_circuit_is_rejected_without_refitting_current():
    case, config = setup()
    parts = route(case, config)
    parts[0].wire_path.v = parts[0].wire_path.v[::-1].copy()
    with pytest.raises(ValueError, match="reversed"):
        validate_directed_circuits(parts)


def test_shared_source_supports_do_not_become_duplicate_print_formers():
    from coilgen.group_circuits import source_support_specs

    case, config = setup()
    second = deepcopy(case.part)
    second.mesh.vertices = second.mesh.vertices + [0.2, 0, 0]
    for loop in second.contour_lines:
        loop.v += [0.2, 0, 0]
    sources = [case.part, second]
    plan = default_group_plan(group_snapshot(sources), config)
    plan["mode"] = "custom"
    for row in plan["groups"]:
        row["circuit"] = "A" if row["key"].endswith("g0") else "B"
    circuits = prepare_group_circuits(sources, replace(config, routing_group_plan=plan))
    specs = source_support_specs(circuits)
    assert sum(mesh.n_faces for _, mesh, _ in specs) == sum(part.mesh.n_faces for part in sources)
    assert {source for _, _, ids in specs for source in ids} == {0, 1}
    second.contour_step = 2
    unequal_plan = {**plan, "fingerprint": group_snapshot(sources)["fingerprint"]}
    with pytest.raises(ValueError, match="current|contour_step"):
        prepare_group_circuits(sources, replace(config, routing_group_plan=unequal_plan))


@pytest.mark.parametrize("change", ["geometry", "study", "owner", "root"])
def test_group_snapshot_cannot_follow_changed_source_or_owner(change):
    from coilgen.application import ProjectDocument

    case, config = setup(mode="existing")
    config = replace(config, routing_group_plan=None)
    project = ProjectDocument.from_config(config)
    editor = GroupEditor.from_parts([case.part], config)
    editor.bind(project)
    if change == "geometry":
        # This route fixture supplies its mesh directly, before choosing a document support.
        changed = project.update_inputs({"levels": config.levels + 2}, allow_incomplete=True)
    elif change == "study":
        changed = project.update_inputs({}, {"validation_resolution": project.study["validation_resolution"] + 1})
    elif change == "owner":
        changed = ProjectDocument.from_config(config)
    else:
        changed = replace(project, project_root=project.project_root / "other-project-root")
    before = deepcopy(editor.plan)
    assert not editor.synchronize(changed)
    assert editor.plan == before


def test_conflicting_plan_revision_preserves_unapplied_group_draft():
    from coilgen.application import ProjectDocument

    case, config = setup(mode="existing")
    config = replace(config, routing_group_plan=None)
    project = ProjectDocument.from_config(config)
    editor = GroupEditor.from_parts([case.part], config)
    editor.bind(project)
    editor.assign("per_group")
    draft = deepcopy(editor.plan)
    other = GroupEditor.from_parts([case.part], config)
    other.update(other.rows()[0]["key"], cut="high")
    changed = project.update_inputs({"routing_group_plan": other.plan}, allow_incomplete=True)
    assert not editor.synchronize(changed)
    assert editor.plan == draft and editor.dirty
    editor.discard()
    assert editor.synchronize(changed)
    assert editor.plan == other.plan and not editor.dirty
