"""PCB physical development, continuity, clearance and scale regressions."""

from copy import deepcopy
from dataclasses import replace
import json
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("shapely")
from coilgen.geometry import build_cylinder_mesh, rotation_matrix_by_vector
from coilgen.models import CoilPart, ConnectedGroup, Mesh, WirePath
from coilgen.pcb_layout import BoardFrame, PcbDesign, PcbSettings
from _coilgen_test_support.pcb_layout import plane_part, pcb_session


@pytest.mark.parametrize("angle", [0.0, 0.7])
def test_cylindrical_development_has_deterministic_frame_and_round_trip(angle):
    mesh = build_cylinder_mesh(0.2, 0.075, 48, 8, 1, 0, 0, 0)
    rotation = rotation_matrix_by_vector(np.array([1.0, 2.0, 3.0]), angle)
    mesh.vertices = mesh.vertices @ rotation.T + [0.3, -0.2, 0.7]
    frame = BoardFrame.fit(mesh, 125)
    np.testing.assert_allclose(frame.wrap(frame.project(mesh.vertices)), mesh.vertices, atol=1e-12)
    assert frame.size[0] == pytest.approx(2 * np.pi * 0.075)
    permutation = np.arange(len(mesh.vertices))[::-1]
    other = Mesh(mesh.vertices[permutation], (len(permutation) - 1) - mesh.faces)
    reordered = BoardFrame.fit(other, 125)
    np.testing.assert_allclose(reordered.project(mesh.vertices), frame.project(mesh.vertices), atol=1e-10)


def test_planar_projection_is_exact_under_rigid_transform():
    part = plane_part()
    rotation = rotation_matrix_by_vector(np.array([2.0, 1.0, 3.0]), 0.81)
    part.mesh.vertices = part.mesh.vertices @ rotation.T + [1, 2, 3]
    frame = BoardFrame.fit(part.mesh)
    assert frame.kind == "plane"
    np.testing.assert_allclose(frame.wrap(frame.project(part.mesh.vertices)), part.mesh.vertices, atol=1e-12)


def test_unsupported_curvature_is_rejected():
    part = plane_part()
    part.mesh.vertices[2, 2] = 0.02
    with pytest.raises(ValueError, match="without distortion"):
        BoardFrame.fit(part.mesh)


def test_seam_cuts_are_real_disconnections_and_never_long_cross_board_lines():
    mesh = build_cylinder_mesh(0.2, 0.075, 32, 8, 1, 0, 0, 0)
    frame = BoardFrame.fit(mesh)
    uv = np.array([[frame.size[0] - 0.01, 0.05], [frame.size[0] + 0.01, 0.06]])
    part = CoilPart(mesh, wire_path=WirePath(frame.wrap(uv)))
    design = PcbDesign(part, PcbSettings())
    assert any(i.code == "seam" for i in design.issues)
    assert design.path is None
    pieces = design.flat_lines["wire"]
    assert len(pieces) == 2
    assert all(np.ptp(p[:, 0]) <= 0.010001 for p in pieces)


def test_width_clearance_and_nonlocal_turn_short_are_checked():
    part = plane_part([[0.02, 0.02, 0], [0.08, 0.02, 0], [0.08, 0.025, 0], [0.02, 0.025, 0]])
    safe = PcbDesign(part, PcbSettings(width_m=0.001, clearance_m=0.001))
    assert not safe.errors
    unsafe = PcbDesign(part, PcbSettings(width_m=0.0045, clearance_m=0.001))
    assert any(i.code == "turn_clearance" for i in unsafe.issues)
    assert unsafe.path is None


def test_support_opening_is_preserved_in_artwork_and_checked():
    # Rectangular frame mesh with a 20 × 20 mm central aperture.
    outer = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]])
    inner = np.array([[0.04, 0.04, 0], [0.06, 0.04, 0], [0.06, 0.06, 0], [0.04, 0.06, 0]])
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([[i, j, 4 + j], [i, 4 + j, 4 + i]])
    part = CoilPart(
        Mesh(np.vstack((outer, inner)), np.array(faces)),
        wire_path=WirePath(np.array([[0.02, 0.05, 0], [0.08, 0.05, 0]])),
    )
    design = PcbDesign(part, PcbSettings())
    assert design.board.area == pytest.approx(0.01 - 0.0004, abs=1e-9)
    assert len(design.board.interiors) == 1
    assert any(i.code == "outside" for i in design.issues)


def test_copper_outside_page_is_reported_before_export_clipping():
    part = plane_part([[-0.03, 0.03, 0], [-0.01, 0.03, 0]])
    design = PcbDesign(part, PcbSettings())
    issue = next(i for i in design.issues if i.code == "outside")
    assert issue.uv[0] < 0
    assert design.path is None
    assert design.copper["top"].is_empty


def test_lifted_path_issue_links_flat_and_wrapped_witnesses():
    part = plane_part([[0.02, 0.03, 0], [0.08, 0.03, 0.004]])
    design = PcbDesign(part, PcbSettings())
    issue = next(i for i in design.scene()["issues"] if i["code"] == "off_surface")
    np.testing.assert_allclose(design.frame.wrap([issue["uv"]])[0], issue["xyz"])
    np.testing.assert_allclose(issue["xyz"], [0.08, 0.03, 0])


def test_two_layers_require_explicit_transition_and_terminal_assignment():
    part = plane_part()
    a = np.array([[0.02, 0.03, 0], [0.05, 0.03, 0]])
    b = np.array([[0.05, 0.03, 0], [0.08, 0.06, 0]])
    part.connected_groups = [ConnectedGroup(a, a, a, b)]
    settings = PcbSettings(mode="spirals", layer_gap_m=0.0001)
    disconnected = PcbDesign(part, settings)
    assert disconnected.path is None
    edits = dict(
        connections=[dict(a="g1-in:end", b="g1-out:start", layer="top")], terminals=["g1-in:start", "g1-out:end"]
    )
    connected = PcbDesign(part, settings, edits)
    assert connected.path is not None
    assert connected.vias
    assert connected.path[0, 2] == pytest.approx(0.0001)
    assert connected.path[-1, 2] == pytest.approx(0)
    collapsed = PcbDesign(part, replace(settings, layer_gap_m=0), edits)
    assert collapsed.path is None
    assert any(i.code == "layer_stack" for i in collapsed.issues)


def test_source_immutable_undo_and_draft_binding():
    session = pcb_session()
    source = deepcopy(session.parts[0].wire_path.v)
    identity = session.identity
    session.change("settings", width_m=0.002)
    assert session.identity != identity
    np.testing.assert_array_equal(session.parts[0].wire_path.v, source)
    session.undo()
    assert session.identity == identity
    session.undo(True)
    assert session.design.settings.width_m == 0.002
    bad = session.document()
    bad["source_id"] = "different"
    with pytest.raises(ValueError, match="different saved winding"):
        session.load(bad)
    session.load(session.document())


def test_export_uses_exact_copper_shared_scale_and_origin_without_background(tmp_path):
    session = pcb_session()
    path = session.export(tmp_path / "pcb.zip")
    with zipfile.ZipFile(path) as package:
        svg = ET.fromstring(package.read("part1_top.svg"))
        assert svg.attrib["width"] == "100mm"
        assert not svg.findall("{http://www.w3.org/2000/svg}rect")
        polygons = svg.findall("{http://www.w3.org/2000/svg}path")
        assert polygons and polygons[0].attrib["fill-rule"] == "evenodd"
        dxf = package.read("part1_top.dxf").decode().replace("\r\n", "\n")
        assert "$INSUNITS\n70\n4\n" in dxf
        manifest = json.loads(package.read("pcb-manifest.json"))
        assert manifest["identity"] == session.identity
        assert "not a fabrication" in manifest["scope"]
        assert json.loads(package.read("layout.cgpcb")) == session.document()
    with pytest.raises(FileExistsError):
        session.export(path)


def test_field_uses_saved_current_and_invalidates_after_change(tmp_path):
    session = pcb_session()
    session.change("settings", layer_gap_m=0.0)
    report = session.evaluate()
    assert report["delta_rms_uT"] == pytest.approx(0, abs=1e-12)
    assert report["trace_resistance_ohm"] == pytest.approx(0.018e-6 * 0.06 / (0.001 * 70e-6))
    assert report["field_budget"]["design_margin"] == session.design_margin
    with zipfile.ZipFile(session.export(tmp_path / "evaluated.zip")) as package:
        assert json.loads(package.read("pcb-manifest.json"))["evaluation"] == report
    session.change("settings", width_m=0.002)
    assert session.evaluation is None


@pytest.mark.parametrize(
    "changes", [{"width_m": float("nan")}, {"seam_deg": 360}, {"clearance_m": -0.1}, {"copper_m": 0}]
)
def test_invalid_physical_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(PcbSettings(), **changes)
