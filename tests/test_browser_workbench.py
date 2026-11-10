"""Browser workbench: same actions, settings, datasets and core as the desktop."""

import asyncio
from html.parser import HTMLParser
import re

import numpy as np
import pytest

pytest.importorskip("trame")
pytest.importorskip("aiohttp")

from coilgen.application.scene_spec import DATASET_GROUPS  # noqa: E402
from coilgen.application.session import ACTION_IDS, SETTINGS  # noqa: E402
from coilgen.geometry import load_mesh_from_config  # noqa: E402
from coilgen.web.app import BOUND_ACTIONS, create_app  # noqa: E402
from coilgen.web.hub import Hub, safe_user_name  # noqa: E402


def test_every_session_action_is_bound_in_both_workbenches():
    assert set(BOUND_ACTIONS) == set(ACTION_IDS)
    pytest.importorskip("PySide6")
    from coilgen.gui import workbench

    assert set(workbench.SESSION_ACTION_METHODS) == set(ACTION_IDS)
    source = open(workbench.__file__, encoding="utf-8").read()
    for action, name in workbench.SESSION_ACTION_METHODS.items():
        assert hasattr(workbench.CoilGenWorkbench, name) or re.search(rf"self\.{name}\s*=", source), (action, name)


def test_browser_lists_the_shared_datasets_and_settings(tmp_path):
    server, browser = create_app(tmp_path, user="tester", server_name="test-shared-lists")
    state = server.state
    state.template = "halbach_gz"
    browser.load_template()
    labels = [item["title"] for item in state.datasets]
    assert labels == [label for _group, entries in DATASET_GROUPS for label, _key in entries]
    for node in {item.node for item in SETTINGS}:
        state.node = node
        browser._refresh_settings()
        assert [field["key"] for field in state.settings_fields] == [
            item.key for item in browser.session.settings_fields(node)
        ]
    assert state.settings_values == {} or "B0" in state.b0_text


def test_rendered_workbench_has_labeled_panel_navigation_and_selectors(tmp_path):
    # Trame silently omits unregistered HTML attributes. Inspect the template
    # delivered to Vue, where responsive visibility and button state must agree.
    elements = []

    class Elements(HTMLParser):
        def handle_starttag(self, tag, attrs):
            elements.append((tag, dict(attrs)))

    server, browser = create_app(tmp_path, server_name="test-web-pane-navigation")
    try:
        Elements().feed(server.state.trame__template_main)
        assert any(attrs.get(":data-panel") == "workspace_panel" for _, attrs in elements)
        # Flush the actual choice, not whichever nested value happened to be
        # visible before Vue's implicit v-model setter ran.
        assert any(
            tag == "vselect"
            and attrs.get(":modelvalue") == "settings_values[field.key]"
            and attrs.get("@update:modelvalue")
            == "settings_values = {...settings_values, [field.key]: $event}; flushState('settings_values')"
            for tag, attrs in elements
        )
        landmarks = {
            attrs["aria-label"] for tag, attrs in elements if tag in {"main", "aside"} and "aria-label" in attrs
        }
        assert {"Model Builder", "Plot and results", "Design plan"} <= landmarks
        for pane in ("model", "center", "plan"):
            assert any(
                tag == "button"
                and attrs.get("@click") == f"workspace_panel = '{pane}'"
                and attrs.get(":aria-pressed") == f"workspace_panel === '{pane}'"
                for tag, attrs in elements
            )
        labels = {attrs["for"] for tag, attrs in elements if tag == "label" and "for" in attrs}
        selects = [attrs.get("id") for tag, attrs in elements if tag == "select"]
        assert selects and len(selects) == len(set(selects))
        assert set(selects) <= labels
        # A plain value attribute leaves a dynamically populated native select
        # on its first option, even if the plot opened on the routed winding.
        for control, model in (
            ("coilgen-shielding-source", "shielding_source"),
            ("coilgen-shielding-component", "shielding_component"),
            ("coilgen-shielding-scale", "shielding_detail"),
        ):
            assert any(attrs.get("id") == control and attrs.get("v-model") == model for _, attrs in elements)
        tree_images = [attrs for tag, attrs in elements if tag == "img" and attrs.get("class") == "cg-tree-symbol"]
        assert tree_images and all(attrs.get("alt") == "" for attrs in tree_images)
        browser.load_template("cylinder")
        original = {node["value"]: node["symbol"] for node in server.state.nodes}
        browser.load_template("biplanar")
        updated = {node["value"]: node["symbol"] for node in server.state.nodes}
        assert len(updated) == 7 and all(src.startswith("data:image/svg+xml;base64,") for src in updated.values())
        assert original["geometry"] != updated["geometry"] and original["conductor"] == updated["conductor"]
    finally:
        browser.session.close()


def test_gallery_selection_previews_without_loading_and_commits_the_clicked_option(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-web-native-gallery")
    state, session = server.state, browser.session
    assert session.project is None
    browser.load_template("cylinder")
    original = session.project
    state.template = "halbach_gx"
    assert session.project is original
    options = {item["value"]: item for item in state.templates}
    assert options["cylindrical_xyz_gradient"]["title"].startswith("Coil systems ·")
    assert options["cylindrical_xyz_gradient"]["description"] != options["halbach_gx"]["description"]
    # The click carries the visible selection even if its state update has not reached the server.
    browser.load_template("cylindrical_xyz_gradient")
    assert state.template == "cylindrical_xyz_gradient"
    assert [channel.id for channel in session.channels] == ["gx", "gy", "gz"]
    state.settings_values = {**state.settings_values, "cylinder_radius_mm": "99"}
    current = session.assembly
    browser.load_template("halbach_gx")
    assert session.assembly is current and state.settings_dirty


def test_browser_edit_absolute_field_and_offset_roi_without_advanced_settings(tmp_path):
    from coilgen.target import define_target_field

    server, browser = create_app(tmp_path, server_name="test-web-world-roi")
    state, session = server.state, browser.session
    try:
        browser.new_project()
        state.node = "target"
        browser._refresh_settings()
        original = session.project
        assert state.target_amplitude_label == "Gradient amplitude [T/m]"
        state.settings_values = {**state.settings_values, "target_gradient_normalization": "scale"}
        assert browser._check_edits() and state.target_amplitude_label == "Amplitude multiplier"
        browser._refresh_settings()
        # Saving the displayed defaults keeps the original legacy grid.
        browser.apply_settings()
        assert session.project is original
        state.settings_values = {
            **state.settings_values,
            "field_shape_function": "0.020*(x-0.020)*(z-0.025)",
            "target_gradient_normalization": "off",
            "target_region_radius_mm": "25",
            "target_region_center_x_mm": "20",
            "target_region_center_y_mm": "-10",
            "target_region_center_z_mm": "25",
        }
        browser.apply_settings()
        config = session.project.config
        region = config.target_region_geometry
        assert session.project.revision == original.revision + 1
        assert region["shape"] == "sphere" and region["radius_m"] == pytest.approx(0.025)
        np.testing.assert_allclose(region["center_m"], [0.02, -0.01, 0.025])
        assert state.target_scaling_mode == "none"
        target = define_target_field([], None, None, config)
        x, _, z = target.coords.T
        np.testing.assert_allclose(target.b[:, 2], 0.02 * (x - 0.02) * (z - 0.025))
        state.dataset = "preview_target"
        browser._render()
        assert "center [20, -10, 25] mm" in state.scene_caption
        assert "target samples" in state.scene_caption
        assert "Absolute field" in state.scene_caption

        session.apply_config({"target_gradient_normalization": "none"})
        browser._refresh_settings()
        normalization = next(item for item in state.settings_fields if item["key"] == "target_gradient_normalization")
        absolute = [option for option in normalization["choices"] if option["title"] == "Absolute field expression"]
        assert absolute == [{"title": "Absolute field expression", "value": "none"}]
        saved = session.project
        browser.apply_settings()
        assert session.project is saved

        session.apply_config({"field_shape_function": "1e-4", "target_gradient_normalization": "linear_fit"})
        browser._render()
        assert "Applied field scale ×0.02 (fallback · review Target scaling)" in state.scene_caption
    finally:
        browser.session.close()


def test_geometry_form_edits_the_selected_support(tmp_path):
    server, browser = create_app(tmp_path, user="tester", server_name="test-geometry-sources")
    state = server.state
    for template, expected in (
        ("cylinder", "cylinder_radius_mm"),
        ("biplanar", "biplanar_separation_mm"),
        ("conical_harmonic", "conical_bottom_radius_mm"),
        ("shielded_gy", None),
    ):
        state.template = template
        browser.load_template()
        keys = set(state.settings_values)
        if expected is None:
            assert keys == {"coil_mesh_file"}
            assert "mesh in metres" in state.geometry_help
        else:
            prefix = expected.split("_", 1)[0] + "_"
            assert expected in keys and all(k == "coil_mesh_file" or k.startswith(prefix) for k in keys)
        assert keys == {field["key"] for field in state.settings_fields}

    state.template = "biplanar"
    browser.load_template()
    state.settings_values = {
        **state.settings_values,
        "biplanar_width_mm": 240,
        "biplanar_height_mm": 210,
        "biplanar_separation_mm": 120,
        "biplanar_center_x_mm": 5,
    }
    browser.apply_settings()
    project = browser.session.project
    mesh = load_mesh_from_config(project.config, project.project_root)
    np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), [0.120, 0.240, 0.210])
    np.testing.assert_allclose([mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()], [-0.055, 0.065])
    assert not browser._check_edits()
    state.settings_values = {**state.settings_values, "coil_mesh_file": "create planary mesh"}
    browser.apply_settings()
    assert "planar_width_mm" in state.settings_values and "biplanar_width_mm" not in state.settings_values
    assert not browser.session.project.config.surface_is_cylinder_flag
    browser.session.apply_config({"coil_mesh_file": "create circular mesh"})
    assert set(state.settings_values) == {"coil_mesh_file"}
    assert "circular_mesh_parameter_list" in state.geometry_help


def test_support_and_former_dimensions_follow_the_draft_selection(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-browser-conditional-dimensions")
    state, session = server.state, browser.session
    try:
        browser.new_project()
        original = session.project
        cylinder = dict(state.settings_values)
        frustum = "create conical frustum mesh"
        state.settings_values = {**cylinder, "coil_mesh_file": frustum}
        assert browser._check_edits() and session.project is original
        assert "conical_top_radius_mm" in state.settings_values and "cylinder_radius_mm" not in state.settings_values
        default_top_radius = state.settings_values["conical_top_radius_mm"]
        state.settings_values = {**state.settings_values, "conical_top_radius_mm": "101"}
        browser._refresh_settings()  # The settings dialog's discard handler.
        assert state.settings_values == cylinder and not browser._check_edits()
        state.settings_values = {**cylinder, "coil_mesh_file": frustum}
        assert browser._check_edits() and state.settings_values["conical_top_radius_mm"] == default_top_radius
        state.settings_values = {
            **state.settings_values,
            "conical_bottom_radius_mm": "95",
            "conical_top_radius_mm": "60",
            "conical_length_mm": "160",
            "conical_length_divisions": "2.5",
        }
        browser.apply_settings()
        assert session.project is original
        state.settings_values = {**state.settings_values, "conical_length_divisions": "12"}
        state.settings_values = {**state.settings_values, "coil_mesh_file": cylinder["coil_mesh_file"]}
        assert not browser._check_edits() and state.settings_values == cylinder
        state.settings_values = {**state.settings_values, "coil_mesh_file": frustum}
        assert browser._check_edits() and state.settings_values["conical_top_radius_mm"] == "60"
        browser.apply_settings()
        geometry = session.project
        assert geometry.revision == original.revision + 1
        assert geometry.config.conical_frustum_mesh_parameter_list[:3] == pytest.approx((0.160, 0.095, 0.060))
        assert geometry.config.cylinder_mesh_parameter_list == original.config.cylinder_mesh_parameter_list

        state.node = "print"
        browser._refresh_settings()
        assert set(state.settings_values) == {"support_body_mode"}
        state.settings_values = {"support_body_mode": "outside"}
        assert browser._check_edits() and session.project is geometry
        state.settings_values = {
            **state.settings_values,
            "support_body_thickness_mm": "3.5",
            "support_body_groove_clearance_mm": "0.25",
        }
        state.settings_values = {**state.settings_values, "support_body_mode": "off"}
        assert not browser._check_edits() and state.settings_values == {"support_body_mode": "off"}
        state.settings_values = {"support_body_mode": "outside"}
        assert browser._check_edits() and state.settings_values["support_body_thickness_mm"] == "3.5"
        browser.apply_settings()
        assert session.project.revision == geometry.revision + 1
        assert session.project.config.support_body_mode == "outside"
        assert session.project.config.support_body_thickness == pytest.approx(0.0035)
        assert session.project.config.support_body_groove_clearance == pytest.approx(0.00025)
        browser.undo()
        assert session.project is geometry
        browser.undo()
        assert session.project is original
    finally:
        browser.session.close()


def test_stl_scale_only_applies_to_stl_inputs(tmp_path):
    from coilgen.manufacturing import save_mesh_as_stl
    from coilgen.models import Mesh

    server, browser = create_app(tmp_path, user="tester", server_name="test-stl-scale-controls")
    state = server.state
    state.template = "cylinder"
    browser.load_template()
    raw = tmp_path / "millimetres.stl"
    save_mesh_as_stl(
        Mesh(vertices=np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 40.0, 0.0]]), faces=np.array([[0, 1, 2]])), raw
    )
    browser.session.apply_config({"coil_mesh_file": str(raw), "imported_mesh_length_scale_to_m": 0.001})
    assert set(state.settings_values) == {"coil_mesh_file", "imported_mesh_length_scale_to_m"}
    assert "saved STL" in state.geometry_help
    state.mesh_upload = {"name": raw.name, "content": list(raw.read_bytes())}
    browser.apply_settings()
    assert set(state.settings_values) == {"coil_mesh_file"}
    project = browser.session.project
    assert not project.config.surface_is_cylinder_flag
    np.testing.assert_allclose(
        np.ptp(load_mesh_from_config(project.config, project.project_root).vertices, axis=0), [0.050, 0.040, 0]
    )


def test_conflicting_form_and_preset_or_upload_edits_are_not_committed(tmp_path):
    server, browser = create_app(tmp_path, user="tester", server_name="test-preset-with-draft")
    state = server.state
    state.template = "cylinder"
    browser.load_template()
    state.node = "conductor"
    browser._refresh_settings()
    project = browser.session.project
    state.settings_values = {**state.settings_values, "conductor_width_mm": 3}
    state.profile_preset = "omega"
    browser.apply_settings()
    assert browser.session.project is project and state.settings_values["conductor_width_mm"] == 3
    state.profile_preset = None
    state.mesh_upload = {"name": "test.stl", "content": []}
    browser.apply_settings()
    assert browser.session.project is project and state.settings_values["conductor_width_mm"] == 3
    assert not (tmp_path / "uploads").exists()
    state.mesh_upload = None
    browser.apply_settings()
    state.profile_preset = "omega"
    browser.apply_settings()
    assert browser.session.project.config.conductor_cross_section_width == pytest.approx(0.003)
    assert browser.session.project.config.conductor_path_reference == "profile_origin"


def test_browser_edits_reach_the_shared_session(tmp_path):
    server, browser = create_app(tmp_path, user="tester", server_name="test-shared-edits")
    state = server.state
    state.template = "cylinder"
    browser.load_template()
    state.node = "winding"
    browser._refresh_settings()
    state.settings_values = {**state.settings_values, "levels": 12, "interconnection_axis": "support_axis"}
    browser.apply_settings()
    config = browser.session.project.config
    assert config.levels == 12 and config.interconnection_axis == "support_axis"
    browser.undo()
    assert browser.session.project.config.levels != 12
    state.node = "conductor"
    browser._refresh_settings()
    state.profile_preset = "omega"
    browser.apply_settings()
    assert browser.session.project.config.conductor_path_reference == "profile_origin"
    browser.save_project()
    assert state.project_download.startswith("downloads/") and state.project_download.endswith(".coilgen")
    browser.save_project("json")
    assert state.project_download.startswith("downloads/") and state.project_download.endswith(".coilgen.json")
    assert not state.download_url


def test_dismissing_browser_settings_discards_only_unapplied_edits(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-dismiss-settings")
    state = server.state
    try:
        browser.load_template("cylinder")
        state.ready()
        session = browser.session
        original = session.project
        undo_count = len(session.history.undo)
        with state:
            state.settings_dialog = True
        radius = state.settings_values["cylinder_radius_mm"]
        with state:
            state.settings_values = {**state.settings_values, "cylinder_radius_mm": radius + 1}
        assert state.settings_dirty
        # Vuetify writes false to this model for Cancel, Escape and backdrop dismissal.
        with state:
            state.settings_dialog = False
        assert state.settings_values["cylinder_radius_mm"] == radius
        assert not state.settings_dirty and session.project is original
        with state:
            state.settings_dialog = True
        with state:
            state.mesh_upload = {"name": "pending.stl", "content": []}
        assert state.settings_dirty and not (tmp_path / "uploads").exists()
        with state:
            state.settings_dialog = False
        assert state.mesh_upload is None and not state.settings_dirty
        with state:
            state.node, state.settings_dialog = "conductor", True
        with state:
            state.profile_preset = "omega"
        with state:
            state.settings_dialog = False
        assert state.profile_preset is None and session.project is original
        assert len(session.history.undo) == undo_count

        with state:
            state.settings_dialog = True
        with state:
            state.settings_values = {**state.settings_values, "conductor_width_mm": 3}
        browser.apply_settings()
        applied = session.project
        with state:
            state.settings_values = {**state.settings_values, "conductor_width_mm": 4}
        with state:
            state.settings_dialog = False
        assert session.project is applied and state.settings_values["conductor_width_mm"] == 3
        assert len(session.history.undo) == undo_count + 1

        with state:
            browser.open_advanced()
        config = state.advanced_config
        with state:
            state.advanced_config = "{invalid"
        browser.apply_advanced()
        assert state.advanced_dialog and session.project is applied
        with state:
            state.advanced_dialog = False
        assert state.advanced_config == state.advanced_study == ""
        browser.open_advanced()
        assert state.advanced_config == config and session.project is applied
    finally:
        browser.plotter.close()
        browser.session.close()


def test_user_names_are_validated_without_lossy_sanitization():
    assert safe_user_name("Anna-M") == "Anna-M"
    for invalid in (" Anna Müller ", "../.."):
        with pytest.raises(ValueError):
            safe_user_name(invalid)


def test_hub_requires_password_and_binds_sessions_to_cookies(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    hub = Hub(tmp_path, password="secret", max_users=2)
    started = []

    async def fake_user(user):
        started.append(user)
        return type("Entry", (), {"port": 1, "last_seen": 0, "sockets": 0, "alive": lambda self: True})()

    hub.ensure_user = fake_user

    async def scenario():
        async with TestClient(TestServer(hub.application())) as client:
            page = await client.get("/")
            assert "Private workspace passphrase" in await page.text()
            refused = await client.post("/login", data={"name": "anna", "password": "wrong"}, allow_redirects=False)
            assert refused.status == 302 and "error" in refused.headers["Location"]
            accepted = await client.post(
                "/login",
                data={"name": "anna", "password": "secret", "action": "create", "passphrase": "Private test phrase"},
                allow_redirects=False,
            )
            assert accepted.status == 302 and accepted.headers["Location"].startswith("/u/")
            sid = accepted.headers["Location"].split("/")[2]
            assert hub.sessions[sid] == "anna" and started == ["anna"]
            client.session.cookie_jar.clear()
            stolen = await client.get(f"/u/{sid}/index.html", allow_redirects=False)
            assert stolen.status == 302 and stolen.headers["Location"] == "/"

    asyncio.run(scenario())


def test_browser_u_shape_point_count_and_profile_table(tmp_path):
    from coilgen.application.conductor_profiles import validate_conductor_profile

    server, browser = create_app(tmp_path, user="tester", server_name="test-profile-presets")
    state = server.state
    state.template = "cylinder"
    browser.load_template()
    state.node = "conductor"
    browser._refresh_settings()
    project = browser.session.project
    state.profile_segments = "5"
    state.profile_preset = "round"
    browser.apply_settings()
    assert browser.session.project is project
    state.profile_segments = "8"
    state.profile_equal_area = True
    browser.apply_settings()
    config = browser.session.project.config
    assert len(config.cross_sectional_points) == 8
    assert validate_conductor_profile(config.cross_sectional_points).area_m2 == pytest.approx(
        np.pi * (config.conductor_cross_section_width / 2) ** 2
    )
    state.profile_preset = "u_shape"
    browser.apply_settings()
    assert browser.session.project.config.cross_sectional_points != config.cross_sectional_points
    browser._refresh_settings()
    assert state.profile_upload is None and not state.settings_dirty
    table = {"name": "profile.csv", "content": list(b"u;n\n-1;0\n1;0\n0;1,5\n")}
    state.profile_upload = table
    state.profile_preset = "round"
    browser.apply_settings()  # two pending replacements are refused together
    assert len(browser.session.project.config.cross_sectional_points) != 3
    state.profile_preset = None
    browser.apply_settings()
    points = browser.session.project.config.cross_sectional_points
    assert [tuple(point) for point in points] == [(-0.001, 0.0), (0.001, 0.0), (0.0, 0.0015)]
    browser._refresh_settings()
    state.profile_upload = {"name": "profile.stl", "content": list(b"0 0")}
    browser.apply_settings()
    assert browser.session.project.config.cross_sectional_points == points


def test_browser_applies_custom_round_wire_diameter_in_one_revision(tmp_path):
    from coilgen.application.conductor_profiles import round_wire_profile

    server, browser = create_app(tmp_path, user="tester", server_name="test-round-wire-diameter")
    state = server.state
    state.template = "cylinder"
    browser.load_template()
    state.node = "conductor"
    browser._refresh_settings()
    original = browser.session.project
    state.profile_preset = "round"
    state.profile_diameter_mm = ""
    browser.apply_settings()
    assert browser.session.project is original
    state.profile_diameter_mm = "1.2"
    browser.apply_settings()
    applied = browser.session.project
    assert applied.revision == original.revision + 1
    assert applied.config.conductor_cross_section_width == pytest.approx(0.0012)
    assert applied.config.conductor_cross_section_height == pytest.approx(0.0012)
    np.testing.assert_allclose(applied.config.cross_sectional_points, round_wire_profile(0.0012), atol=1e-18)
    assert state.applied_profile == "Round wire · 1.2 mm diameter"
    browser.undo()
    assert browser.session.project is original


def test_model_tree_shows_each_section_readiness_and_filters_dialog_findings(tmp_path):
    from coilgen.application.preflight import PreflightIssue, PreflightReport
    from coilgen.config import CoilGenConfig
    from coilgen.web.workbench_shell import model_tree_nodes

    report = PreflightReport(
        (
            PreflightIssue("No target", "target", "field_shape_function", "Enter a target", title="Missing target"),
            PreflightIssue("Thin wire", "material", "conductor_thickness", "Check wire", severity="warning"),
        )
    )
    nodes = {node["value"]: node for node in model_tree_nodes(CoilGenConfig(), report)}
    assert nodes["target"]["state"] == "blocking" and nodes["target"]["summary"] == "Needs input"
    assert "Enter a target" in nodes["target"]["tooltip"] and nodes["target"]["action"] == "Enter a target"
    # Preflight's material section is the browser's conductor node.
    assert nodes["conductor"]["state"] == "warning" and nodes["conductor"]["summary"] == "1 tip"
    assert nodes["print"]["state"] == "unchecked" or nodes["print"]["summary"] == "Add"
    assert nodes["geometry"]["state"] == "set"
    assert {node["state"] for node in model_tree_nodes(CoilGenConfig())} == {"unchecked"}

    server, browser = create_app(tmp_path, server_name="test-web-section-readiness")
    try:
        browser.load_template("cylinder")
        state = server.state
        assert all(node["state"] in {"set", "warning", "off"} for node in state.nodes)
        browser.session.apply_config({"field_shape_function": "none"})
        browser._refresh_model()
        target = next(node for node in state.nodes if node["value"] == "target")
        assert target["state"] == "blocking" and target["summary"] == "Needs input"
        assert state.readiness["state"] == "blocking" and state.readiness["action"] == "Fix"
        assert state.readiness["node"] == "target" and "Missing target" in state.readiness["detail"]
        assert next(node for node in state.nodes if node["value"] == "print")["summary"] == "Add"
        issue = next(issue for issue in state.preflight if issue["blocking"])
        assert issue["node"] == "target" and issue["section"] == "Target field and B0"
        assert issue["heading"] == "Missing target" and issue["action"]
        template = state.trame__template_main
        assert "readiness.action" in template and "cg-readiness" in template
        assert "preflight.filter(i => i.node === node)" in template
        assert 'data-state="item.state"' in template or ':data-state="item.state"' in template
    finally:
        browser.session.close()
