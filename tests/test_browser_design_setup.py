"""Browser setup uses the same transaction and keeps details in the sidebar."""

import pytest

pytest.importorskip("trame")
from coilgen.web.app import create_app


@pytest.fixture
def browser(tmp_path):
    server, app = create_app(tmp_path, server_name="setup-" + tmp_path.name)
    app.session.load_template("cylinder")
    yield app
    app.plotter.close()


def edit(browser, **values):
    browser.server.state.setup_values = {**browser.server.state.setup_values, **values}
    browser.setup._changed()


def test_browser_has_all_sections_inline_and_no_default_plan(browser):
    state = browser.server.state
    assert len(state.setup_sections) == 7
    assert not state.plan_open
    assert "cg-setup-scroll" in state.trame__template_main
    assert 'v-show="setup_open.includes(section.key)"' in state.trame__template_main
    assert not state.settings_dialog


def test_browser_applies_multiple_sections_as_one_revision_and_undo(browser):
    before = browser.session.project
    edit(browser, cylinder_radius_mm="142", target_amplitude_native="32", validation_resolution="8")
    assert browser.server.state.setup_dirty
    assert browser._ready_to_submit()
    after = browser.session.project
    assert after.revision == before.revision + 1
    assert after.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.142)
    assert after.config.target_gradient_strength == 0.032
    assert after.study["validation_resolution"] == 8
    browser.session.undo()
    assert browser.session.project.fingerprint == before.fingerprint


def test_browser_invalid_details_retained_and_compute_blocked(browser):
    before = browser.session.project
    edit(browser, constrained_optimization_tolerance="1e")
    assert not browser._ready_to_submit()
    assert browser.session.project is before
    state = browser.server.state
    assert state.setup_values["constrained_optimization_tolerance"] == "1e"
    assert state.setup_error_key == "constrained_optimization_tolerance"
    assert "optimization" in state.setup_open


def test_browser_expression_policy_still_applies_before_commit(browser):
    before = browser.session.project
    edit(browser, setup_field="custom", field_shape_function="__import__('os').system('echo nope')")
    assert not browser.setup.apply()
    assert browser.session.project is before


def test_browser_keeps_advanced_and_setup_drafts_until_resolved(browser):
    browser.open_advanced()
    state = browser.server.state
    state.advanced_config = '{"levels":'
    edit(browser, cylinder_radius_mm="91")
    before = browser.session.project
    assert not browser.setup.apply()
    assert browser.session.project is before
    assert state.setup_values["cylinder_radius_mm"] == "91"
    state.advanced_dialog = False
    assert browser.setup.apply()
    assert browser.session.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.091)


def test_duplicate_channel_uses_applied_setup(browser):
    edit(browser, cylinder_radius_mm="91")
    browser.duplicate_channel()
    assert len(browser.session.assembly.channels) == 2
    assert all(
        channel.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.091)
        for channel in browser.session.assembly.channels
    )
