"""Pre-run warnings for routing settings the conductor's own checks cannot pass."""

import pytest

from coilgen.application.preflight import check_preflight
from coilgen.application.templates import template_assembly

TITLES = {
    "lift": "Return-path lift below conductor clearance",
    "cut": "Opening cut narrower than the bend radius allows",
    "spacing": "Turn spacing below conductor clearance",
}


def _warnings(project, stage="evaluated"):
    report = check_preflight(project, pipeline_stage=stage)
    found = {name for name, title in TITLES.items() if any(issue.title == title for issue in report.issues)}
    return report, found


@pytest.mark.parametrize("template", ["cylinder", "biplanar", "cylinder_gx", "halbach_gz"])
def test_templates_are_consistent(template):
    assert _warnings(template_assembly(template).channels[0].project)[1] == set()


def test_inconsistencies_warn_without_blocking():
    project = (
        template_assembly("cylinder")
        .channels[0]
        .project.update_config({"sweep_min_bend_radius": 0.006, "min_turn_distance": 0.002})
    )
    report, found = _warnings(project)
    assert report.ready
    assert found == {"cut", "spacing"}
    issue = next(issue for issue in report.issues if issue.title == TITLES["cut"])
    assert not issue.blocking and "6 mm" in issue.message
    spacing = next(issue for issue in report.issues if issue.title == TITLES["spacing"])
    assert "2 mm" in spacing.message and "3.1 mm" in spacing.action


def test_low_lift_is_flagged_with_the_required_value():
    project = template_assembly("cylinder").channels[0].project.update_config({"normal_shift_length": 0.002})
    report, found = _warnings(project)
    assert "lift" in found
    issue = next(issue for issue in report.issues if issue.title == TITLES["lift"])
    assert "3.08 mm" in issue.message and "3.1 mm" in issue.action


def test_consistent_settings_are_quiet():
    project = template_assembly("cylinder").channels[0].project.update_config({"min_turn_distance": 0.0031})
    assert _warnings(project)[1] == set()


@pytest.mark.parametrize(
    "pitch, required_width",
    [(0.004, 0.008), (0.010, 0.010), (0.015, 0.010), (0.020, 0.010), (0.030, 0.010)],
)
@pytest.mark.parametrize("width_factor", [0.99, 1.01])
def test_s_bend_cut_width_saturates_at_two_bend_radii(pitch, required_width, width_factor):
    project = (
        template_assembly("cylinder")
        .channels[0]
        .project.update_config(
            {
                "sweep_min_bend_radius": 0.005,
                "min_turn_distance": pitch,
                "within_group_cut_width": width_factor * required_width,
            }
        )
    )
    report, found = _warnings(project)
    assert report.ready
    assert ("cut" in found) == (width_factor < 1.0)
    if "cut" in found:
        issue = next(issue for issue in report.issues if issue.title == TITLES["cut"])
        assert not issue.blocking
        assert f"at least {required_width * 1000:.3g} mm" in issue.message


@pytest.mark.parametrize("stage", ["prepared", "sheet", "contours"])
def test_only_routed_stages_are_checked(stage):
    project = template_assembly("cylinder").channels[0].project
    assert _warnings(project, stage)[1] == set()
