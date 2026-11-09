"""DRAFT export notices name failed checks in words, grouped by check."""

from coilgen.application.cad import draft_export_notice
from coilgen.application.diagnostics import failure_reason_title, readable_failure_reasons


def test_codes_become_titles_grouped_by_body():
    reasons = [
        "part001_body0001:minimum_bend_radius_violation;curvature_envelope_violation;unresolved_frame_sampling",
        "part002_body0001:minimum_bend_radius_violation;support_surface_penetration",
        "part003_body0001:minimum_bend_radius_violation",
        "assembly_clearance_violation",
    ]
    readable = readable_failure_reasons(reasons)
    assert readable[0] == "Conductor bend radius is too small (3 bodies)"
    assert "Bend is too tight for the conductor width (part001_body0001)" in readable
    assert "Conductor penetrates its support (part002_body0001)" in readable
    assert "Bodies of the assembly are too close to each other" in readable
    assert not any("_violation" in item for item in readable)


def test_titles_and_free_text_pass_through():
    readable = readable_failure_reasons(
        ["Conductor bend radius is too small", "part001_body0001: draft solid not written (disk full)", "odd_code"]
    )
    assert readable == [
        "Conductor bend radius is too small",
        "draft solid not written (disk full) (part001_body0001)",
        "odd_code",
    ]


def test_field_budget_and_notice_wording():
    assert failure_reason_title("fabrication_field_shape_error_budget_exceeded") == (
        "Fabricated field exceeds its shape error budget"
    )
    notice = draft_export_notice(["part001_body0001:minimum_bend_radius_violation"])
    assert "NOT construction-released" in notice
    assert "Failed checks: Conductor bend radius is too small (part001_body0001)" in notice
