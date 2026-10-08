"""Saved engineering observations with explicit representation and scope."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True, slots=True)
class EngineeringObservation:
    key: str
    title: str
    value: float | None
    unit: str
    scope: str
    body_id: str = ""


def _finite(value):
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) else None


def gradient_review(metadata):
    """Read verified metadata; descriptive values never create acceptance limits."""
    summary = metadata.get("summary") or {}
    rows = []

    def row(key, title, value, unit, scope, body=""):
        rows.append(EngineeringObservation(key, title, _finite(value), unit, scope, body))

    row(
        "sheet_K",
        "Maximum surface current density K",
        summary.get("max_current_density_a_per_m"),
        "A/m",
        "Continuous current sheet; not copper volume current density",
    )
    row(
        "contour_spacing",
        "Minimum contour centerline spacing",
        summary.get("minimum_spacing_m"),
        "m",
        summary.get("minimum_spacing_scope", "Unconnected contour centerlines; envelope clearance is separate"),
    )
    validation = metadata.get("winding_validation") or {}
    linearity = validation.get("linearity") or {}
    scope = "Recorded electrical paths · independent grid · original signed current · thin-wire model"
    for key, title, unit in (
        ("linearity_achieved_gradient_T_per_m", "Achieved gradient", "T/m"),
        ("linearity_requested_gradient_T_per_m", "Requested gradient", "T/m"),
        ("linearity_gradient_strength_error_percent", "Gradient strength error", "%"),
        ("linearity_rms_percent", "RMS linearity", "%"),
        ("linearity_peak_percent", "Peak linearity", "%"),
        ("linearity_cross_gradient_percent", "Cross-gradient error", "%"),
    ):
        row(key, title, linearity.get(key) if validation.get("available") else None, unit, scope)
    for body in (metadata.get("fabrication") or {}).get("electrical_metrics", {}).get("bodies", ()):
        body_id = str(body.get("body_id", ""))
        complete = body.get("circuit_complete") is True
        circuit_scope = (
            "Recorded complete electrical path; self inductance excludes mutual terms"
            if complete
            else "Recorded open path; partial inductance; external return incomplete"
        )
        for key, title, unit in (
            ("current_A", "Original body current", "A"),
            ("length_m", "Electrical path length", "m"),
            ("resistance_ohm", "DC resistance", "Ω"),
            ("inductance_H", "Self / partial inductance", "H"),
            ("joule_power_W", "Joule loss at original current", "W"),
        ):
            row(body_id + ":" + key, title, body.get(key), unit, circuit_scope, body_id)
        area, current = _finite(body.get("cross_section_area_m2")), _finite(body.get("current_A"))
        row(
            body_id + ":wire_J",
            "Mean copper current density |I|/area",
            abs(current) / area if current is not None and area and area > 0 else None,
            "A/m²",
            "Uniform cross-section DC screening; no skin/proximity or thermal acceptance",
            body_id,
        )
        achieved = _finite(linearity.get("linearity_achieved_gradient_T_per_m"))
        bodies = (metadata.get("fabrication") or {}).get("electrical_metrics", {}).get("bodies", ())
        row(
            body_id + ":efficiency",
            "Gradient efficiency",
            achieved / current if complete and len(bodies) == 1 and achieved is not None and current else None,
            "T/m/A",
            scope + "; single complete body only",
            body_id,
        )
    drive = (metadata.get("engineering_review") or {}).get("drive") or {}
    for body in drive.get("bodies", ()):
        body_id = body["body_id"]
        for key, title, unit in (
            ("current_margin_A", "Amplifier current margin", "A"),
            ("voltage_margin_V", "Amplifier voltage margin", "V"),
            ("rms_current_A", "Waveform RMS current", "A"),
            ("average_joule_loss_W", "Waveform average Joule loss", "W"),
        ):
            row(
                body_id + ":" + key,
                title,
                body.get(key),
                unit,
                drive.get("scope", "") + " " + drive.get("inductance_scope", ""),
                body_id,
            )
    thermal = (metadata.get("engineering_review") or {}).get("thermal") or {}
    if metadata.get("engineering_review"):
        scope = str(thermal.get("model") or thermal.get("reason") or "Thermal model unavailable")
        for key, title, unit in (
            ("peak_temperature_K", "Lumped model peak temperature", "K"),
            ("temperature_margin_K", "Lumped model temperature margin", "K"),
            ("time_constant_s", "Measured lumped thermal time constant", "s"),
        ):
            row(
                "thermal:" + key,
                title,
                thermal.get(key) if thermal.get("available") else None,
                unit,
                scope + "; no qualification claim",
            )
    return tuple(rows)
