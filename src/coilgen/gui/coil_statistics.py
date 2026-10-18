"""Cheap, immutable presentation of saved conductor and circuit quantities.

No scientific arrays, artifact I/O or geometry calculations belong here. Strict
review and pipeline geometry are alternative representations, never additive
material. Density estimates require an explicit user assumption.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from numbers import Real

from .run_metadata import read_run_metadata


@dataclass(frozen=True, slots=True)
class CoilStatistic:
    key: str
    category: str
    body_id: str
    quantity: str
    value: float | str | None
    unit: str
    scope: str
    status: str
    source_key: str


@dataclass(frozen=True, slots=True)
class CoilStatisticsReport:
    rows: tuple[CoilStatistic, ...]
    message: str
    density_available: bool = False
    mass_unavailable_reason: str = "No conductor density recorded; enter an assumed density to estimate mass."
    overview_scope: str = "Circuit/volume scope unrecorded · insulation excluded"

    def row(self, key: str) -> CoilStatistic | None:
        return next((row for row in self.rows if row.key == key), None)

    @property
    def has_values(self) -> bool:
        return any(row.value is not None and row.source_key for row in self.rows)


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _records(value):
    return tuple(item for item in value if isinstance(item, Mapping)) if isinstance(value, (tuple, list)) else ()


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _value(value):
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        return value if value.strip() else None
    return _number(value)


def build_coil_statistics(record=None) -> CoilStatisticsReport:
    """Read saved scalar evidence without copying witnesses or deriving physics."""
    metadata = _mapping(read_run_metadata(record))
    summary = _mapping(metadata.get("summary"))
    config = _mapping(metadata.get("config"))
    fabrication = _mapping(metadata.get("fabrication"))
    electrical = _mapping(fabrication.get("electrical_metrics"))
    bodies = _records(electrical.get("bodies"))
    reviewed = _records(fabrication.get("paths"))
    pipeline = _mapping(metadata.get("pipeline_sweeps"))
    pipeline_bodies = _records(pipeline.get("bodies"))
    rows = []

    def add(key, category, quantity, value, unit, scope, source, body=""):
        value = _number(value) if unit else _value(value)
        rows.append(
            CoilStatistic(
                key,
                category,
                body,
                quantity,
                value,
                unit,
                scope,
                "Recorded" if value is not None else "Unavailable",
                source,
            )
        )

    complete_single = len(bodies) == 1 and bodies[0].get("circuit_complete") is True
    circuit_scope = (
        "One complete modeled electrical path; original current; self inductance excludes mutual coupling."
        if complete_single
        else "Whole-circuit estimate unavailable: multiple or incomplete paths; self inductances cannot be added without mutual coupling."
        if bodies
        else "Saved reviewed circuit estimate; body completeness and mutual-coupling scope unrecorded."
    )
    for key, quantity, field, unit, body_field in (
        ("coil:resistance", "DC resistance", "resistance_ohm", "Ω", "resistance_ohm"),
        ("coil:inductance", "Self inductance", "inductance_h", "H", "inductance_H"),
        ("coil:current", "Original circuit current", "current_a", "A", "current_A"),
    ):
        value, source = summary.get(field), "summary." + field
        if bodies and not complete_single:
            value = None
        elif value is None and complete_single:
            value, source = bodies[0].get(body_field), "fabrication.electrical_metrics.bodies[0]." + body_field
        add(key, "Reviewed circuit", quantity, value, unit, circuit_scope, source)
    add(
        "coil:length",
        "Reviewed circuit",
        "Total electrical path length",
        summary.get("wire_length_m"),
        "m",
        "Recorded sum of reviewed electrical path lengths; no extra turn-multiplicity expansion.",
        "summary.wire_length_m",
    )
    add(
        "coil:power",
        "Reviewed circuit",
        "Total Joule loss at original currents",
        summary.get("fabricated_power_w", electrical.get("total_joule_power_W")),
        "W",
        "Reviewed electrical paths at their original signed currents; no thermal acceptance.",
        "summary.fabricated_power_w",
    )
    add(
        "coil:mass",
        "Material",
        "Conductor mass",
        summary.get("conductor_mass_kg"),
        "kg",
        "Explicitly recorded conductor mass; no material or density inferred.",
        "summary.conductor_mass_kg",
    )

    selected = reviewed if reviewed else pipeline_bodies
    selected_prefix = "reviewed" if reviewed else "pipeline"
    single_volume = _mapping(selected[0].get("diagnostics")).get("absolute_volume_m3") if len(selected) == 1 else None
    volume_scope = (
        "Faceted conductor mesh; excludes insulation; "
        + ("strict reviewed representation" if reviewed else "pipeline representation")
        + "; recorded mesh volume is not a validated material union."
    )
    if (reviewed and fabrication.get("ready") is False) or (not reviewed and pipeline.get("checks_passed") is False):
        volume_scope += " Geometry checks failed."
    if len(selected) > 1:
        volume_scope += " Whole-coil volume unrecorded; per-body values are listed separately."
    add(
        "coil:volume",
        "Conductor geometry",
        "Faceted conductor volume",
        single_volume,
        "m³",
        volume_scope,
        "fabrication.paths[0].diagnostics.absolute_volume_m3"
        if reviewed
        else "pipeline_sweeps.bodies[0].diagnostics.absolute_volume_m3",
    )

    for index, body in enumerate(bodies):
        body_id = str(body.get("body_id") or f"body {index + 1}")
        complete = body.get("circuit_complete") is True
        scope = "Complete modeled electrical path" if complete else "Open/incomplete path; external return unmodeled"
        scope += "; original signed body current; self/partial inductance excludes mutual coupling."
        for field, quantity, unit in (
            ("current_A", "Original current", "A"),
            ("length_m", "Electrical path length", "m"),
            ("cross_section_area_m2", "Conducting cross-section area", "m²"),
            ("resistance_ohm", "DC resistance", "Ω"),
            ("inductance_H", "Self / partial inductance", "H"),
            ("joule_power_W", "Joule loss at original current", "W"),
            ("inductance_kind", "Inductance model", ""),
            ("circuit_complete", "Modeled circuit complete", ""),
            ("mutual_inductance_included", "Mutual inductance included", ""),
        ):
            add(
                f"body:{body_id}:{field}",
                "Reviewed bodies",
                quantity,
                body.get(field),
                unit,
                scope,
                f"fabrication.electrical_metrics.bodies[{index}].{field}",
                body_id,
            )

    for prefix, category, records, accepted, source in (
        ("reviewed", "Reviewed conductor geometry", reviewed, fabrication.get("ready"), "fabrication.paths"),
        (
            "pipeline",
            "Pipeline conductor geometry",
            pipeline_bodies,
            pipeline.get("checks_passed"),
            "pipeline_sweeps.bodies",
        ),
    ):
        for index, body in enumerate(records):
            body_id = str(body.get("body_id") or f"body {index + 1}")
            scope = (
                "Alternative "
                + prefix
                + " representation; excludes insulation; not additive with other representations."
            )
            if accepted is False or body.get("manufacturing_ok") is False:
                scope += " Geometry checks failed; inspection estimate only."
            diagnostics = _mapping(body.get("diagnostics"))
            for key, field, quantity, unit in (
                ("mesh_volume", "absolute_volume_m3", "Faceted conductor mesh volume", "m³"),
                (
                    "nominal_volume",
                    "sweep_expected_profile_volume_m3",
                    "Nominal conducting volume (profile × path)",
                    "m³",
                ),
                ("area", "sweep_nominal_cross_section_area_m2", "Nominal conducting cross-section area", "m²"),
                (
                    "volume_error",
                    "sweep_profile_volume_relative_error",
                    "Mesh/profile relative volume difference",
                    "fraction",
                ),
            ):
                add(
                    f"{prefix}:{body_id}:{key}",
                    category,
                    quantity,
                    diagnostics.get(field),
                    unit,
                    scope,
                    f"{source}[{index}].diagnostics.{field}",
                    body_id,
                )
            for field, quantity, unit in (
                ("length_m", "Conductor path length", "m"),
                ("turn_multiplicity", "Representative turn multiplicity", ""),
                ("closed", "Conductor path closed", ""),
            ):
                add(
                    f"{prefix}:{body_id}:{field}",
                    category,
                    quantity,
                    body.get(field),
                    unit,
                    scope,
                    f"{source}[{index}].{field}",
                    body_id,
                )

    for field, quantity, unit in (
        ("conductor_cross_section_width", "Configured conductor width", "m"),
        ("conductor_cross_section_height", "Configured conductor height", "m"),
        ("conductor_thickness", "Continuous-sheet thickness", "m"),
        ("specific_conductivity_conductor", "Conductor resistivity", "Ω·m"),
        ("sweep_insulation_thickness", "Isotropic insulation thickness", "m"),
        ("routing_circuit_scope", "Requested circuit scope", ""),
    ):
        add(
            "material:" + field,
            "Material",
            quantity,
            config.get(field),
            unit,
            "Saved configuration; resistivity does not identify material density; insulation volume/mass unrecorded.",
            "config." + field,
        )
    add(
        "material:density",
        "Material",
        "Conductor density",
        None,
        "kg/m³",
        "No density recorded by the saved conductor model; explicit assumption required for mass.",
        "",
    )
    add(
        "material:insulated_volume",
        "Material",
        "Insulated envelope volume",
        None,
        "m³",
        "No insulated solid volume recorded; conductor mesh volume excludes insulation.",
        "",
    )
    for field, quantity, unit, scope in (
        ("power_w", "Sheet Joule power", "W", "Continuous sheet before contouring/routing; not conductor loss."),
        ("energy_j", "Sheet magnetic energy", "J", "Continuous-sheet solution; separate from circuit inductance."),
        (
            "max_current_density_a_per_m",
            "Maximum surface current density K",
            "A/m",
            "Continuous sheet; not volume current density A/m².",
        ),
        (
            "minimum_spacing_m",
            "Minimum contour centerline spacing",
            "m",
            str(
                summary.get("minimum_spacing_scope")
                or "Unconnected contour centerlines; routed envelope clearance is separate."
            ),
        ),
        (
            "contour_components",
            "Actual contour components",
            "",
            "Actual saved contours; not configured contour levels or expanded weighted turns.",
        ),
        ("vertices", "Support vertices", "", "Saved support mesh."),
        ("triangles", "Support triangles", "", "Saved support mesh."),
    ):
        add(
            "model:" + field,
            "Continuous sheet" if field in {"power_w", "energy_j", "max_current_density_a_per_m"} else "Model",
            quantity,
            summary.get(field),
            unit,
            scope,
            "summary." + field,
        )
    for field, quantity, category, unit in (
        ("force_n", "Sheet force", "Continuous sheet", "N"),
        ("torque_nm", "Sheet torque", "Continuous sheet", "N·m"),
        ("fabricated_force_n", "Reviewed conductor force", "Reviewed circuit", "N"),
        ("fabricated_torque_nm", "Reviewed conductor torque", "Reviewed circuit", "N·m"),
    ):
        vector = summary.get(field)
        for index, axis in enumerate("xyz"):
            add(
                f"model:{field}:{axis}",
                category,
                quantity + " " + axis,
                vector[index] if isinstance(vector, (tuple, list)) and len(vector) == 3 else None,
                unit,
                "Saved force/torque coordinates and B0 model; no magnitude or load limit inferred.",
                f"summary.{field}[{index}]",
            )
    for index, plan in enumerate(_records(fabrication.get("winding")) or _records(metadata.get("winding_steps"))):
        body_id = str(plan.get("body_id") or f"body {index + 1}")
        for field, quantity in (
            ("drive_feed_designed", "Drive feed designed"),
            ("terminal_assumption", "Terminal assumption"),
        ):
            add(
                f"feed:{body_id}:{field}",
                "Reviewed bodies",
                quantity,
                plan.get(field),
                "",
                "A closed mathematical circuit does not establish a physical feed break or drive connection.",
                f"winding_steps[{index}].{field}",
                body_id,
            )
    engineering = _mapping(metadata.get("engineering_review"))
    drive = _mapping(engineering.get("drive"))
    for index, body in enumerate(_records(drive.get("bodies"))):
        body_id = str(body.get("body_id") or f"body {index + 1}")
        scope = (
            str(drive.get("scope") or "Saved drive model")
            + "; "
            + str(drive.get("inductance_scope") or "inductance scope unrecorded")
        )
        scope += "; recorded waveform/drive assumptions; no qualification claim."
        for field, quantity, unit in (
            ("peak_current_A", "Drive peak current", "A"),
            ("current_limit_A", "Amplifier current limit", "A"),
            ("current_margin_A", "Amplifier current margin", "A"),
            ("peak_voltage_V", "Drive peak voltage", "V"),
            ("voltage_limit_V", "Amplifier voltage limit", "V"),
            ("voltage_margin_V", "Amplifier voltage margin", "V"),
            ("rms_current_A", "Waveform RMS current", "A"),
            ("average_joule_loss_W", "Waveform average Joule loss", "W"),
        ):
            add(
                f"drive:{body_id}:{field}",
                "Drive model",
                quantity,
                body.get(field) if drive.get("available") is True else None,
                unit,
                scope,
                f"engineering_review.drive.bodies[{index}].{field}",
                body_id,
            )
    thermal = _mapping(engineering.get("thermal"))
    if thermal:
        scope = (
            str(thermal.get("model") or thermal.get("reason") or "Thermal model unrecorded")
            + "; no thermal qualification claim."
        )
        for field, quantity, unit in (
            ("peak_temperature_K", "Lumped model peak temperature", "K"),
            ("temperature_limit_K", "Lumped model temperature limit", "K"),
            ("temperature_margin_K", "Lumped model temperature margin", "K"),
            ("time_constant_s", "Measured lumped thermal time constant", "s"),
        ):
            add(
                "thermal:" + field,
                "Thermal model",
                quantity,
                thermal.get(field) if thermal.get("available") is True else None,
                unit,
                scope,
                "engineering_review.thermal." + field,
            )
    message = "Recorded estimates retain their circuit, current and geometry scope. Pipeline and reviewed bodies are separate representations."
    if fabrication.get("ready") is False or pipeline.get("checks_passed") is False:
        message += " Conductor geometry checks failed; numerical estimates do not establish manufacturability."
    message += " Mass needs an explicitly entered density; insulation, feeds and former mass are not included."
    short_circuit = (
        "One modeled closed path · self L, no mutual"
        if complete_single
        else ("Multiple/incomplete electrical paths" if bodies else "Circuit scope unrecorded")
    )
    short_volume = ("reviewed" if reviewed else "pipeline") + " conductor mesh" if selected else "volume unrecorded"
    short_failure = (
        " · geometry failed" if fabrication.get("ready") is False or pipeline.get("checks_passed") is False else ""
    )
    overview_scope = short_circuit + " · " + short_volume + short_failure + " · insulation excluded"
    reason = (
        ""
        if _number(summary.get("conductor_mass_kg")) is not None
        else ("No conductor density recorded; enter an assumed density to estimate mass.")
    )
    report = CoilStatisticsReport(tuple(rows), message, mass_unavailable_reason=reason, overview_scope=overview_scope)
    # Record the selected representation in a scalar row for density estimates.
    selection = CoilStatistic(
        "material:mass_representation",
        "Material",
        "",
        "Mass-estimate representation",
        selected_prefix,
        "",
        "Reviewed representation preferred; pipeline used only when reviewed bodies are absent.",
        "Recorded",
        "",
    )
    return replace(report, rows=(*report.rows, selection))


def mass_estimates(report: CoilStatisticsReport, density_kg_per_m3=None) -> CoilStatisticsReport:
    """Estimate bare conductor mass from saved volumes and an entered density.

    Nominal profile volumes and diagnostic faceted-mesh volumes remain distinct.
    Only one representation contributes to each whole-coil estimate. This is
    material accounting for the recorded bodies, not a validated solid union.
    """
    if density_kg_per_m3 is None or density_kg_per_m3 == 0:
        return report
    density = _number(density_kg_per_m3)
    if density is None or density <= 0:
        raise ValueError("Assumed conductor density must be finite and positive, or zero for unspecified.")
    selection = report.row("material:mass_representation")
    prefix = selection.value if selection is not None else "reviewed"
    rows = [row for row in report.rows if not row.key.startswith("estimate:")]
    rows.append(
        CoilStatistic(
            "estimate:density",
            "Density estimates",
            "",
            "Entered conductor density",
            density,
            "kg/m³",
            "Explicit user assumption; no material inferred; insulation excluded.",
            "Assumed density estimate",
            "user_density",
        )
    )
    totals = {}
    for kind in ("nominal_volume", "mesh_volume"):
        volumes = [row for row in report.rows if row.key.startswith(str(prefix) + ":") and row.key.endswith(":" + kind)]
        estimates = []
        for row in volumes:
            volume = _number(row.value)
            value = _number(volume * density) if volume is not None and volume >= 0 else None
            estimates.append(value)
            rows.append(
                CoilStatistic(
                    "estimate:" + row.key + ":mass",
                    "Density estimates",
                    row.body_id,
                    "Nominal conductor mass" if kind == "nominal_volume" else "Faceted-mesh mass diagnostic",
                    value,
                    "kg",
                    row.scope + f" Entered density {density:g} kg/m³; bare conductor only.",
                    "Assumed density estimate" if value is not None else "Unavailable",
                    row.source_key + " × user_density",
                )
            )
        try:
            totals[kind] = (
                _number(math.fsum(estimates)) if estimates and all(value is not None for value in estimates) else None
            )
        except OverflowError:
            totals[kind] = None
    original_mass = report.row("coil:mass")
    scope = f"Sum of {prefix} body profile volumes × entered density {density:g} kg/m³; unexpanded recorded bodies only; excludes insulation, feeds and former. Geometry may be rejected; no solid-union or fabrication claim."
    if original_mass is not None and original_mass.status != "Recorded":
        estimate = replace(
            original_mass,
            value=totals["nominal_volume"],
            scope=scope,
            status="Assumed density estimate" if totals["nominal_volume"] is not None else "Unavailable",
            source_key=str(prefix) + ".nominal_volume × user_density",
        )
        rows = [estimate if row.key == "coil:mass" else row for row in rows]
    else:
        rows.append(
            CoilStatistic(
                "estimate:coil:mass",
                "Density estimates",
                "",
                "Nominal conductor mass estimate",
                totals["nominal_volume"],
                "kg",
                scope,
                "Assumed density estimate" if totals["nominal_volume"] is not None else "Unavailable",
                str(prefix) + ".nominal_volume × user_density",
            )
        )
    rows.append(
        CoilStatistic(
            "estimate:coil:mesh_mass",
            "Density estimates",
            "",
            "Faceted-mesh mass diagnostic",
            totals["mesh_volume"],
            "kg",
            scope.replace("profile volumes", "faceted mesh volumes"),
            "Assumed density estimate" if totals["mesh_volume"] is not None else "Unavailable",
            str(prefix) + ".mesh_volume × user_density",
        )
    )
    return CoilStatisticsReport(
        tuple(rows),
        report.message + f" Density estimate uses entered {density:g} kg/m³.",
        True,
        ""
        if totals["nominal_volume"] is not None
        else "No finite nominal conducting volumes recorded for the selected representation.",
        report.overview_scope,
    )
