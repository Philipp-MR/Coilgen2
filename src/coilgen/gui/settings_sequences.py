"""User decisions over retained workbench controls; no second scientific model."""

from __future__ import annotations

from PySide6.QtWidgets import QAbstractSpinBox, QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit

from ..engineering_limits import engineering_contouring_enabled, max_wire_current_A, min_wire_distance_m
from .focused_editors import FocusedFormSection, control_form


GRAPHICAL_SETTINGS = {
    "sf_opt_method": {
        "constrained_quadratic": (
            "constraint",
            "Constrained solve",
            "Optimize with the configured field and hardware bounds.",
        ),
        "tikhonov": (
            "regularisation",
            "Tikhonov fit",
            "Fit a soft field residual with regularization. Existing hard research constraints must be removed explicitly.",
        ),
    },
    "objective": {
        "power": ("power-loss", "Joule power", "Minimize sheet Joule loss in watts."),
        "minimax": ("current-density", "Peak sheet current", "Minimize peak surface current density in A/m."),
        "magnetic_energy": ("inductance", "Magnetic energy", "Minimize stored magnetic energy in joules."),
        "Native objective (see below)": (
            "stream-function",
            "Native objective",
            "Preserve the loaded native objective, weights and regularization.",
        ),
    },
    "conductor_mounting_mode": {
        "centered": (
            "conductor-profile",
            "Path reference",
            "Retain profile coordinates relative to the routed path, including a custom offset.",
        ),
        "surface_outer": ("surface-normal", "Outside support", "Mount outside the support along its normal."),
        "surface_inner": ("surface-normal", "Inside support", "Mount on the opposite side of the support normal."),
    },
    "contour_discretization_method": {
        "level_count": (
            "contours",
            "Contour levels",
            "Choose the number of stream-function levels. Δψ sets the resulting wire current.",
        ),
        "engineering": (
            "wire-spacing",
            "Current and spacing",
            "Derive contour spacing from the configured wire current and centerline-distance limits.",
        ),
        "auto": (
            "current-density",
            "Automatic policy",
            "Use current/spacing contouring when both engineering limits are configured; otherwise use level count.",
        ),
    },
}


class _Sequence:
    def __init__(self, window, form, layout):
        self.window, self.form, self.layout = window, form, layout
        self.sections = []

    def control(self, name):
        return self.window._draft_controls.get(name) or self.form.parentWidget().findChild(QComboBox, name)

    def text(self, name):
        control = self.control(name)
        if control is None:
            return ""
        if isinstance(control, QAbstractSpinBox):
            return control.text() if control.hasAcceptableInput() else "Incomplete entry"
        if isinstance(control, QComboBox):
            value = control.currentText()
            return GRAPHICAL_SETTINGS.get(name, {}).get(value, (None, value))[1]
        if isinstance(control, QCheckBox):
            return "On" if control.isChecked() else "Off"
        if isinstance(control, QLineEdit):
            return control.text()
        return control.summary_text() if hasattr(control, "summary_text") else ""

    def labelled(self, keys):
        values = [
            f"{control.accessibleName()}: {self.text(key)}"
            for key in keys
            if (control := self.control(key)) is not None and self.text(key)
        ]
        return " · ".join(values)

    def add(self, title, keys, symbol, *, summary=None, extras=(), existing=None, size=None):
        controls = tuple(control for key in keys if (control := self.control(key)) is not None)
        if not controls and existing is None:
            return None
        if summary is None:

            def summary():
                return " · ".join(self.text(key) for key in keys if self.text(key))

        section = existing or FocusedFormSection(
            self.form,
            controls,
            title=title,
            icon_name=symbol,
            summary=summary,
            apply_callback=self.window.apply_settings,
        )
        if existing is None:
            self.form.addRow(section)
            self.window._focused_sections.append(section)
        for widget in extras:
            if widget is not None:
                self.layout.removeWidget(widget)
                section.dialog.layout().insertWidget(section.dialog.layout().count() - 1, widget)
        section.set_revert_callback(self.window.revert_draft)
        if size:
            bounds = section.dialog.screen().availableGeometry()
            section.dialog.resize(min(size[0], bounds.width() - 24), min(size[1], bounds.height() - 64))
        self.sections.append(section)
        return section

    def finish(self, next_page=None):
        window = self.window
        window._settings_decisions = {section.dialog.windowTitle(): section for section in self.sections}

        def open_decision(title, apply=False):
            if apply and not window.apply_settings():
                return
            section = window._settings_decisions.get(title)
            if section is not None:
                section.open()

        def close_other_decisions(opened):
            for section in self.sections:
                if section is not opened:
                    section.dialog.hide()

        def open_next_section():
            if not window.apply_settings():
                return
            window.select_node(next_page[0])
            first = next(iter(window._settings_decisions.values()), None)
            if first is not None:
                first.open()

        for number, section in enumerate(self.sections, 1):
            section.set_step(number, len(self.sections))
            section.opened.connect(lambda current=section: close_other_decisions(current))
            before = self.sections[number - 2].dialog.windowTitle() if number > 1 else None
            after = self.sections[number].dialog.windowTitle() if number < len(self.sections) else None
            section.configure_navigation(
                back_callback=(lambda title=before: open_decision(title)) if before else None,
                next_callback=(lambda title=after: open_decision(title, apply=True))
                if after
                else open_next_section
                if next_page
                else None,
                next_label=f"Apply & {next_page[1]}" if not after and next_page else "Apply & next",
            )
        for row in range(self.form.rowCount()):
            item = self.form.itemAt(row, QFormLayout.ItemRole.SpanningRole)
            if item is not None and isinstance(item.widget(), QLabel):
                item.widget().hide()
        # Choices can be opened in any order. Numbering supplies orientation,
        # never a claim that inputs, feasibility or engineering checks passed.


def _extras(layout, *names):
    return tuple(
        widget
        for index in range(layout.count())
        if (widget := layout.itemAt(index).widget()) is not None and widget.objectName() in names
    )


def build_settings_sequence(window, page, form, layout):
    """Move complete registered rows into short, freely navigable decisions."""
    sequence = _Sequence(window, form, layout)
    controls = window._draft_controls
    config = window.project.config
    if page == "geometry":
        gallery = sequence.control("support_type")
        if gallery is not None:
            gallery.set_expanded(True)
            sequence.add(
                "Choose support",
                ("support_type",),
                "support-gallery",
                size=(620, 620),
                summary=lambda: sequence.text("support_type") or "No support selected",
                extras=_extras(layout, "import_support"),
            )
        placement = tuple(key for key in controls if key.startswith(("rotation_", "center_", "plane_normal_")))
        mesh = tuple(key for key in controls if "divisions" in key or key == "mesh_edge_mm")
        dimensions = tuple(key for key in controls if key not in (*placement, *mesh))
        sequence.add(
            "Surface dimensions", dimensions, "conductor-profile", summary=lambda: sequence.labelled(dimensions)
        )
        sequence.add(
            "Placement",
            placement,
            "placement",
            summary=lambda: _placement_summary(sequence, placement),
            extras=_extras(layout, "geometry_definition"),
        )
        sequence.add("Mesh", mesh, "mesh", summary=lambda: sequence.labelled(mesh))
    elif page == "target":
        sequence.add(
            "Choose field", ("target_mode",), "target-gradient",
            summary=lambda: sequence.text("target_mode") or "No target selected",
            extras=_extras(layout, "import_target"),
        )
        imported = config.target_field_definition_file != "none"
        if "harmonic_degree" in controls:
            sequence.add(
                "Harmonic basis",
                ("harmonic_degree", "harmonic_order", "harmonic_reference_radius_mm"),
                "target-harmonic",
                summary=lambda: (
                    f"l = {sequence.text('harmonic_degree')} · m = {sequence.text('harmonic_order')} · R = {sequence.text('harmonic_reference_radius_mm')}"
                ),
            )
            sequence.add("Field amplitude", ("harmonic_coefficient_uT",), "field-component")
            sequence.add(
                "Harmonic origin", tuple(key for key in controls if key.startswith("harmonic_center_")), "placement"
            )
        elif imported:
            sequence.add("Map component", ("target_field_component",), "target-map")
            # The map owns its values and sample coordinates. Ignored analytic
            # preferences remain registered but do not occupy the user sequence.
            for key in (
                "field_shape_function",
                "target_gradient_strength",
                "target_gradient_normalization",
                "target_gradient_axis",
                "target_field_units",
                "design_context",
            ):
                form.setRowVisible(controls[key].parentWidget(), False)
        else:
            linear = sequence.control("linear_gradient_direction") is not None
            field_keys = (
                ("linear_gradient_direction", "target_gradient_strength") if linear else ("field_shape_function",)
            )
            sequence.add(
                "Field amplitude" if linear else "Field expression",
                field_keys,
                "field-component",
                extras=_extras(layout, "linear_gradient_summary"),
                size=(600, 350) if linear else None,
            )
            scaling = (
                "target_gradient_normalization",
                "target_field_units",
                "target_gradient_axis",
                "design_context",
                "target_field_component",
            )
            if not linear:
                scaling = ("target_gradient_strength", *scaling)
            else:
                scaling = ("field_shape_function", *scaling)
            section = sequence.add(
                "Target scaling",
                scaling,
                "constraint",
                extras=_extras(layout, "target_scaling_info", "advanced_target_controls"),
                summary=lambda: (
                    f"{sequence.text('target_field_units')} · {sequence.text('target_gradient_normalization')} · {sequence.text('target_field_component')}"
                ),
                size=(600, 480),
            )
            legacy_reveal = form.parentWidget().findChild(QCheckBox, "advanced_target_controls")
            if legacy_reveal is not None:
                # Retain the old programmatic control; choosing the dedicated
                # details window directly reveals its controls, without a model edit.
                section.opened.connect(lambda: legacy_reveal.setChecked(True))
                legacy_reveal.hide()
        region_editor = window._target_region_editor
        if not imported:
            sequence.add(
                "Choose field region", ("target_region_shape",), "roi", extras=_extras(layout, "import_region")
            )
            dimensions = (
                "target_region_radius",
                *(key for key in controls if key.startswith("roi_size_")),
                "roi_length_mm",
            )
            sequence.add("Region dimensions", dimensions, "roi")
            if hasattr(region_editor, "placement_section"):
                placement = region_editor.placement_section
                # Relocate the existing launcher into the chosen sequence order.
                form.removeWidget(placement)
                form.addRow(placement)
                sequence.add("Region placement", (), "placement", existing=placement)
            sequence.add(
                "Sampling",
                ("target_region_sampling", "target_region_resolution"),
                "independent-validation",
                summary=lambda: (
                    f"{sequence.text('target_region_sampling')} · resolution {sequence.text('target_region_resolution')}"
                ),
            )
        else:
            for key in ("target_region_radius", "target_region_resolution"):
                form.setRowVisible(controls[key].parentWidget(), False)
            region_editor.gallery.hide()
            region_editor.sampling.hide()
    elif page == "material":
        profile = sequence.add(
            "Cross-section profile",
            ("cross_sectional_points",),
            "conductor-profile",
            summary=lambda: controls["cross_sectional_points"].summary_text(),
            size=(780, 550),
        )
        window._conductor_profile_section = profile
        profile.form.setRowWrapPolicy(profile.form.RowWrapPolicy.WrapAllRows)
        for key in ("wire_width", "wire_height"):
            controls[key].valueChanged.connect(profile.refresh_summary)
            controls[key].lineEdit().textEdited.connect(profile.refresh_summary)
        sequence.add("Nominal dimensions", ("wire_width", "wire_height"), "conductor-profile")
        sequence.add(
            "Insulation",
            ("insulation",),
            "insulation",
            extras=_extras(layout, "conductor_cross_section_preview"),
            size=(600, 430),
        )
        sequence.add(
            "Physical placement",
            ("conductor_mounting_mode", "return_path_lift", "normal_shift_mm"),
            "surface-normal",
            summary=lambda: (
                f"{sequence.text('conductor_mounting_mode')} · "
                + (
                    f"return-path lift {sequence.text('normal_shift_mm')}"
                    if controls["return_path_lift"].isChecked()
                    else "return-path lift off"
                )
            ),
        )
        sequence.add("Continuous-sheet material", ("resistivity", "sheet_thickness"), "resistance")
    elif page == "winding":
        sequence.add(
            "Circuit intention",
            ("winding_intent",),
            "connected-winding",
            extras=_extras(layout, "winding_scope"),
            size=(620, 480),
        )

        def contour_summary():
            from dataclasses import replace

            if any(
                not controls[key].hasAcceptableInput()
                for key in ("levels", "max_wire_current_A", "min_wire_distance_mm", "min_turn_distance")
                if key in window._changed_controls
            ):
                return "Incomplete contour entries"
            candidate = replace(
                config,
                contour_discretization_method=window._value("contour_discretization_method"),
                max_wire_current_A=window._value("max_wire_current_A"),
                min_wire_distance_m=window._value("min_wire_distance_mm") * 1e-3,
                min_turn_distance=window._value("min_turn_distance") * 1e-3,
            )
            if engineering_contouring_enabled(candidate):
                return f"Current / spacing · ≤{max_wire_current_A(candidate):g} A · ≥{min_wire_distance_m(candidate) * 1e3:g} mm centerline"
            return f"Level count · {sequence.text('levels')} levels"

        contour = sequence.add(
            "Contour policy",
            ("contour_discretization_method", "levels", "max_wire_current_A", "min_wire_distance_mm"),
            "contours",
            summary=contour_summary,
            size=(600, 430),
        )
        controls["min_turn_distance"].valueChanged.connect(contour.refresh_summary)
        controls["min_turn_distance"].lineEdit().textEdited.connect(contour.refresh_summary)
        sequence.add("Clearance and bend limits", ("min_turn_distance", "min_bend"), "wire-spacing")
        sequence.add(
            "Conductor solids",
            ("sweep",),
            "conductor-sweep",
            summary=lambda: "Generate solids" if controls["sweep"].isChecked() else "Centerlines only",
        )
    elif page == "optimization":
        sequence.add(
            "Solve strategy",
            ("sf_opt_method", "objective"),
            "stream-function",
            extras=_extras(layout, "optimization_mode_info"),
            size=(620, 340),
        )
        sequence.add(
            "Field and current limits",
            ("target_error_tolerance_percent", "current_limit"),
            "constraint",
            summary=lambda: (
                f"Field budget {sequence.text('target_error_tolerance_percent')} · "
                + (
                    f"K ≤ {sequence.text('current_limit')}"
                    if window._value("current_limit") > 0
                    else "explicit K bound off"
                )
            ),
        )
        sequence.add(
            "Numerical settings",
            ("tikonov_reg_factor", "acceptance_slack"),
            "regularisation",
            extras=_extras(layout, "acceptance_policy_info"),
            summary=lambda: (
                f"Acceptance slack {sequence.text('acceptance_slack')}"
                + (
                    f" · Tikhonov factor {sequence.text('tikonov_reg_factor')}"
                    if window._raw_value("sf_opt_method") == "tikhonov"
                    else ""
                )
            ),
            size=(620, 300),
        )
        controls["sf_opt_method"].currentTextChanged.connect(sequence.sections[-1].refresh_summary)
    elif page == "validation":

        def limits():
            ratio = controls["design_margin"]
            if not ratio.hasAcceptableInput():
                return "Incomplete budget ratio"
            value = ratio.valueFromText(ratio.text())
            return f"Design {config.target_error_tolerance_percent:g}% · validation {config.target_error_tolerance_percent / value:g}%"

        sequence.add("Independent sampling", ("validation_resolution",), "independent-validation")
        sequence.add("Validation limits", ("design_margin",), "constraint", summary=limits)
        sequence.add(
            "Fabrication review",
            ("fabricate",),
            "conductor-sweep",
            summary=lambda: (
                "Strict review requested"
                if controls["fabricate"].isChecked()
                else "Evaluation without strict fabrication review"
            ),
        )
    elif page.startswith("stage:"):
        sequence.add("Choose output", ("study_boundary",), "compute-study")
    following = {
        "geometry": ("target", "Target"),
        "target": ("material", "Conductor"),
        "material": ("winding", "Winding"),
        "winding": ("optimization", "Solve"),
        "optimization": ("validation", "Validation"),
        "validation": (f"stage:{window.stage_selector.currentData()}", "Study"),
    }
    sequence.finish(following.get(page))
    for title, key in (
        ("Choose field", "target_mode"),
        ("Choose output", "study_boundary"),
        ("Choose field region", "target_region_shape"),
    ):
        section = window._settings_decisions.get(title)
        if section is not None:
            control = sequence.control(key)
            section.set_direct_choice(control)
            if title == "Choose field region" or (title == "Choose field" and control.count() <= 3):
                bounds = section.dialog.screen().availableGeometry()
                section.dialog.resize(min(650, bounds.width() - 24), min(400, bounds.height() - 64))
    _configure_visibility(sequence, page)
    return sequence.sections


def _configure_visibility(sequence, page):
    """Hide inactive rows without discarding their values or pending drafts."""
    window, controls = sequence.window, sequence.window._draft_controls

    def visible(key, state):
        control = controls.get(key)
        if control is not None:
            control_form(window, control, sequence.form).setRowVisible(control.parentWidget(), state)

    if page == "optimization":

        def strategy(*_args):
            tikhonov = window._raw_value("sf_opt_method") == "tikhonov"
            visible("objective", not tikhonov)
            visible("tikonov_reg_factor", tikhonov)

        controls["sf_opt_method"].currentTextChanged.connect(strategy)
        strategy()
    elif page == "winding":

        def contour_policy(*_args):
            from dataclasses import replace

            candidate = replace(
                window.project.config,
                contour_discretization_method=window._value("contour_discretization_method"),
                max_wire_current_A=window._value("max_wire_current_A"),
                min_wire_distance_m=window._value("min_wire_distance_mm") * 1e-3,
                min_turn_distance=window._value("min_turn_distance") * 1e-3,
            )
            engineering = engineering_contouring_enabled(candidate)
            visible("levels", not engineering)
            visible("max_wire_current_A", engineering or window._raw_value("contour_discretization_method") == "auto")
            visible("min_wire_distance_mm", engineering or window._raw_value("contour_discretization_method") == "auto")

        for name in (
            "contour_discretization_method",
            "max_wire_current_A",
            "min_wire_distance_mm",
            "min_turn_distance",
        ):
            control = controls[name]
            signal = control.currentTextChanged if isinstance(control, QComboBox) else control.valueChanged
            signal.connect(contour_policy)
        contour_policy()
    elif page == "material":

        def lift(*_args):
            visible("normal_shift_mm", controls["return_path_lift"].isChecked())

        controls["return_path_lift"].toggled.connect(lift)
        lift()
    elif (
        page == "target"
        and "harmonic_degree" not in controls
        and "target_gradient_normalization" in controls
        and window.project.config.target_field_definition_file == "none"
    ):

        def scaling(*_args):
            from dataclasses import replace
            from ..target import _target_gradient_normalization_method

            candidate = replace(
                window.project.config,
                target_gradient_normalization=window._value("target_gradient_normalization"),
                design_context=window._value("design_context"),
            )
            method = _target_gradient_normalization_method(candidate, source="expression")
            visible("target_gradient_strength", method != "none")
            visible("target_gradient_axis", method == "linear_fit")

        controls["target_gradient_normalization"].currentTextChanged.connect(scaling)
        controls["design_context"].currentTextChanged.connect(scaling)
        scaling()


def _placement_summary(sequence, keys):
    center = [sequence.text(key) for key in keys if key.startswith("center_")]
    angle = sequence.text("rotation_angle_deg")
    if center:
        return "Center (" + ", ".join(center) + ")" + (f" · angle {angle}" if angle else "")
    axis = [sequence.text(key) for key in keys if key.startswith("rotation_axis_")]
    normal = [sequence.text(key) for key in keys if key.startswith("plane_normal_")]
    parts = (["Normal (" + ", ".join(normal) + ")"] if normal else []) + (
        ["Rotation axis (" + ", ".join(axis) + ")"] if axis else []
    )
    if angle:
        parts.append(f"angle {angle}")
    return " · ".join(parts)
