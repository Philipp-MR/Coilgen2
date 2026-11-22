"""Qt settings-page construction, separate from project/session control."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
)
from ..application.acceptance import (
    field_acceptance_slack_T,
    field_acceptance_slack_overrides,
    independent_field_acceptance_slack_T,
)
from ..application.session import PIPELINE_STAGES
from ..application.workbench_presentation import COMPUTE_CAPTIONS
from ..config import INTERCONNECTION_AXIS_MODES
from .editors import SUPPORT_EDITORS
from .export_panel import GeometryExportPanel, GeometryExportState, export_record_key
from .winding_intent import WINDING_INTENTS, winding_intent, winding_overrides
from .icons import icon
from .visual_selectors import CircuitSelector, SurfaceGallery
from .focused_editors import control_form
from .choice_dialogs import GraphicalChoiceCombo
from .workspace_windows import PanelWindow
from .conductor_preview import ConductorPreview
from .conductor_profile import ConductorProfileEditor
from .optimization_summary import (
    build_optimization_summary,
    normalize_field_budget_metric,
    normalize_optimization_method,
)
from .colors import PRIMARY_ACTION_STYLE


STAGES = tuple((label, key) for key, label in PIPELINE_STAGES)

STAGE_ICONS = {
    "prepared": "mesh",
    "sheet": "stream-function",
    "contours": "contours",
    "routed": "conductor-sweep",
    "evaluated": "independent-validation",
}


def populate_settings_page(window, key, container, layout, form):
    """Build one adapter page and return its engineering notes."""
    config = window.project.config
    notes = []

    def note(text):
        notes.append(text)

    def details(text):
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        button = QToolButton()
        button.setText("Geometry details")
        button.setObjectName("geometry_definition")
        button.setIcon(icon("information"))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setAccessibleName("Show geometry coordinate and boundary details")
        window = PanelWindow("Geometry coordinates and boundaries", label, container, size=(620, 330))
        button.clicked.connect(window.open_panel)
        layout.addWidget(button)

    if key.startswith("stage:"):
        _build_stage(window, key, config, container, layout, form, notes.append, details)
    elif key == "geometry":
        _build_geometry(window, key, config, container, layout, form, notes.append, details)
    elif key == "target":
        _build_target(window, key, config, container, layout, form, notes.append, details)
    elif key == "material":
        _build_material(window, key, config, container, layout, form, notes.append, details)
    elif key == "print":
        _build_print(window, key, config, container, layout, form, notes.append, details)
    elif key == "optimization":
        _build_optimization(window, key, config, container, layout, form, notes.append, details)
    elif key == "winding":
        _build_winding(window, key, config, container, layout, form, notes.append, details)
    elif key == "validation":
        _build_validation(window, key, config, container, layout, form, notes.append, details)
    elif key == "export":
        _build_export(window, key, config, container, layout, form, notes.append, details)
    else:
        _build_results(window, key, config, container, layout, form, notes.append, details)
    return notes


def _build_stage(window, key, config, container, layout, form, note, details):
    stage = key.split(":", 1)[1]
    candidate = window._reuse_plan.through(stage)
    note(
        "Compute runs all required steps through this boundary. "
        + (
            f"Reuse verified {dict((key, label) for label, key in STAGES)[candidate.stage.value].lower()} evidence, then run the remaining steps."
            if candidate
            else "No compatible saved solution is available; computation starts from preparation."
        )
    )
    scope = QLabel(window._study_scope(stage, config))
    scope.setObjectName("study_scope")
    scope.setWordWrap(True)
    descriptions = {
        "prepared": "Inspect support meshes, target samples and field operators.",
        "sheet": "Optimize the stream function and check the resulting field on independent samples.",
        "contours": "Discretize the stream function into wire-current contours using the chosen policy.",
        "routed": "Run the requested interconnections and conductor sweeping. Circuit and solids settings determine the geometry.",
        "evaluated": "Evaluate the requested winding fields and electrical estimates. Review geometry and validation before export.",
    }
    selector = GraphicalChoiceCombo(
        choices={label: (STAGE_ICONS[boundary], label, descriptions[boundary]) for label, boundary in STAGES},
        title="Choose study output",
        columns=2,
    )
    selector.setObjectName("study_boundary")
    for label, boundary in STAGES:
        selector.addItem(label, boundary)
    selector.setCurrentIndex(selector.findData(stage))
    window._register(form, "study_boundary", "Requested output", selector)
    layout.addWidget(scope)
    instruction = QLabel(f"When ready, click {COMPUTE_CAPTIONS[stage]} in the main window.")
    instruction.setObjectName("study_compute_instruction")
    instruction.setWordWrap(True)
    layout.addWidget(instruction)
    note(
        "Saved solutions retain their settings and checks. Execution completion does not imply engineering acceptance."
    )


def _build_geometry(window, key, config, container, layout, form, note, details):
    surface_editor_button = QPushButton("Geometry studio…")
    surface_editor_button.setObjectName("surface_editor_launcher")
    surface_editor_button.clicked.connect(window._surface_editor_dialog)
    layout.addWidget(surface_editor_button)
    note(
        f"Source: {config.coil_mesh_file}. Generated geometry remains editable. Imported assets are owned by the saved project."
    )
    selector = SurfaceGallery()
    selector.setObjectName("support_type")
    selected = selector.findData(config.coil_mesh_file)
    missing = str(config.coil_mesh_file).strip().lower() in {"", "none"}
    if selected < 0 and not missing:
        selector.addItem("Imported / custom support", config.coil_mesh_file)
        selected = selector.count() - 1
    selector.setCurrentIndex(selected)
    window._register(form, "support_type", "Winding support surface", selector)
    editors = {editor.source: editor for editor in SUPPORT_EDITORS}
    applied_source = config.coil_mesh_file
    applied_editor = editors.get(applied_source)
    support_controls = {}

    def add_support_fields(editor, destination, prefix=""):
        values = getattr(config, editor.parameter_field)
        keys = {}
        for field in editor.fields:
            key = prefix + field.key
            value = editor.display_value(field, values)
            if field.integer:
                window._integer(destination, key, field.label, int(value), int(field.minimum), int(field.maximum))
            else:
                window._number(destination, key, field.label, value, field.suffix, field.minimum, field.maximum)
            window._draft_fields[key] = (editor.parameter_field,)
            keys[field.key] = key
        support_controls[editor.source] = keys
        return keys.values()

    applied_keys = ()
    if applied_editor:
        applied_keys = tuple(add_support_fields(applied_editor, form))
        details(applied_editor.description)
    elif Path(applied_source).suffix.lower() == ".stl":
        window._number(
            form,
            "imported_mesh_length_scale_to_m",
            "Source unit → metres",
            config.imported_mesh_length_scale_to_m,
            "",
            1e-12,
            1e9,
        )
        applied_keys = ("imported_mesh_length_scale_to_m",)

    def geometry_changes():
        source = selector.currentData()
        editor = editors.get(source)
        if editor is None:
            return {key: window._value(key) for key in applied_keys if key in window._changed_controls}
        edits = {
            field: window._value(key)
            for field, key in support_controls[source].items()
            if key in window._changed_controls
        }
        return {editor.parameter_field: editor.apply_edits(getattr(config, editor.parameter_field), edits)}

    window._draft_builder = geometry_changes

    def show_support_fields():
        source = selector.currentData()
        editor = editors.get(source)
        choose = next((section for section in window._focused_sections if selector in section.controls), None)
        destination = choose.form if choose is not None else form
        if editor is not None and source not in support_controls:
            for key in add_support_fields(editor, destination, prefix=f"pending_{editor.parameter_field}_"):
                if choose is not None:
                    choose.track_control(window._draft_controls[key])
        for candidate, keys in support_controls.items():
            for key in keys.values():
                window._set_draft_control_active(key, candidate == source)
                if candidate != applied_source:
                    destination.setRowVisible(window._draft_controls[key].parentWidget(), candidate == source)
        if applied_editor is None:
            for key in applied_keys:
                window._set_draft_control_active(key, source == applied_source)
        if choose is not None:
            for section in window._focused_sections:
                if section is not choose:
                    section.setEnabled(source == applied_source)
            if editor is not None and source != applied_source and choose.dialog.isVisible():
                first = window._draft_controls[next(iter(support_controls[source].values()))]
                destination.activate()
                choose.scroll_area.ensureWidgetVisible(first)
        window._draft_dirty = bool(window._changed_controls)
        window._update_pending()

    selector.currentIndexChanged.connect(show_support_fields)
    import_button = QPushButton("Import and apply support mesh…")
    import_button.setObjectName("import_support")
    import_button.setIcon(icon("support-import"))
    import_button.clicked.connect(window._import_mesh_dialog)
    layout.addWidget(import_button)
    note("Choose a support and set its dimensions before Apply. Preview uses the solver's geometry generator.")


def _build_target(window, key, config, container, layout, form, note, details):
    window._build_target_editor(config, form, layout, note)


def _build_material(window, key, config, container, layout, form, note, details):
    window._number(
        form, "resistivity", "Resistivity", config.specific_conductivity_conductor * 1e9, " nΩ·m", 0.000001, 1e9
    )
    window._number(
        form,
        "sheet_thickness",
        "Effective sheet thickness",
        config.conductor_thickness * 1000,
        " mm",
        0.000001,
        10000,
    )
    window._number(
        form,
        "wire_width",
        "Conductor width",
        config.conductor_cross_section_width * 1000,
        " mm",
        0.000001,
        10000,
    )
    window._number(
        form,
        "wire_height",
        "Conductor height",
        config.conductor_cross_section_height * 1000,
        " mm",
        0.000001,
        10000,
    )
    window._number(
        form, "insulation", "Insulation thickness", config.sweep_insulation_thickness * 1000, " mm", 0, 10000
    )
    window._combo(
        form,
        "conductor_mounting_mode",
        "Conductor side of support",
        ["centered", "surface_outer", "surface_inner"],
        config.conductor_mounting_mode,
    )
    window._draft_controls["conductor_mounting_mode"].setToolTip(
        "centered: the conductor straddles the support surface. surface_outer: the whole conductor sits on the "
        "side the support's surface normals point to. surface_inner: on the opposite side. Normals follow the "
        "support mesh's triangle orientation, so check them on imported meshes. Review the saved fabrication "
        "centroid and field source."
    )
    window._combo(
        form,
        "conductor_path_reference",
        "Sweep reference point",
        ["centroid", "profile_origin"],
        config.conductor_path_reference,
    )
    window._draft_controls["conductor_path_reference"].setToolTip(
        "Profile point that the exported 3D path, the fabricated field source and the bend checks follow. "
        "centroid: area centre of the profile. profile_origin: the profile's (0, 0) point, e.g. a round "
        "litz wire centre inside an omega-shaped printed holder."
    )
    window._check(form, "return_path_lift", "Lift crossing return paths", not config.skip_normal_shift)
    window._number(form, "normal_shift_mm", "Return-path lift", config.normal_shift_length * 1000, " mm", 0, 10000)
    window._draft_controls["normal_shift_mm"].setToolTip(
        "Normal displacement used by connected return-path routing. This is separate from conductor mounting and profile coordinates."
    )
    profile_editor = ConductorProfileEditor()
    nominal_width = config.conductor_cross_section_width
    nominal_height = config.conductor_cross_section_height
    profile_editor.set_profile(
        config.cross_sectional_points,
        width_mm=nominal_width * 1000,
        height_mm=nominal_height * 1000,
        round_diameter_m=nominal_width if nominal_width == nominal_height else None,
    )
    window._register(
        form, "cross_sectional_points", "Cross-section", profile_editor, initial=config.cross_sectional_points
    )

    def conductor_changes():
        points = window._value("cross_sectional_points")
        diameter = profile_editor.round_diameter_m()
        return {
            "specific_conductivity_conductor": window._value("resistivity") * 1e-9,
            "conductor_thickness": window._value("sheet_thickness") / 1000,
            "conductor_cross_section_width": window._value("wire_width") / 1000 if diameter is None else diameter,
            "conductor_cross_section_height": window._value("wire_height") / 1000 if diameter is None else diameter,
            "sweep_insulation_thickness": window._value("insulation") / 1000,
            "cross_sectional_points": points,
            "conductor_mounting_mode": window._value("conductor_mounting_mode"),
            "conductor_path_reference": window._value("conductor_path_reference"),
            "skip_normal_shift": not window._value("return_path_lift"),
            "normal_shift_length": window._value("normal_shift_mm") * 1e-3,
        }

    window._draft_builder = conductor_changes
    conductor_preview = ConductorPreview(container)
    layout.addWidget(conductor_preview)

    def show_cross_section(*_args):
        width = window._draft_controls["wire_width"]
        height = window._draft_controls["wire_height"]
        insulation = window._draft_controls["insulation"]
        mode = profile_editor.mode_combo.currentData()
        polygon, round_wire = mode != "rectangle", mode == "round"
        window._draft_fields["cross_sectional_points"] = (
            ("cross_sectional_points", "conductor_cross_section_width", "conductor_cross_section_height")
            if round_wire
            else ("cross_sectional_points",)
        )
        for control, name in ((width, "width"), (height, "height")):
            label = f"Nominal {name}" if polygon else f"Conductor {name}"
            control_form(window, control, form).labelForField(control.parentWidget()).setText(label)
            control.setAccessibleName(label)
            control.setEnabled(not round_wire)
        width.setToolTip(
            "Change the diameter in Cross-section profile."
            if round_wire
            else "The polygon sets swept geometry, copper area and physical clearance. "
            "Nominal width still sets PCB track width and some routing thresholds; "
            "nominal dimensions remain in legacy reports."
            if polygon
            else "Rectangular conducting width, transverse to the wire within the support surface. Insulation is separate."
        )
        height.setToolTip(
            "Change the diameter in Cross-section profile."
            if round_wire
            else "The polygon sets its physical height. Nominal height remains a rectangle input and legacy report setting."
            if polygon
            else "Rectangular conducting thickness along the support-surface normal. Insulation is separate."
        )
        w, h, t = (
            control.valueFromText(control.text()) if control.hasAcceptableInput() else None
            for control in (width, height, insulation)
        )
        profile_editor.set_rectangle_dimensions(w, h)
        if t is None or (not polygon and (w is None or h is None)):
            conductor_preview.set_dimensions(None, None, None)
            return
        try:
            profile = profile_editor.profile()
        except ValueError as exc:
            conductor_preview.set_invalid_profile(str(exc))
            return
        if np.asarray(profile).shape == (1, 2) and np.all(np.asarray(profile) == 0):
            conductor_preview.set_dimensions(w, h, t)
        else:
            conductor_preview.set_profile(profile, t)

    for field in ("wire_width", "wire_height", "insulation"):
        window._draft_controls[field].valueChanged.connect(show_cross_section)
        window._draft_controls[field].lineEdit().textEdited.connect(show_cross_section)
    profile_editor.changed.connect(show_cross_section)
    show_cross_section()
    note(
        "Sheet thickness belongs to the continuous optimization model. Copper area sets resistance and Joule loss; larger profiles need more routing clearance. Custom points define the swept shape. Nominal width also affects PCB and legacy routing settings, so use a value consistent with the polygon extent."
    )


def _build_print(window, key, config, container, layout, form, note, details):
    mode = str(config.support_body_mode)
    window._check(form, "support_body_enabled", "Print a former with wire grooves", mode != "off")
    window._draft_controls["support_body_enabled"].setToolTip(
        "Turns the support surface into a 3D-printable solid with grooves for the wire. "
        "It is built in the Route / sweep step and exported as its own STL."
    )
    window._combo(form, "support_body_side", "Solid side", ["inside", "outside"], mode if mode != "off" else "inside")
    window._draft_controls["support_body_side"].setToolTip(
        "Which side of the support surface the solid grows to. inside: against the conductor normal, so "
        "conductors mounted on the inner surface sit fully in the groove and centered ones half. "
        "outside: the opposite side."
    )
    window._number(
        form,
        "support_body_thickness_mm",
        "Wall thickness",
        config.support_body_thickness * 1000,
        " mm",
        0,
        10000,
    )
    window._number(
        form,
        "support_body_groove_clearance_mm",
        "Groove clearance",
        config.support_body_groove_clearance * 1000,
        " mm",
        0,
        10000,
    )
    window._draft_controls["support_body_groove_clearance_mm"].setToolTip(
        "Gap around the insulated conductor in the groove, e.g. your printer's fit tolerance."
    )
    window._number(
        form,
        "support_body_groove_extra_depth_mm",
        "Extra groove depth (advanced)",
        config.support_body_groove_extra_depth * 1000,
        " mm",
        0,
        10000,
    )
    window._draft_controls["support_body_groove_extra_depth_mm"].setToolTip(
        "Optional: cut the groove deeper than the conductor, e.g. to sink the wire below the surface."
    )
    window._draft_fields["support_body_enabled"] = ("support_body_mode",)
    window._draft_builder = lambda: {
        "support_body_mode": window._value("support_body_side") if window._value("support_body_enabled") else "off",
        "support_body_thickness": window._value("support_body_thickness_mm") / 1000,
        "support_body_groove_clearance": window._value("support_body_groove_clearance_mm") / 1000,
        "support_body_groove_extra_depth": window._value("support_body_groove_extra_depth_mm") / 1000,
    }

    def former_rows(*_args):
        enabled = window._draft_controls["support_body_enabled"].isChecked()
        for name in (
            "support_body_side",
            "support_body_thickness_mm",
            "support_body_groove_clearance_mm",
            "support_body_groove_extra_depth_mm",
        ):
            control = window._draft_controls[name]
            control_form(window, control, form).setRowVisible(control.parentWidget(), enabled)
            window._set_draft_control_active(name, enabled)
        window._draft_dirty = bool(window._changed_controls)
        if _args:
            window._update_pending()

    window._draft_controls["support_body_enabled"].toggled.connect(former_rows)
    former_rows()
    note(
        "Optional. Switch it on to get a printable former: the support grows into a solid wall and the "
        "conductor paths are cut into it as grooves. After Apply, Support preview shows the wall in beige; "
        "the grooves are cut when the design is computed, and the former is exported as its own STL."
    )


def _build_optimization(window, key, config, container, layout, form, note, details):
    native_objective = "Native objective (see below)"
    primary_keys = (
        "sf_opt_method",
        "stream_function_cost_function",
        "constrained_quadratic_objective_mode",
        "optimization_solver",
        "optimization_solver_residual_tolerance",
        "constrained_optimization_solver",
        "constrained_optimization_tolerance",
        "constrained_optimization_max_iterations",
        "constrained_optimization_threads",
        "constrained_optimization_verbose",
        "constrained_quadratic_fallback_policy",
        "regularization_selection_method",
        "tikonov_reg_factor",
        "stream_function_smoothness_factor",
        "stream_function_basis",
        "target_error_constraint_metric",
        "target_error_tolerance_percent",
        "target_error_tolerance_absolute_T",
        "target_error_rms_limit_T",
        "target_error_std_limit_T",
    )
    snapshot = QGroupBox("Cost function and solver")
    snapshot.setObjectName("optimization_summary")
    snapshot_layout = QVBoxLayout(snapshot)
    summary_title, summary_equation, summary_bounds, summary_solver = QLabel(), QLabel(), QLabel(), QLabel()
    for label, object_name in (
        (summary_title, "optimization_summary_title"),
        (summary_equation, "optimization_cost_equation"),
        (summary_bounds, "optimization_constraints"),
        (summary_solver, "optimization_solver_summary"),
    ):
        label.setObjectName(object_name)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        snapshot_layout.addWidget(label)
    summary_title.setStyleSheet("font-weight: 600")
    summary_equation.setStyleSheet("font-size: 14px")
    all_parameters = QPushButton("All optimization parameters…")
    all_parameters.setObjectName("all_optimization_parameters")
    all_parameters.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    all_parameters.setIcon(icon("settings"))
    all_parameters.setToolTip("Edit every optimization parameter, objective weight, constraint and solver option.")
    all_parameters.clicked.connect(lambda: window._advanced_dialog(focus="optimization"))
    snapshot_layout.addWidget(all_parameters)
    # Keep the mathematical objective visible above the decision cards.
    layout.insertWidget(layout.count() - 1, snapshot)
    formula_button = QPushButton("Custom formula…")
    formula_button.setObjectName("edit_custom_formula")
    formula_button.setIcon(icon("stream-function"))
    formula_button.setToolTip("Edit the stream function objective and add formula constraints.")
    layout.insertWidget(layout.indexOf(snapshot) + 1, formula_button)
    objective = (
        native_objective if config.research_physics is None else config.research_physics.get("objective", "power")
    )
    window._combo(
        form,
        "objective",
        "Objective family",
        [native_objective, "power", "minimax", "magnetic_energy", "mixture", "custom"],
        objective,
    )
    window._draft_controls["objective"].setToolTip(
        "Native preserves the implemented matrix cost. Power minimizes sheet Joule loss [W]; minimax minimizes peak K [A/m]; magnetic energy minimizes stored energy [J]. Mixture uses editable dimensionless weights and physical reference scales in All optimization parameters."
    )
    from ..research.formulas import FORMULA_EXAMPLE_CONSTRAINTS, FORMULA_EXAMPLE_OBJECTIVE

    physics = config.research_physics or {}
    regional_fields = window.project.study.get("requirements") is not None
    window._text(form, "custom_objective", "Minimize scalar expression", physics.get("custom_objective") or "")
    formula = window._draft_controls["custom_objective"]
    formula.setPlaceholderText(FORMULA_EXAMPLE_OBJECTIVE)
    constraints = QPlainTextEdit()
    saved_constraints = physics.get("custom_constraints") or ()
    constraints.setPlainText(saved_constraints if isinstance(saved_constraints, str) else "\n".join(saved_constraints))
    constraints.setPlaceholderText(FORMULA_EXAMPLE_CONSTRAINTS)
    constraints.setTabChangesFocus(True)
    constraints.setFixedHeight(86)
    window._register(form, "custom_constraints", "Additional constraints (one per line)", constraints)
    window._combo(
        form,
        "custom_solver",
        "Formula solver",
        ["auto", "conic", "local"],
        physics.get("custom_solver", "auto"),
    )
    window._draft_controls["custom_solver"].setToolTip(
        "Auto uses a conic solve for disciplined convex formulas and constrained SLSQP for other formulas. "
        "Local starts from a constrained power solution and searches for a local optimum."
    )
    window._check(
        form,
        "include_default_field_constraints",
        "Include configured field budgets",
        not regional_fields and physics.get("include_default_field_constraints", True),
    )
    window._draft_controls["include_default_field_constraints"].setToolTip(
        "Adds the configured design and suppression field budgets to the formula constraints. "
        "Studies with named regions use their region and pose constraints."
    )
    for name in (
        "custom_objective",
        "custom_constraints",
        "custom_solver",
        "include_default_field_constraints",
    ):
        window._draft_fields[name] = ("research_physics",)

    def open_formula():
        section = window._settings_decisions["Solve strategy"]
        section.open()  # Capture the previous strategy for Cancel, Escape, or window close.
        if normalize_optimization_method(window._value("sf_opt_method")) != "constrained_quadratic":
            window._draft_controls["sf_opt_method"].setCurrentText("constrained_quadratic")
        if window._value("objective") == native_objective or not window._value("custom_constraints").strip():
            window._draft_controls["objective"].setCurrentText("custom")
        section.open_for(formula if window._value("objective") == "custom" else constraints)

    formula_button.clicked.connect(open_formula)
    effective_method = "constrained_quadratic" if config.research_physics is not None else config.sf_opt_method
    window._combo(
        form,
        "sf_opt_method",
        "Solver formulation",
        ["constrained_quadratic", "tikhonov", "target_error_constrained_efficiency"],
        effective_method,
    )
    window._combo(
        form,
        "stream_function_cost_function",
        "Native cost implementation",
        ["field_error_plus_resistance", "standard", "least_squares_plus_resistance"],
        config.stream_function_cost_function,
    )
    window._draft_controls["stream_function_cost_function"].setToolTip(
        "These names are aliases of the implemented soft residual plus matrix regularizer. The displayed equation describes the effective selected formulation; physical objectives override this native cost."
    )
    window._combo(
        form,
        "constrained_quadratic_objective_mode",
        "Native constrained objective",
        ["balanced", "smoothness_priority", "minimax_current_density"],
        config.constrained_quadratic_objective_mode,
    )
    window._combo(
        form,
        "regularization_selection_method",
        "Regularization selection",
        ["auto", "fixed", "l_curve"],
        config.regularization_selection_method,
    )
    window._number(
        form, "tikonov_reg_factor", "Tikhonov regularization factor", config.tikonov_reg_factor, "", 1e-12, 1e15
    )
    mode_info = QLabel()
    mode_info.setObjectName("optimization_mode_info")
    mode_info.setWordWrap(True)
    layout.addWidget(mode_info)

    def selected_physics():
        if (
            normalize_optimization_method(window._value("sf_opt_method")) != "constrained_quadratic"
            or window._value("objective") == native_objective
        ):
            return None
        family = window._value("objective")
        draft = {**physics, "objective": family}
        if family == "custom":
            draft["custom_objective"] = window._value("custom_objective").strip()
        elif physics.get("objective") == "custom":
            draft.pop("custom_objective", None)
        if "custom_constraints" in window._changed_controls:
            draft["custom_constraints"] = [
                line.strip() for line in window._value("custom_constraints").splitlines() if line.strip()
            ]
        if "include_default_field_constraints" in window._changed_controls:
            draft["include_default_field_constraints"] = window._value("include_default_field_constraints")
        if "custom_solver" in window._changed_controls:
            draft["custom_solver"] = window._value("custom_solver")
        return draft

    def show_mode(_text=None):
        method = normalize_optimization_method(window._raw_value("sf_opt_method"))
        tikhonov = method == "tikhonov"
        physical = selected_physics() is not None
        has_formulas = physical and (
            window._value("objective") == "custom" or bool(window._value("custom_constraints").strip())
        )
        window._draft_controls["objective"].setEnabled(method == "constrained_quadratic")
        formula.setEnabled(physical and window._value("objective") == "custom")
        constraints.setEnabled(physical)
        window._draft_controls["custom_solver"].setEnabled(has_formulas)
        window._draft_controls["include_default_field_constraints"].setEnabled(physical and not regional_fields)
        window._draft_controls["tikonov_reg_factor"].setEnabled(method != "constrained_quadratic")
        window._draft_controls["regularization_selection_method"].setEnabled(tikhonov and not physical)
        window._draft_controls["stream_function_cost_function"].setEnabled(not physical)
        window._draft_controls["constrained_quadratic_objective_mode"].setEnabled(
            method == "constrained_quadratic" and not physical
        )
        for name in ("optimization_solver", "optimization_solver_residual_tolerance"):
            if name in window._draft_controls:
                window._draft_controls[name].setEnabled(not physical)
        for name in (
            "constrained_optimization_solver",
            "constrained_optimization_tolerance",
            "constrained_optimization_max_iterations",
            "constrained_optimization_threads",
            "constrained_optimization_verbose",
            "constrained_quadratic_fallback_policy",
        ):
            if name in window._draft_controls:
                window._draft_controls[name].setEnabled(
                    method == "constrained_quadratic"
                    and (
                        name not in {"constrained_optimization_verbose", "constrained_quadratic_fallback_policy"}
                        or not physical
                    )
                )
        if "stream_function_smoothness_factor" in window._draft_controls:
            window._draft_controls["stream_function_smoothness_factor"].setEnabled(not physical)
        metric = normalize_field_budget_metric(
            window._value("target_error_constraint_metric")
            if "target_error_constraint_metric" in window._draft_controls
            else config.target_error_constraint_metric
        )
        explicit_key = "target_error_" + metric + "_limit_T"
        explicit = (
            window._value(explicit_key) if explicit_key in window._draft_controls else getattr(config, explicit_key, 0)
        )
        default_budgets = not regional_fields and (not physical or window._value("include_default_field_constraints"))
        for name in ("target_error_constraint_metric", "target_error_rms_limit_T", "target_error_std_limit_T"):
            if name in window._draft_controls:
                window._draft_controls[name].setEnabled(default_budgets)
        for name in ("target_error_tolerance_percent", "target_error_tolerance_absolute_T"):
            if name in window._draft_controls:
                window._draft_controls[name].setEnabled(
                    default_budgets and not (metric in {"rms", "std"} and explicit > 0)
                )
        overrides = {name: window._value(name) for name in primary_keys if name in window._draft_controls}
        overrides["research_physics"] = selected_physics()
        if "current_limit" in window._draft_controls:
            overrides["surface_current_density_limit_A_per_m"] = window._value("current_limit")
        # Allowance display depends only on its unit map, even while a formula is incomplete.
        policy = {key: physics[key] for key in ("acceptance_absolute_by_unit",) if key in physics}
        policy_config = replace(config, research_physics=policy if physical else None)
        if "acceptance_slack" in window._changed_controls:
            allowance = field_acceptance_slack_overrides(policy_config, window._value("acceptance_slack") * 1e-9)
            policy_config = replace(policy_config, **allowance)
            if physical:
                overrides["research_physics"].update(allowance["research_physics"])
            else:
                overrides.update(allowance)
        description = build_optimization_summary(
            config,
            overrides if window._changed_controls else None,
            requirements=window.project.study.get("requirements"),
        )
        summary_title.setText(description.title + (" · pending draft" if window._changed_controls else ""))
        summary_equation.setText(description.equation)
        summary_bounds.setText("\n".join(description.constraint_lines))
        summary_bounds.setVisible(bool(description.constraint_lines))
        summary_solver.setText(description.solver_text)
        snapshot.setToolTip("\n".join((*description.terms, *description.notes)))
        if tikhonov:
            message = "Tikhonov fits a soft field residual with regularization. Field budgets are checked after solving. Physical force/torque, power/energy, field or scenario constraints must be explicitly removed in Advanced settings before switching. The hardware-limit policy can reject or rescale the solution; inspect independent validation."
        elif method == "constrained_quadratic" and window._raw_value("objective") == native_objective:
            message = f"Native quadratic objective: mode={window._value('constrained_quadratic_objective_mode')!r}; cost function={window._value('stream_function_cost_function')!r}. Configured native quadratic penalties remain active; the Tikhonov factor is used only by a requested scan fallback; no research physical objective is configured. Selecting a physical objective switches to the physical convex solver. Full controls are available in All optimization parameters."
        elif method == "constrained_quadratic":
            budgets = (
                "the field limits for each region and pose, and configured physical/scenario bounds"
                if regional_fields
                else "configured default field budgets and explicit physical/scenario bounds"
                if default_budgets
                else "explicit physical/scenario bounds; default native field budgets are disabled"
            )
            message = f"The physical solver uses {budgets} while minimizing the selected objective. Solver feasibility and independent field validation are reported separately."
            if has_formulas:
                message += " Enter one scalar objective and one comparison per constraint line. In Auto mode, disciplined convex formulas use the conic backend. Other formulas use a local constrained solve."
        elif method == "target_error_constrained_efficiency":
            message = "The native field-budget search selects a feasible regularized candidate with the lowest configured resistance norm. It is a factor search, rather than a direct conic optimality certificate. Factor range, candidate count and retry controls are editable in All optimization parameters."
        else:
            message = f"The imported formulation {method!r} remains active. Its controls are available in Advanced settings. Choose constrained_quadratic to configure physical convex objectives."
        mode_info.setText(message)
        if "acceptance_slack" in window._draft_controls:
            if "acceptance_slack" not in window._changed_controls:
                allowance = field_acceptance_slack_T(policy_config) * 1e9
                widget = window._draft_controls["acceptance_slack"]
                with QSignalBlocker(widget):
                    widget.setValue(allowance)
                window._widget_initial["acceptance_slack"] = widget.value()
                window._draft_initial["acceptance_slack"] = allowance
            independent = independent_field_acceptance_slack_T(policy_config) * 1e9
            acceptance_info.setText(
                "This control edits the selected solver's absolute field allowance. "
                + (
                    "The physical solver and independent check use the same policy. "
                    if physical
                    else "The native solver allowance and independent research check use separate existing policies. "
                )
                + f"Selected-policy independent-grid allowance: {independent:.6g} nT. Relative budgets remain additional constraints."
            )

    window._draft_controls["sf_opt_method"].currentTextChanged.connect(show_mode)
    window._draft_controls["objective"].currentTextChanged.connect(show_mode)
    formula.textChanged.connect(show_mode)
    constraints.textChanged.connect(show_mode)
    window._draft_controls["custom_solver"].currentTextChanged.connect(show_mode)
    window._draft_controls["include_default_field_constraints"].toggled.connect(show_mode)
    show_mode()
    window._number(
        form,
        "target_error_tolerance_percent",
        "Relative field budget",
        config.target_error_tolerance_percent,
        " %",
        0,
        99.999999,
    )
    window._number(
        form,
        "acceptance_slack",
        "Solver absolute acceptance slack",
        field_acceptance_slack_T(config) * 1e9,
        " nT",
        0,
        1e12,
    )

    window._combo(
        form,
        "target_error_constraint_metric",
        "Field-budget metric",
        ["norm", "rms", "std"],
        config.target_error_constraint_metric,
    )
    for name, label in (
        ("target_error_tolerance_absolute_T", "Absolute field budget"),
        ("target_error_rms_limit_T", "Explicit RMS limit (0 = use relative/absolute budget)"),
        ("target_error_std_limit_T", "Explicit STD limit (0 = use relative/absolute budget)"),
    ):
        window._number(form, name, label, getattr(config, name), " T", 0, 1e9)
        window._draft_controls[name].setToolTip(
            "Field budget in tesla, distinct from numerical acceptance slack. A positive explicit RMS/STD limit overrides the relative and absolute budgets for that metric. STD does not constrain the mean/DC offset."
        )
    window._combo(
        form,
        "optimization_solver",
        "Linear solve policy",
        ["auto", "solve", "pinv"],
        config.optimization_solver,
    )
    window._number(
        form,
        "optimization_solver_residual_tolerance",
        "Linear solve residual tolerance",
        config.optimization_solver_residual_tolerance,
        "",
        0,
        1,
    )
    window._combo(
        form,
        "constrained_optimization_solver",
        "Conic solver backend",
        ["auto", "clarabel", "ecos", "scs"],
        config.constrained_optimization_solver,
    )
    window._number(
        form,
        "constrained_optimization_tolerance",
        "Conic solver tolerance",
        config.constrained_optimization_tolerance,
        "",
        1e-18,
        1,
    )
    window._integer(
        form,
        "constrained_optimization_max_iterations",
        "Maximum solver iterations",
        config.constrained_optimization_max_iterations,
        1,
        2147483647,
    )
    window._integer(
        form,
        "constrained_optimization_threads",
        "Solver thread request (0 = default)",
        config.constrained_optimization_threads,
        0,
        4096,
    )
    window._check(
        form,
        "constrained_optimization_verbose",
        "Verbose solver output",
        config.constrained_optimization_verbose,
    )
    window._combo(
        form,
        "constrained_quadratic_fallback_policy",
        "Conic failure policy",
        ["target_error_scan", "strict"],
        config.constrained_quadratic_fallback_policy,
    )
    window._number(
        form,
        "stream_function_smoothness_factor",
        "Native smoothness penalty",
        config.stream_function_smoothness_factor,
        "",
        0,
        1e15,
    )
    window._combo(
        form,
        "stream_function_basis",
        "Stream-function basis",
        ["vertex", "cotangent_spectral_lowpass"],
        config.stream_function_basis,
    )
    for name in (
        "optimization_solver_residual_tolerance",
        "constrained_optimization_tolerance",
        "target_error_tolerance_absolute_T",
        "target_error_rms_limit_T",
        "target_error_std_limit_T",
    ):
        control = window._draft_controls[name]
        with QSignalBlocker(control):
            control.setDecimals(18)
            control.setValue(getattr(config, name))
        window._widget_initial[name] = window._raw_value(name)
    window._draft_fields["acceptance_slack"] = (
        "research_physics",
        "target_error_constraint_acceptance_absolute_T",
    )
    acceptance_info = QLabel()
    acceptance_info.setObjectName("acceptance_policy_info")
    acceptance_info.setWordWrap(True)
    layout.addWidget(acceptance_info)
    window._draft_controls["acceptance_slack"].valueChanged.connect(show_mode)
    show_mode()
    window._number(
        form,
        "current_limit",
        "Maximum surface current density K",
        config.surface_current_density_limit_A_per_m,
        " A/m",
        0,
        1e12,
    )

    def optimization_changes():
        method = window._value("sf_opt_method")
        draft = selected_physics()
        if draft is None and ("sf_opt_method" in window._changed_controls or "objective" in window._changed_controls):
            constrained = [
                name
                for name in (
                    "sheet_power_limit_W",
                    "magnetic_energy_limit_J",
                    "force_limit_N",
                    "torque_limit_Nm",
                    "force_equality_N",
                    "torque_equality_Nm",
                )
                if physics.get(name) is not None
            ]
            constrained.extend(name for name in ("field_constraints", "scenarios") if physics.get(name))
            if window._value("custom_constraints").strip():
                constrained.append("custom_constraints")
            if constrained:
                raise ValueError(
                    (
                        "Tikhonov"
                        if normalize_optimization_method(method) == "tikhonov"
                        else "The selected native formulation"
                    )
                    + " does not enforce these research constraints: "
                    + ", ".join(constrained)
                    + ". Explicitly remove them in Advanced settings before switching formulations; the current model is unchanged."
                )
        if draft is not None and "custom_constraints" in window._changed_controls:
            from ..research.formulas import parse_formulas

            # Retain editor line numbers in syntax errors and store canonical nonempty constraints.
            program = parse_formulas(draft.get("custom_objective", ""), window._value("custom_constraints"))
            draft["custom_constraints"] = list(program.constraint_lines)
        changes = {
            **{name: window._value(name) for name in primary_keys},
            "research_physics": draft,
            "sf_opt_method": method,
            "tikonov_reg_factor": window._value("tikonov_reg_factor"),
            "target_error_tolerance_percent": window._value("target_error_tolerance_percent"),
            "surface_current_density_limit_A_per_m": window._value("current_limit"),
        }
        if "acceptance_slack" in window._changed_controls:
            candidate = replace(config, research_physics=changes["research_physics"], sf_opt_method=method)
            changes.update(field_acceptance_slack_overrides(candidate, window._value("acceptance_slack") * 1e-9))
        return changes

    window._draft_builder = optimization_changes
    for name in primary_keys:
        control = window._draft_controls[name]
        signal = (
            control.currentTextChanged
            if isinstance(control, QComboBox)
            else control.toggled
            if isinstance(control, QCheckBox)
            else control.valueChanged
        )
        if name != "sf_opt_method":
            signal.connect(show_mode)
    window._draft_controls["current_limit"].valueChanged.connect(show_mode)
    show_mode()
    note(
        "All optimization parameters exposes every native optimization setting and nested physical objective/constraint setting. Inactive primary controls retain their values and can be edited there for a later formulation. Zero removes only the explicit surface-current bound; configured wire-current and spacing limits can still derive a K bound."
    )


def _build_winding(window, key, config, container, layout, form, note, details):
    circuit = CircuitSelector()
    if winding_intent(config) == "Groups & circuits plan":
        circuit.addItem("Groups & circuits plan")
    circuit.setCurrentText(winding_intent(config))
    window._register(form, "winding_intent", "Circuit intention", circuit)
    window._draft_fields["winding_intent"] = (
        "skip_postprocessing",
        "contour_only_sweep_mode",
        "routing_circuit_scope",
        "routing_group_plan",
        "skip_sweep",
    )
    window._integer(form, "levels", "Contour levels", config.levels, 2, 1000)
    if not config.skip_postprocessing and not config.contour_only_sweep_mode:
        window._check(form, "return_path_lift", "Lift crossing return paths", not config.skip_normal_shift)
        window._number(
            form, "normal_shift_mm", "Return-path lift", config.normal_shift_length * 1000, " mm", -10000, 10000
        )
        window._draft_controls["normal_shift_mm"].setToolTip(
            "Height of crossover bridges along the support normal. Zero keeps returns at their original layer."
        )
    window._combo(
        form,
        "contour_discretization_method",
        "Discretization",
        ["level_count", "engineering", "auto"],
        config.contour_discretization_method,
    )
    window._number(form, "max_wire_current_A", "Wire current limit", config.max_wire_current_A, " A", 0, 1e12)
    window._draft_controls["max_wire_current_A"].setToolTip(
        f"Maximum Δψ per wire in engineering contouring. Zero retains the general max_current_A fallback ({config.max_current_A:g} A); it does not remove that bound."
    )
    window._number(
        form,
        "min_wire_distance_mm",
        "Engineering centerline distance",
        config.min_wire_distance_m * 1000,
        " mm",
        0,
        10000,
    )
    window._draft_controls["min_wire_distance_mm"].setToolTip(
        "Minimum centerline distance in engineering contouring. Zero uses the general minimum turn spacing. This is not an edge-to-edge copper clearance."
    )
    window._combo(
        form,
        "contour_path_regularization_mode",
        "Contour processing",
        ["off", "parameterized_bspline", "resample"],
        config.contour_path_regularization_mode,
    )
    window._combo(
        form,
        "contour_smoothing_scale_mode",
        "Smoothing scale",
        ["mesh_coupled", "legacy"],
        config.contour_smoothing_scale_mode,
    )
    window._number(
        form,
        "contour_smoothing_mesh_gain",
        "Mesh-scale amplitude retained",
        config.contour_smoothing_mesh_gain,
        "",
        1e-6,
        1 - 1e-6,
    )
    window._draft_controls["contour_smoothing_mesh_gain"].setToolTip(
        "Amplitude retained at the local mesh wavelength. Smaller values smooth more strongly. "
        "The backend reduces smoothing when necessary to retain valid surface topology."
    )
    window._number(
        form,
        "contour_smoothing_factor",
        "Legacy smoothing strength",
        config.contour_smoothing_factor,
        "",
        0,
        1,
    )
    window._draft_controls["contour_smoothing_factor"].setToolTip(
        "Legacy dimensionless B-spline fit strength. Zero disables this scale mode. "
        "Mesh-coupled smoothing provides a mesh-relative alternative."
    )
    window._number(
        form,
        "contour_displacement_mm",
        "Maximum path displacement",
        config.contour_smoothing_max_displacement * 1e3,
        " mm",
        0,
        10000,
    )
    window._draft_controls["contour_displacement_mm"].setToolTip(
        "Maximum permitted displacement from each original contour. Zero disables this optional cap."
    )
    window._number(
        form,
        "contour_segment_mm",
        "Target segment length",
        config.contour_resample_target_segment_length * 1e3,
        " mm",
        0,
        10000,
    )
    window._draft_controls["contour_segment_mm"].setToolTip(
        "Requested physical sampling interval; smaller values increase geometry size and computation. "
        "Zero uses automatic mesh-relative sampling during smoothing and disables resampling-only mode."
    )
    window._check(
        form,
        "contour_smoothing_preserve_min_turn_distance",
        "Preserve centerline spacing",
        config.contour_smoothing_preserve_min_turn_distance,
    )
    window._draft_controls["contour_smoothing_preserve_min_turn_distance"].setToolTip(
        "Reject fits below the requested spacing. With zero spacing, or when the original turns "
        "already miss the limit, preserve their original minimum separation. Final copper clearance "
        "and field accuracy are evaluated separately."
    )
    window._check(
        form,
        "contour_smoothing_local_fallback",
        "Try local fits when a whole-loop fit fails",
        config.contour_smoothing_local_fallback,
    )
    window._draft_controls["contour_smoothing_local_fallback"].setToolTip(
        "Requires local mesh smoothing, a positive displacement limit and spacing protection. "
        "Try a bounded local fit before keeping an extracted loop. Tight corners can stay pinned."
    )
    contour_info = QLabel(
        "Smoothing changes closed contour paths before cutting and interconnection. "
        "Paths retain surface provenance and contour current; the field is evaluated again. "
        "Resampling only adds points and does not round corners; with connected routing it is deferred."
    )
    contour_info.setObjectName("contour_smoothing_scope")
    contour_info.setWordWrap(True)
    layout.addWidget(contour_info)

    def enable_contour_smoothing(mode):
        if (
            mode == "parameterized_bspline"
            and window._raw_value("contour_smoothing_scale_mode") == "legacy"
            and window._raw_value("contour_smoothing_factor") == 0
        ):
            window._draft_controls["contour_smoothing_scale_mode"].setCurrentText("mesh_coupled")

    window._draft_controls["contour_path_regularization_mode"].currentTextChanged.connect(enable_contour_smoothing)
    window._number(
        form,
        "min_turn_distance",
        "Minimum centerline spacing",
        config.min_turn_distance * 1000,
        " mm",
        0,
        10000,
    )
    window._number(form, "min_bend", "Minimum bend radius", config.sweep_min_bend_radius * 1000, " mm", 0, 10000)
    if not config.skip_postprocessing and not config.contour_only_sweep_mode:
        window._combo(
            form,
            "interconnection_axis",
            "Interconnection direction",
            list(INTERCONNECTION_AXIS_MODES),
            config.interconnection_axis,
        )
        window._draft_controls["interconnection_axis"].setToolTip(
            "Direction of the cuts, return rails and connectors. Choose b0 for the main field "
            "or support_axis for the support's bore axis. Field targets and errors use the B₀ direction."
        )
    window._check(form, "sweep", "Generate conductor solids", not config.skip_sweep)
    window._terminal_handling_control(form, config)
    winding_info = QLabel()
    winding_info.setObjectName("winding_scope")
    winding_info.setWordWrap(True)
    layout.addWidget(winding_info)

    def show_winding_scope(_value=None):
        mode = WINDING_INTENTS[window._raw_value("winding_intent")]
        sweep = window._raw_value("sweep")
        window._draft_controls["sweep"].setEnabled(mode != "contours")
        if mode == "independent":
            text = "Each retained contour is an independent closed turn; interconnections are skipped. "
            text += (
                "A separate solid is swept for each retained turn."
                if sweep
                else "Solid generation is disabled; only contour geometry and field evidence will be available."
            )
        elif mode == "groups":
            text = "The Groups & circuits plan assigns the retained groups to explicit circuits at the common evaluation current."
        elif mode in {"single", "per_part"}:
            text = (
                "Contours are cut and interconnected separately on each support part. "
                if mode == "per_part"
                else "Contours are cut and interconnected using the single-circuit routing scope. "
            )
            text += (
                "The connected paths are swept into conductor solids."
                if sweep
                else "Solid generation is disabled; the connected centerlines remain available."
            )
        else:
            text = "Retain unconnected contour centerlines and their available field evidence. Solid generation requires a connected circuit or independent closed turns."
            if sweep:
                text += " The saved solid preference has no route to sweep in this mode."
        winding_info.setText(text)

    def change_intent(_value):
        if window._raw_value("winding_intent") == "Contours only":
            window._draft_controls["sweep"].setChecked(False)
        show_winding_scope()

    window._draft_controls["winding_intent"].currentTextChanged.connect(change_intent)
    window._draft_controls["sweep"].toggled.connect(show_winding_scope)
    show_winding_scope()
    window._draft_builder = lambda: {
        "levels": window._value("levels"),
        **(
            {"skip_normal_shift": not window._value("return_path_lift")}
            if "return_path_lift" in window._changed_controls
            else {}
        ),
        **(
            {"normal_shift_length": window._value("normal_shift_mm") / 1000}
            if "normal_shift_mm" in window._changed_controls
            else {}
        ),
        "contour_discretization_method": window._value("contour_discretization_method"),
        "max_wire_current_A": window._value("max_wire_current_A"),
        "min_wire_distance_m": window._value("min_wire_distance_mm") * 1e-3,
        "contour_path_regularization_mode": window._value("contour_path_regularization_mode"),
        "contour_smoothing_scale_mode": window._value("contour_smoothing_scale_mode"),
        "contour_smoothing_mesh_gain": window._value("contour_smoothing_mesh_gain"),
        "contour_smoothing_factor": window._value("contour_smoothing_factor"),
        "contour_resample_target_segment_length": window._value("contour_segment_mm") * 1e-3,
        "contour_smoothing_max_displacement": window._value("contour_displacement_mm") * 1e-3,
        "contour_smoothing_preserve_min_turn_distance": window._value("contour_smoothing_preserve_min_turn_distance"),
        "contour_smoothing_local_fallback": window._value("contour_smoothing_local_fallback"),
        "min_turn_distance": window._value("min_turn_distance") / 1000,
        "sweep_min_bend_radius": window._value("min_bend") / 1000,
        "interconnection_axis": (
            window._value("interconnection_axis")
            if "interconnection_axis" in window._draft_controls
            and window._value("interconnection_axis") in INTERCONNECTION_AXIS_MODES
            else config.interconnection_axis
        ),
        "skip_sweep": not window._value("sweep"),
        "layout_terminal_strategy": window._value("layout_terminal_strategy"),
        **winding_overrides(config, window._value("winding_intent")),
    }
    note(
        "Contour step Δψ gives the wire current [A]; surface current density K is [A/m]. Feed/return conductors and drive terminals require explicit definition. Geometry closure alone does not establish a complete driven circuit."
    )


def _build_validation(window, key, config, container, layout, form, note, details):
    from ..application.targets import field_budget_text
    from ..application.run_progress import worker_limit_text

    note(field_budget_text(config, window.design_margin))
    note(worker_limit_text(getattr(window.jobs, "policy", None)))
    window._integer(form, "validation_resolution", "Independent grid resolution", window.validation_resolution, 2, 128)
    window._number(form, "design_margin", "Design / validation budget ratio", window.design_margin, "", 0.001, 1.0)
    window._check(form, "fabricate", "Strict fabrication review", window.fabricate)
    window._draft_controls["fabricate"].setEnabled(window.stage_selector.currentData() == "evaluated")
    note(
        "These settings apply to the next study. Validation limit = design limit ÷ budget ratio. A ratio of 1 uses the same limit; 0.8 permits 1.25× the design limit. This does not change optimization tolerance. Independent sampling checks field generalization; mesh convergence requires a separate refinement study."
    )


def _build_export(window, key, config, container, layout, form, note, details):
    record = window.runs.get(window._selected_run)
    options, diagnostic = (), window._run_verification_error(window._selected_run)
    if record is not None and not diagnostic:
        from ..application.cad import export_catalog

        try:
            options = export_catalog(record)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            diagnostic = f"Saved geometry verification failed: {exc}"
    window.geometry_export_panel = GeometryExportPanel(
        record,
        options,
        diagnostic=diagnostic,
        state=window._geometry_export_states.setdefault(export_record_key(record), GeometryExportState()),
    )
    window.geometry_export_panel.set_context(
        window.project, assembly=window.assembly, pending=window._has_input_drafts()
    )
    if record is not None and not window._has_export_source(record):
        window.geometry_export_panel.include_project.setChecked(False)
        window.geometry_export_panel.include_project.setEnabled(False)
        window.geometry_export_panel.include_project.setToolTip(
            "Open this run’s exact source revision to include it, or export geometry alone."
        )
        window.geometry_export_panel.scope.setText(
            window.geometry_export_panel.scope.text()
            + " Exact source revision unavailable; geometry-only export is available."
        )
    window.geometry_export_panel.export_button.setStyleSheet(PRIMARY_ACTION_STYLE)
    window.geometry_export_panel.export_button.setIcon(icon("export-conductor", foreground="primary"))
    window.geometry_export_panel.set_exporting(window._geometry_export_task is not None)
    window.geometry_export_panel.export_requested.connect(window._geometry_export_dialog)
    window.geometry_export_panel.review_requested.connect(lambda: window._guidance_action("review_fabrication"))
    window.geometry_export_panel.problems_requested.connect(lambda: window._guidance_action("inspect_problems"))
    layout.addWidget(window.geometry_export_panel)


def _build_results(window, key, config, container, layout, form, note, details):
    note(
        "Select a completed job or run below to inspect its immutable geometry, constraints and provenance. Previous runs remain visible after model edits and are marked stale when their design fingerprint differs."
    )
    if window._selected_run is not None:
        window._build_run_inspector(layout, note)
