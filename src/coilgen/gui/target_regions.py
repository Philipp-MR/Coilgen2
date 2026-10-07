"""Region controls over the same owned, SI target definition used by scripts."""

from copy import deepcopy
import math

from PySide6.QtWidgets import QLabel, QPushButton

from ..target_regions import TargetRegion
from .focused_editors import group_form_controls
from .icons import icon
from .region_selectors import RegionSamplingSelector, TargetRegionGallery


def default_region(config, shape="sphere", sampling="volume"):
    """Carry explicit placement between primitives, keeping dimensions physical."""
    try:
        previous = (
            TargetRegion.from_dict(config.target_region_geometry).to_dict()
            if config.target_region_geometry is not None
            else {}
        )
    except (TypeError, ValueError):
        previous = {}
    radius = float(previous.get("radius_m", config.target_region_radius))
    region = {
        "shape": shape,
        "sampling": sampling,
        "center_m": list(previous.get("center_m", (0, 0, 0))),
        "rotation_axis": list(previous.get("rotation_axis", (0, 0, 1))),
        "rotation_angle_rad": previous.get("rotation_angle_rad", 0.0),
    }
    if shape == "sphere":
        region["radius_m"] = radius
    elif shape == "ellipsoid":
        region["semi_axes_m"] = list(previous.get("semi_axes_m", (radius, radius, radius)))
    elif shape == "box":
        region["size_m"] = list(previous.get("size_m", (2 * radius,) * 3))
    elif shape == "cylinder":
        region.update(radius_m=radius, length_m=previous.get("length_m", 2 * radius))
    elif shape == "enclosing_mesh":
        region = {"shape": shape, "sampling": sampling}
    else:
        raise ValueError(f"Unknown target region: {shape}.")
    return region


class TargetRegionEditor:
    """A small form adapter; the workbench owns draft validation and undo."""

    def __init__(self, window, config, form, layout, note):
        self.window, self.config = window, config
        self.focused_sections = []
        self.region = deepcopy(config.target_region_geometry)
        self.imported_map = imported_map = config.target_field_definition_file != "none"
        # A field map owns its sample coordinates. Ignored analytic settings may
        # be incomplete or invalid; preserve them without parsing or repairing.
        display, self.region_error = None, None
        if self.region is not None and not imported_map:
            try:
                display = TargetRegion.from_dict(self.region).to_dict()
            except (TypeError, ValueError) as exc:
                self.region_error = str(exc)
        self.shape = None if self.region_error else display["shape"] if display else "sphere"
        legacy_mesh = not imported_map and self.region is None and config.target_mesh_file != "none"
        radius_row = form.takeRow(window._draft_controls["target_region_radius"].parentWidget())
        resolution_row = form.takeRow(window._draft_controls["target_region_resolution"].parentWidget())
        form.addRow(QLabel("Field region"))
        self.gallery = TargetRegionGallery(compact=True)
        self.gallery.setObjectName("target_region_shape")
        if legacy_mesh:
            self.gallery.add_legacy_mesh()
        self.gallery.setCurrentIndex(
            self.gallery.findData("imported_map" if imported_map else "legacy_mesh" if legacy_mesh else self.shape)
        )
        form.addRow(self.gallery)
        self.gallery.setEnabled(not imported_map)
        self.gallery.currentIndexChanged.connect(lambda: window._switch_target_region(self.gallery.currentData()))
        self.sampling = RegionSamplingSelector()
        self.sampling.setObjectName("target_region_sampling")
        self.sampling.setCurrentIndex(
            self.sampling.findData(
                display["sampling"]
                if display
                else "surface"
                if config.use_only_target_mesh_verts and legacy_mesh
                else "volume"
            )
        )
        self.sampling.setEnabled(not imported_map and not legacy_mesh and not self.region_error)
        form.addRow(self.sampling)
        self.sampling.currentIndexChanged.connect(
            lambda: window._switch_target_region_sampling(self.sampling.currentData())
        )
        form.addRow(radius_row.labelItem.widget(), radius_row.fieldItem.widget())
        radius = window._draft_controls["target_region_radius"]
        radius.setEnabled(not imported_map and not legacy_mesh and self.shape in {"sphere", "cylinder"})
        form.setRowVisible(radius.parentWidget(), self.shape in {"sphere", "cylinder"} and not legacy_mesh)
        if display and "radius_m" in display:
            window._draft_fields["target_region_radius"] = ("target_region_geometry", "target_region_radius")
        self.dimensions = {}
        if self.shape in {"ellipsoid", "box"} and not legacy_mesh:
            values = self.region["semi_axes_m" if self.shape == "ellipsoid" else "size_m"]
            factor = 2000 if self.shape == "ellipsoid" else 1000
            for i, axis in enumerate("xyz"):
                key = f"roi_size_{axis}_mm"
                window._number(
                    form,
                    key,
                    f"{'Diameter' if self.shape == 'ellipsoid' else 'Full size'} {axis}",
                    values[i] * factor,
                    " mm",
                    0.001,
                    10000,
                )
                window._draft_fields[key] = ("target_region_geometry",)
                window._draft_controls[key].setEnabled(not imported_map)
                self.dimensions[key] = ("semi_axes_m" if self.shape == "ellipsoid" else "size_m", i, 1 / factor)
        elif self.shape == "cylinder":
            window._number(
                form,
                "roi_length_mm",
                "Full axial length (local z)",
                self.region["length_m"] * 1000,
                " mm",
                0.001,
                10000,
            )
            window._draft_fields["roi_length_mm"] = ("target_region_geometry",)
            window._draft_controls["roi_length_mm"].setEnabled(not imported_map)
        form.addRow(resolution_row.labelItem.widget(), resolution_row.fieldItem.widget())
        self.sample_count_label = QLabel("Sample count available after preview")
        self.sample_count_label.setObjectName("target_region_sample_count")
        self.sample_count_label.setWordWrap(True)
        self.sample_count_label.setToolTip(
            "Actual retained target samples from the current committed model. Resolution is a grid control, not the sample count."
        )
        resolution_row.fieldItem.widget().layout().addWidget(self.sample_count_label)
        if display:
            resolution = window._draft_controls["target_region_resolution"]
            method = (
                "Volume: bins per local bounding-box axis, then filtered to the region interior (up to resolution³ candidates). This grid can contain substantially more samples than the legacy sphere at the same resolution."
                if display["sampling"] == "volume"
                else "Imported surface: 6 × resolution² samples distributed by triangle area across the complete mesh."
                if self.shape == "enclosing_mesh"
                else "Analytic surface: angular or face resolution proportional to this value; samples constrain the boundary only."
            )
            resolution.setToolTip(
                f"{method} Preview reports the actual retained sample count. Independent validation has its own resolution under Study → Validation settings."
            )
        self.pose = {}
        if not imported_map and not legacy_mesh and not self.region_error and self.shape != "enclosing_mesh":
            window._check(
                form, "set_roi_into_mesh_center", "Center legacy sphere on support", config.set_roi_into_mesh_center
            )
            centering = window._draft_controls["set_roi_into_mesh_center"]
            centering.setEnabled(self.region is None)
            form.setRowVisible(centering.parentWidget(), self.region is None)
            initial = display or default_region(config)
            for vector, label, scale in (("center_m", "World center", 1e-3), ("rotation_axis", "Rotation axis", 1.0)):
                for i, axis in enumerate("xyz"):
                    key = f"roi_{vector}_{axis}"
                    window._number(
                        form,
                        key,
                        f"{label} {axis}",
                        initial[vector][i] / scale,
                        " mm" if vector == "center_m" else "",
                        -10000,
                        10000,
                    )
                    window._draft_fields[key] = ("target_region_geometry", "set_roi_into_mesh_center")
                    self.pose[key] = (vector, i, scale)
            window._number(
                form,
                "roi_angle_deg",
                "Rotation angle",
                math.degrees(initial["rotation_angle_rad"]),
                " °",
                -36000,
                36000,
            )
            window._draft_fields["roi_angle_deg"] = ("target_region_geometry", "set_roi_into_mesh_center")
            for key in (*self.pose, "roi_angle_deg"):
                control = window._draft_controls[key]
                control.setEnabled(not (self.region is None and config.set_roi_into_mesh_center))
            centering.toggled.connect(
                lambda checked: [
                    window._draft_controls[key].setEnabled(not checked) for key in (*self.pose, "roi_angle_deg")
                ]
            )

            def placement_summary():
                if centering.isChecked() and self.region is None:
                    return "Support vertex centroid"
                center = ", ".join(window._draft_controls[f"roi_center_m_{axis}"].text() for axis in "xyz")
                angle = window._draft_controls["roi_angle_deg"].text()
                return f"World center ({center}) · rotation {angle}"

            self.placement_section = group_form_controls(
                window,
                form,
                "Region placement",
                ("set_roi_into_mesh_center", *self.pose, "roi_angle_deg"),
                icon_name="placement",
                summary=placement_summary,
                apply_callback=window.apply_settings,
            )
            self.placement_section.setObjectName("roi_placement_details")
            self.placement_section.button.setToolTip(
                "Edit region position and orientation. The field expression and harmonic center remain in world coordinates."
            )
            self.focused_sections.append(self.placement_section)
        if imported_map:
            note("Samples use the imported map coordinates.")
        elif self.region_error:
            note(f"Invalid saved target region: {self.region_error} Open Choose field region to replace it explicitly.")
        elif legacy_mesh:
            note("Legacy mesh samples retained · vertices or bounding-box grid.")
        elif self.shape == "enclosing_mesh":
            note("Owned world mesh · volume requires a closed enclosure.")
        else:
            note("Surface sampling checks the boundary only.")
        if not imported_map:
            button = QPushButton("Import region surface…")
            button.setObjectName("import_region")
            button.setIcon(icon("roi-enclosing-mesh"))
            button.clicked.connect(window._import_region_dialog)
            layout.addWidget(button)

    def set_preflight(self, info):
        """Display committed preview evidence without constructing a new grid."""
        target = info.get("target", {}) if isinstance(info, dict) else {}
        count = target.get("sample_count")
        if target.get("available") and isinstance(count, int):
            active = target.get("active_sample_count")
            suffix = f" · {active:,} active" if isinstance(active, int) and active != count else ""
            self.sample_count_label.setText(f"{count:,} target samples{suffix} · current model")
        elif info and info.get("message"):
            self.sample_count_label.setText("Sample count unavailable · check model preview")
        else:
            self.sample_count_label.setText("Sample count available after preview")

    def changes(self):
        if self.imported_map or self.region_error:
            return {
                "target_region_geometry": deepcopy(self.region),
                "set_roi_into_mesh_center": self.config.set_roi_into_mesh_center,
            }
        window = self.window
        region = deepcopy(self.region)
        explicit = set(self.pose) | {"roi_angle_deg"}
        if region is None and explicit.intersection(window._changed_controls):
            region = default_region(self.config)
        if region is not None:
            for key, (vector, index, scale) in {**self.dimensions, **self.pose}.items():
                if key in window._changed_controls:
                    if vector not in region:
                        region[vector] = TargetRegion.from_dict(region).to_dict()[vector]
                    region[vector][index] = window._value(key) * scale
            if "target_region_radius" in window._changed_controls and self.shape in {"sphere", "cylinder"}:
                region["radius_m"] = window._value("target_region_radius") * 1e-3
            if "roi_length_mm" in window._changed_controls:
                region["length_m"] = window._value("roi_length_mm") * 1e-3
            if "roi_angle_deg" in window._changed_controls:
                region["rotation_angle_rad"] = math.radians(window._value("roi_angle_deg"))
        return {
            "target_region_geometry": region,
            "set_roi_into_mesh_center": False
            if region is not None
            else window._value("set_roi_into_mesh_center")
            if "set_roi_into_mesh_center" in window._draft_controls
            else self.config.set_roi_into_mesh_center,
        }
