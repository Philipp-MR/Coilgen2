"""Small editor schemas; native configuration remains the scientific model."""

from dataclasses import dataclass
import math
from types import SimpleNamespace


@dataclass(frozen=True, slots=True)
class GeometryField:
    key: str
    label: str
    index: int
    scale: float = 1e-3
    suffix: str = " mm"
    minimum: float = 0.01
    maximum: float = 10000
    integer: bool = False


@dataclass(frozen=True, slots=True)
class SupportEditor:
    label: str
    source: str
    parameter_field: str
    cylindrical: bool
    fields: tuple[GeometryField, ...]
    description: str = ""
    icon_name: str = "support-import"
    category: str = "Bore surfaces"

    def display_value(self, field, parameters):
        """Convert one native parameter to its labelled physical display."""
        value = parameters[field.index]
        if self.cylindrical and field.key == "length_mm":
            divisions = max(1, int(parameters[3]))
            value *= divisions / (divisions + 1)
        return value / field.scale

    def apply_edits(self, parameters, edits):
        """Translate explicit edits, preserving untouched native values exactly."""
        updated = list(parameters)
        for field in self.fields:
            if field.key in edits:
                updated[field.index] = edits[field.key] * field.scale
        if self.cylindrical and {"length_mm", "length_divisions"}.intersection(edits):
            previous_divisions = max(1, int(parameters[3]))
            span = (
                edits["length_mm"] * 1e-3
                if "length_mm" in edits
                else parameters[0] * previous_divisions / (previous_divisions + 1)
            )
            divisions = max(1, int(updated[3]))
            # The legacy cylinder drops its last axial row. A GUI refinement
            # preserves physical span without changing script-kernel semantics.
            updated[0] = span * (divisions + 1) / divisions
        self.validate_parameters(updated)
        return updated

    def validate_parameters(self, parameters):
        """Reuse backend bounds, then guard known degenerate GUI primitives."""
        from ..config_validation import _validate_generated_mesh_parameters

        errors = []
        _validate_generated_mesh_parameters(
            SimpleNamespace(coil_mesh_file=self.source, **{self.parameter_field: parameters}),
            lambda _field, message: errors.append(message),
        )
        if errors:
            raise ValueError(" ".join(errors))
        rotation = next((field for field in self.fields if field.key == "rotation_angle_deg"), None)
        if rotation is not None and len(parameters) > rotation.index and parameters[rotation.index] != 0:
            axis = parameters[rotation.index - 3 : rotation.index]
            # Match the builder's identity fallback; do not change its script API.
            if math.hypot(*axis) < 1e-15:
                raise ValueError("Rotation axis must be nonzero when rotation angle is nonzero.")
        normal = [field for field in self.fields if field.key.startswith("plane_normal_")]
        if normal and math.hypot(*(parameters[field.index] for field in normal)) < 1e-15:
            raise ValueError("Surface normal must be nonzero.")
        if self.source == "create circular mesh" and parameters[1] < 4:
            raise ValueError("Disc grid requires at least four divisions to retain usable triangles.")
        if self.source == "create annular disc mesh" and parameters[0] <= 0:
            raise ValueError("Annulus inner radius must be positive; use Disc for a surface without a central opening.")
        polar = next((field for field in self.fields if field.key == "polar_angle_deg"), None)
        if polar is not None and parameters[polar.index] >= math.pi - 1e-12:
            raise ValueError("Open cap polar angle must be less than 180° to avoid a collapsed rim.")
        if self.source in {"create partial cylinder mesh", "create toroidal segment mesh"}:
            if parameters[3] - parameters[2] >= 2 * math.pi - 1e-12:
                raise ValueError("Open segment span must be less than 360°; coincident end seams are not welded.")
        if self.source == "create circular bi-planary mesh" and math.ceil(2 * parameters[1] / parameters[0]) > 512:
            raise ValueError("Circular biplanar grid exceeds 512 divisions; increase the grid-step target.")


def _rotation(start):
    return tuple(
        GeometryField(f"rotation_axis_{axis}", f"Rotation axis {axis}", start + index, 1, "", -1e6, 1e6)
        for index, axis in enumerate("xyz")
    ) + (GeometryField("rotation_angle_deg", "Rotation angle", start + 3, math.pi / 180, " °", -36000, 36000),)


def _angle(key, label, index, minimum=-36000, maximum=36000):
    return GeometryField(key, label, index, math.pi / 180, " °", minimum, maximum)


def _normal(start):
    return tuple(
        GeometryField(f"plane_normal_{axis}", f"Surface normal {axis}", start + index, 1, "", -1e6, 1e6)
        for index, axis in enumerate("xyz")
    )


def _center(start, axes="xyz"):
    return tuple(
        GeometryField(f"center_{axis}_mm", f"World center {axis}", start + index, minimum=-10000)
        for index, axis in enumerate(axes)
    )


def _divisions(key, label, index, minimum=2):
    return GeometryField(key, label, index, 1, "", minimum, 512, True)


SUPPORT_EDITORS = (
    SupportEditor(
        "Cylinder",
        "create cylinder mesh",
        "cylinder_mesh_parameter_list",
        True,
        (
            GeometryField("radius_mm", "Radius", 1),
            GeometryField("length_mm", "Length", 0),
            _divisions("angular_divisions", "Angular divisions", 2, 4),
            _divisions("length_divisions", "Axial divisions", 3),
            *_rotation(4),
        ),
        "Open cylinder centered at the world origin; local axis z. Length is the actual mesh span; axial refinement preserves it. Rotation is right-handed about the given axis.",
        "support-cylinder",
    ),
    SupportEditor(
        "Biplanar",
        "create bi-planary mesh",
        "biplanar_mesh_parameter_list",
        False,
        (
            GeometryField("width_mm", "Width", 1),
            GeometryField("height_mm", "Height", 0),
            GeometryField("separation_mm", "Plane separation", 10),
            _divisions("lateral_divisions", "Lateral divisions", 2),
            _divisions("longitudinal_divisions", "Longitudinal divisions", 3),
            *tuple(
                GeometryField(f"plane_normal_{axis}", f"Plane normal {axis}", 4 + index, 1, "", -1e6, 1e6)
                for index, axis in enumerate("xyz")
            ),
            *_center(7),
        ),
        "Two rectangular supports, normal to the specified world vector. Separation is the full gap along that normal; world center translates the pair.",
        "support-biplanar",
        "Paired surfaces",
    ),
    SupportEditor(
        "Conical frustum",
        "create conical frustum mesh",
        "conical_frustum_mesh_parameter_list",
        False,
        (
            GeometryField("bottom_radius_mm", "Bottom radius (local −z)", 1),
            GeometryField("top_radius_mm", "Top radius (local +z)", 2),
            GeometryField("length_mm", "Axial length", 0),
            _divisions("angular_divisions", "Angular divisions", 3, 3),
            _divisions("length_divisions", "Axial divisions", 4, 1),
            *_rotation(5),
            *_center(9),
        ),
        "Open frustum with local axis z and positive radii at local −z and +z. World center is translated after the right-handed rotation.",
        "support-frustum",
    ),
    SupportEditor(
        "Rectangle",
        "create planary mesh",
        "planar_mesh_parameter_list",
        False,
        (
            GeometryField("width_mm", "Width (local y)", 1),
            GeometryField("height_mm", "Height (local x)", 0),
            _divisions("lateral_divisions", "Width divisions", 2, 1),
            _divisions("longitudinal_divisions", "Height divisions", 3, 1),
            *_rotation(4),
            *_center(8),
        ),
        "Single rectangular surface in local xy. The native builder measures height along local x and width along local y. World center is translated after rotation.",
        "support-plane",
        "Planar surfaces",
    ),
    SupportEditor(
        "Disc",
        "create circular mesh",
        "circular_mesh_parameter_list",
        False,
        (
            GeometryField("radius_mm", "Radius", 0),
            _divisions("radial_divisions", "Grid divisions across diameter", 1, 4),
            *_rotation(2),
            *_center(6),
        ),
        "Single circular surface in local xy. A Cartesian grid is clipped and its boundary projected to the radius; division count is across the diameter. World center is translated after rotation.",
        "support-disc",
        "Planar surfaces",
    ),
    SupportEditor(
        "Annulus",
        "create annular disc mesh",
        "annular_disc_mesh_parameter_list",
        False,
        (
            GeometryField("inner_radius_mm", "Inner radius", 0),
            GeometryField("outer_radius_mm", "Outer radius", 1),
            _divisions("radial_divisions", "Radial divisions", 2, 1),
            _divisions("angular_divisions", "Angular divisions", 3, 3),
            *_normal(4),
            *_center(7),
        ),
        "Annular surface with a positive inner radius and a larger outer radius. The specified world normal sets its orientation; world center translates it. Choose Disc when no central opening is needed.",
        "support-annulus",
        "Planar surfaces",
    ),
    SupportEditor(
        "Partial cylinder",
        "create partial cylinder mesh",
        "partial_cylinder_mesh_parameter_list",
        False,
        (
            GeometryField("length_mm", "Axial length", 0),
            GeometryField("radius_mm", "Radius", 1),
            _angle("start_angle_deg", "Sector start", 2),
            _angle("end_angle_deg", "Sector end", 3),
            _divisions("angular_divisions", "Sector divisions", 4, 1),
            _divisions("length_divisions", "Axial divisions", 5, 1),
            *_rotation(6),
            *_center(10),
        ),
        "Open cylindrical sector with local axis z. Angles run from local +x toward +y; end must exceed start and span must remain below 360°. Both axial ends and angular seams stay open.",
        "support-partial-cylinder",
    ),
    SupportEditor(
        "Elliptical cylinder",
        "create elliptical cylinder mesh",
        "elliptical_cylinder_mesh_parameter_list",
        False,
        (
            GeometryField("length_mm", "Axial length", 0),
            GeometryField("radius_x_mm", "Radius (local x)", 1),
            GeometryField("radius_y_mm", "Radius (local y)", 2),
            _divisions("angular_divisions", "Angular divisions", 3, 3),
            _divisions("length_divisions", "Axial divisions", 4, 1),
            *_rotation(5),
            *_center(9),
        ),
        "Open elliptical cylinder with local axis z. Radii are semi-axes, not diameters; axial length is the full mesh span. World center is translated after rotation.",
        "support-elliptical-cylinder",
    ),
    SupportEditor(
        "Superellipse",
        "create superellipse cylinder mesh",
        "superellipse_cylinder_mesh_parameter_list",
        False,
        (
            GeometryField("length_mm", "Axial length", 0),
            GeometryField("radius_x_mm", "Half extent (local x)", 1),
            GeometryField("radius_y_mm", "Half extent (local y)", 2),
            GeometryField("exponent", "Profile exponent n", 3, 1, "", 0.500001, 100),
            _divisions("angular_divisions", "Profile divisions", 4, 8),
            _divisions("length_divisions", "Axial divisions", 5, 1),
            *_rotation(6),
            *_center(10),
        ),
        "Open superellipse cylinder with local axis z. n=2 gives an ellipse; n=4 gives a rounded rectangle. Half extents set the cross-section, and the exponent must exceed 0.5. World center is translated after rotation.",
        "support-superellipse",
    ),
    SupportEditor(
        "Circular biplanar",
        "create circular bi-planary mesh",
        "circular_biplanar_mesh_parameter_list",
        False,
        (
            GeometryField("radius_mm", "Support radius", 1),
            GeometryField("separation_mm", "Plane separation", 8),
            GeometryField("mesh_edge_mm", "Grid-step target", 0),
            *_normal(2),
            *_center(5),
        ),
        "Two circular supports oriented by the world normal. Separation is the full gap; world center translates the pair. Grid-step target sets diameter divisions approximately, and is not a guaranteed triangular edge length.",
        "support-circular-biplanar",
        "Paired surfaces",
    ),
    SupportEditor(
        "Spherical cap",
        "create spherical cap mesh",
        "spherical_cap_mesh_parameter_list",
        False,
        (
            GeometryField("radius_mm", "Sphere radius", 0),
            _angle("polar_angle_deg", "Polar opening from local +z", 1, 0.0001, 179.999999),
            _divisions("angular_divisions", "Azimuthal divisions", 2, 3),
            _divisions("polar_divisions", "Polar divisions", 3, 1),
            *_rotation(4),
            *_center(8),
        ),
        "Open cap measured from the local +z pole: 90° is a hemisphere. The rim remains open and the azimuthal seam is welded. Use less than 180° to avoid a collapsed south-pole ring. World center is translated after rotation.",
        "support-cap",
        "Curved caps and segments",
    ),
    SupportEditor(
        "Ellipsoidal cap",
        "create ellipsoidal cap mesh",
        "ellipsoidal_cap_mesh_parameter_list",
        False,
        (
            GeometryField("radius_x_mm", "Semi-axis (local x)", 0),
            GeometryField("radius_y_mm", "Semi-axis (local y)", 1),
            GeometryField("radius_z_mm", "Semi-axis (local z)", 2),
            _angle("polar_angle_deg", "Polar opening from local +z", 3, 0.0001, 179.999999),
            _divisions("angular_divisions", "Azimuthal divisions", 4, 3),
            _divisions("polar_divisions", "Polar divisions", 5, 1),
            *_rotation(6),
            *_center(10),
        ),
        "Open ellipsoidal cap measured from the local +z pole. Radii are semi-axes; 90° reaches the equatorial plane. The rim stays open, azimuthal seam is welded, and opening must be less than 180°. World center is translated after rotation.",
        "support-ellipsoidal-cap",
        "Curved caps and segments",
    ),
    SupportEditor(
        "Toroidal segment",
        "create toroidal segment mesh",
        "toroidal_segment_mesh_parameter_list",
        False,
        (
            GeometryField("major_radius_mm", "Major radius", 0),
            GeometryField("minor_radius_mm", "Tube radius", 1),
            _angle("start_angle_deg", "Major start", 2),
            _angle("end_angle_deg", "Major end", 3),
            _angle("tube_start_angle_deg", "Tube start", 4),
            _angle("tube_end_angle_deg", "Tube end", 5),
            _divisions("major_divisions", "Major divisions", 6, 3),
            _divisions("minor_divisions", "Tube divisions", 7, 3),
            *_rotation(8),
            *_center(12),
        ),
        "Open toroidal segment about local z; tube radius must be smaller than major radius. Major span is below 360° and its end seams stay open. Tube span up to 360° is supported, with the full tube seam welded. World center is translated after rotation.",
        "support-toroid",
        "Curved caps and segments",
    ),
)
