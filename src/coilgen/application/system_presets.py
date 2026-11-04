"""Reproducible independent-channel starting systems in a common SI world frame.

These examples request complete per-channel studies. They do not certify
inter-channel conductor clearance, mutual inductance, external feeds, combined
drive limits, thermal operation or construction release.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from .assembly import ChannelDocument, ProjectAssembly
from .gradient_geometry import CylinderSurface, CylindricalGeometry
from .presets import preset_config, preset_study
from .project import ProjectDocument
from .targets import SolidHarmonicTarget


SYSTEM_EXAMPLE_TITLES = {
    "biplanar_gradient_shim": "Biplanar gradients + second-order shims",
    "cylindrical_xyz_gradient": "Cylindrical XYZ gradient system (Gx, Gy, Gz)",
    "shielded_xyz_gradient": "Actively shielded XYZ foil gradient system (Gx, Gy, Gz)",
}
SYSTEM_PRESET_NAMES = tuple(SYSTEM_EXAMPLE_TITLES)

BIPLANAR_ROI_RADIUS_M = 0.02
BIPLANAR_GRADIENT_T_PER_M = 0.02
BIPLANAR_Z0_FIELD_T = 100e-6
BIPLANAR_SECOND_ORDER_PEAK_T = 10e-6
BIPLANAR_PLANE_PITCH_M = 0.006

CYLINDRICAL_LENGTH_M = 0.20
CYLINDRICAL_SUPPORT_RADII_M = (0.080, 0.086, 0.092)
CYLINDRICAL_ROI_RADIUS_M = 0.020
CYLINDRICAL_GRADIENT_T_PER_M = 0.020
SHIELD_RADII_M = (0.110, 0.116, 0.122)

# Names are canonical Cartesian Bz modes; gradients supply the complete degree-1
# basis. Positive XZ/YZ require negative real-Y2,+/-1 coefficients because the
# existing harmonic evaluator includes the Condon--Shortley phase.
_MODES = (
    ("gx", "Gx", 1, None, "Bz = 0.02*x T; dBz/dx = 20 mT/m"),
    ("gy", "Gy", 1, None, "Bz = 0.02*y T; dBz/dy = 20 mT/m"),
    ("gz", "Gz", 1, None, "Bz = 0.02*z T; dBz/dz = 20 mT/m"),
    ("z0", "Z0", 0, 0, "Bz = 100 µT; a uniform perturbation of the 3 T background"),
    ("z2", "Z2", 2, 0, "Bz = A*(2*z²-x²-y²)/(2*R²)"),
    ("xz", "XZ", 2, 1, "Bz = A*2*x*z/R²"),
    ("yz", "YZ", 2, -1, "Bz = A*2*y*z/R²"),
    ("x2-y2", "X²−Y²", 2, 2, "Bz = A*(x²-y²)/R²"),
    ("xy", "XY", 2, -2, "Bz = A*2*x*y/R²"),
)


def system_description(kind="biplanar_gradient_shim") -> str:
    if kind not in SYSTEM_EXAMPLE_TITLES:
        raise ValueError(f"Unknown system example {kind!r}; choose {', '.join(SYSTEM_PRESET_NAMES)}.")
    if kind == "shielded_xyz_gradient":
        return (
            "Three independent 20 mT/m Bz gradients over a common 40 mm DSV. "
            "Gx, Gy and Gz each have a jointly optimized primary and active shield "
            "with a common winding current. Primary radii are 80, 86 and 92 mm. "
            "Shield radii are 110, 116 and 122 mm. Each 300 mm support carries "
            "2 mm wide, 0.20 mm thick copper foil. The 20 µT area-weighted vector RMS "
            "limit applies to stream function fields sampled on the common closed cylinder "
            "of radius 150 mm and length 400 mm. Routed fields and physical clearance "
            "require separate review."
        )
    if kind == "cylindrical_xyz_gradient":
        return (
            "Three independently driven Gx, Gy and Gz channels on concentric cylindrical "
            "supports, all 200 mm long and coaxial with world z. Support radii are "
            "80, 86 and 92 mm respectively. Each targets Bz = 0.02*x, 0.02*y or "
            "0.02*z T in the same world-origin 40 mm spherical DSV; B₀ = 3 T along "
            "world z. Each channel uses 2 × 2 mm copper, 0.1 mm insulation per side, "
            "18 nΩ·m resistivity, a 6 mm minimum bend radius and 16 contour levels. "
            "The 1 mm homogenized optimization sheet thickness is separate from the "
            "physical conductor size. A complete study fits each stream function to 3% RMS field error, "
            "checks it against 5% on a distinct grid, extracts contours, connects "
            "the winding, sweeps conductors and runs strict fabrication review at "
            "the solved current. This unshielded insert example supplies nominal "
            "6 mm support spacing, not inter-channel conductor-clearance evidence. "
            "Routes, feeds, mutual inductance, external stray field, drive current "
            "and voltage, cooling and duty cycle require system qualification. "
            "Per-channel checks do not establish a combined hardware pass or "
            "construction release."
        )
    return (
        "Nine independently driven channels: Gx, Gy, Gz, Z0 and the five second-order "
        "Bz shims Z2, XZ, YZ, X²−Y² and XY. All use the same world origin, a 40 mm "
        "spherical DSV and B₀ = 3 T along world z. Gradients are 20 mT/m; Z0 is "
        "100 µT. Each quadratic mode has maximum |Bz| = 10 µT on the 20 mm ROI "
        "sphere, with explicit real-Ylm coefficient conversion. Each channel owns "
        "a 300 × 300 mm biplanar support normal to world x. Nominal layer positions "
        "are x = ±(50 + 6i) mm, i = 0…8, giving 100…196 mm plane separation. "
        "Gradient copper is 2 × 2 mm; shim copper is 1 × 1 mm; all use 0.1 mm "
        "insulation per side and 18 nΩ·m resistivity. The 1 mm homogenized sheet "
        "thickness is a separate optimization assumption. Each study fits the "
        "stream function to 4% RMS field error, checks it against 5% on a distinct grid, extracts "
        "contours, connects each support circuit, sweeps conductors and runs "
        "strict fabrication review. These nominally separated layers are a "
        "starting layout: routes, feeds, inter-channel clearance/coupling, drive "
        "voltage, cooling and duty cycle require system qualification. Per-channel "
        "checks do not establish a combined hardware pass or construction release."
    )


def _target(key, degree, order):
    if degree == 1:
        return {
            "field_shape_function": key[-1],
            "target_gradient_axis": key[-1],
            "target_gradient_strength": BIPLANAR_GRADIENT_T_PER_M,
            "target_gradient_normalization": "linear_fit",
            "target_field_definition_file": "none",
            "target_field_units": "T",
            "target_field_component": "Bz",
            "design_context": "mr_gradient",
        }
    if degree == 0:
        return {
            "field_shape_function": repr(BIPLANAR_Z0_FIELD_T),
            "target_gradient_strength": BIPLANAR_Z0_FIELD_T,
            "target_gradient_normalization": "none",
            "target_field_definition_file": "none",
            "target_field_units": "T",
            "target_field_component": "Bz",
            "design_context": "mr_shim",
        }
    peak = BIPLANAR_SECOND_ORDER_PEAK_T
    coefficient = peak / math.sqrt(5 / (4 * math.pi)) if order == 0 else peak / math.sqrt(15 / (16 * math.pi))
    if abs(order) == 1:
        coefficient = -coefficient
    return SolidHarmonicTarget(2, order, coefficient, BIPLANAR_ROI_RADIUS_M).to_config_overrides()


def _cylindrical_xyz_assembly(root: Path) -> ProjectAssembly:
    base = preset_config("cylinder")
    _, _, angular, axial, *orientation = base.cylinder_mesh_parameter_list
    # Match the preset's compensation for the legacy builder's missing end row.
    length_parameter = CYLINDRICAL_LENGTH_M * (axial + 1) / axial
    channels = []
    for axis, radius in zip("xyz", CYLINDRICAL_SUPPORT_RADII_M):
        name = f"G{axis}"
        config = replace(
            base,
            cylinder_mesh_parameter_list=(length_parameter, radius, angular, axial, *orientation),
            field_shape_function=axis,
            target_gradient_axis=axis,
            target_gradient_strength=CYLINDRICAL_GRADIENT_T_PER_M,
            target_region_radius=CYLINDRICAL_ROI_RADIUS_M,
            set_roi_into_mesh_center=False,
            b_0_direction=(0, 0, 1),
            symmetry_reduction="off",
            layout_manufacturing_intent=(
                "Independent cylindrical gradient channel; inter-channel clearance, "
                "feeds, stray field and drive qualification remain unresolved."
            ),
        )
        project = ProjectDocument.from_config(config, name=f"Cylindrical {name}", project_root=root).update_study(
            preset_study("cylinder")
        )
        project = project.with_results(
            project.saved_runs,
            session={
                "system_example": {
                    "kind": "cylindrical_xyz_gradient",
                    "mode": name,
                    "target_definition": f"Bz = 0.02*{axis} T; dBz/d{axis} = 20 mT/m",
                    "coordinate_frame": "world",
                    "reference_radius_m": CYLINDRICAL_ROI_RADIUS_M,
                    "nominal_support_radius_m": radius,
                    "nominal_support_length_m": CYLINDRICAL_LENGTH_M,
                    "scope": "Independent channel; no combined assembly qualification",
                }
            },
        )
        channels.append(ChannelDocument(f"g{axis}", name, project))
    return replace(ProjectAssembly.new(SYSTEM_EXAMPLE_TITLES["cylindrical_xyz_gradient"]), channels=tuple(channels))


def _shielded_xyz_assembly(root: Path) -> ProjectAssembly:
    base = replace(
        preset_config("cylinder"),
        conductor_cross_section_width=0.002,
        conductor_cross_section_height=0.0002,
        normal_shift_length=0.003,
        conductor_thickness=0.0002,
        target_region_radius=CYLINDRICAL_ROI_RADIUS_M,
        target_gradient_strength=CYLINDRICAL_GRADIENT_T_PER_M,
        secondary_target_weight=1.0,
        suppression_field_component="vector",
        suppression_surface_weighting="area",
        suppression_field_rms_limit_T=20e-6,
        target_error_tolerance_percent=4.0,
        set_roi_into_mesh_center=False,
        symmetry_reduction="off",
        level_set_method="primary",
        layout_manufacturing_intent="Foil windings on six cylindrical supports. Interlayer feeds require design.",
    )
    suppression = CylinderSurface(0.150, 0.400, angular_divisions=24, axial_divisions=12)
    channels = []
    for axis, radius, shield_radius in zip("xyz", CYLINDRICAL_SUPPORT_RADII_M, SHIELD_RADII_M):
        name = f"G{axis}"
        geometry = CylindricalGeometry(
            CylinderSurface(radius, 0.300), CylinderSurface(shield_radius, 0.300), suppression
        )
        config = replace(
            base,
            field_shape_function=axis,
            target_gradient_axis=axis,
            sweep_corner_cluster_max_source_deviation_m=0.006 if axis in {"y", "z"} else 0.0,
        )
        project = ProjectDocument.from_config(config, name=f"Shielded {name}", project_root=root)
        project = geometry.apply(project).update_study({**preset_study("cylinder"), "design_margin": 0.8})
        channels.append(ChannelDocument(f"g{axis}", name, project))
    return replace(ProjectAssembly.new(SYSTEM_EXAMPLE_TITLES["shielded_xyz_gradient"]), channels=tuple(channels))


def system_assembly(kind="biplanar_gradient_shim", *, project_root=None) -> ProjectAssembly:
    """Return independent owned channels with explicit targets and full studies.

    Changing or computing one channel does not change another channel's model.
    The GUI and original script pipeline consume the same configurations.
    """
    system_description(kind)
    root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
    if kind == "cylindrical_xyz_gradient":
        return _cylindrical_xyz_assembly(root)
    if kind == "shielded_xyz_gradient":
        return _shielded_xyz_assembly(root)
    base = preset_config("biplanar")
    channels = []
    for index, (key, name, degree, order, polynomial) in enumerate(_MODES):
        gradient = degree == 1
        separation = 0.10 + 2 * index * BIPLANAR_PLANE_PITCH_M
        width = 0.002 if gradient else 0.001
        # These two shims retain a lower-error connected winding on the 14 x 14 support.
        subdivisions = 14 if key in {"z2", "x2-y2"} else 24
        config = replace(
            base,
            **_target(key, degree, order),
            biplanar_mesh_parameter_list=(0.30, 0.30, subdivisions, subdivisions, 1, 0, 0, 0, 0, 0, separation),
            target_region_radius=BIPLANAR_ROI_RADIUS_M,
            target_region_resolution=17,
            set_roi_into_mesh_center=False,
            b_0_strength_T=3.0,
            b_0_direction=(0, 0, 1),
            conductor_cross_section_width=width,
            conductor_cross_section_height=width,
            # Leave space for both insulating layers, mounting tolerance and
            # the tilted outer corners of 2 mm copper on the 6 mm stack pitch.
            normal_shift_length=0.0035 if gradient else 2 * width,
            sweep_corner_cluster_max_source_deviation_m=(
                0.006 if key in {"gx", "gy", "z2", "x2-y2"} else 0.003 if key in {"xy", "yz"} else 0.0
            ),
            sweep_min_bend_radius=3 * width,
            levels=16 if gradient else 12,
            symmetry_reduction="off",
            sweep_surface_realization="mesh",
            # In this Gy layer the high cut for group 2 avoids a near-reversing
            # return on the second plate. All other group choices stay automatic.
            force_cut_selection=("auto", "auto", "high", "auto") if key == "gy" else base.force_cut_selection,
            layout_manufacturing_intent="Independent biplanar channel; inter-channel clearance, feeds and drive qualification remain unresolved.",
        )
        project = ProjectDocument.from_config(config, name=f"Biplanar {name}", project_root=root).update_study(
            preset_study("biplanar")
        )
        project = project.with_results(
            project.saved_runs,
            session={
                "system_example": {
                    "kind": kind,
                    "mode": name,
                    "degree": degree,
                    "order": order,
                    "target_definition": polynomial,
                    "quadratic_amplitude_T": BIPLANAR_SECOND_ORDER_PEAK_T if degree == 2 else None,
                    "reference_radius_m": BIPLANAR_ROI_RADIUS_M,
                    "coordinate_frame": "world",
                    "nominal_half_separation_m": separation / 2,
                    "scope": "Independent channel; no combined assembly qualification",
                }
            },
        )
        channels.append(ChannelDocument(key, name, project))
    return replace(ProjectAssembly.new(SYSTEM_EXAMPLE_TITLES[kind]), channels=tuple(channels))
