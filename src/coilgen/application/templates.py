"""Portable starting designs shared by the GUI, notebooks and batch clients.

Templates specify physical inputs, bounded pre-routing contour smoothing and
a complete study. Acceptance is measured when the user computes the design.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

from .assembly import ProjectAssembly
from .gradient_geometry import CylinderSurface, CylindricalGeometry
from .presets import EXAMPLE_TITLES, preset_config, preset_description, preset_study
from .project import ProjectDocument
from .system_presets import SYSTEM_EXAMPLE_TITLES, system_assembly, system_description
from .targets import SolidHarmonicTarget


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    id: str
    title: str
    category: str
    description: str
    icon: str


_GRADIENT_WORKFLOW = (
    "Bz gradient of 20 mT/m in a 40 mm spherical DSV, with B0 = 3 T along world z. "
    "Fit the stream function, extract contours, connect the winding, sweep 2 × 2 mm copper "
    "and independently review fields and manufacturing geometry. Feeds and cooling require design."
)
_SHIM_DESCRIPTION = (
    "Cylindrical shim insert: 200 mm length, 80 mm support radius, 40 mm spherical DSV; "
    "B0 = 3 T along world z. Sweep 1 × 1 mm copper with 0.1 mm insulation per side. "
    "The absolute Bz target uses world coordinates and a fixed 20 mm reference radius. "
    "The full study retains discrete-wire and manufacturing failures for review; feeds and cooling require design."
)


_HALBACH_DESCRIPTION = (
    "This gallery example generates a 200 mm long cylindrical support of 80 mm radius, rotated 90° about "
    "world x so the bore runs along y while B0 = 50 mT points along world z (vertical). "
    "Bz = 0.02*{axis} T in a 40 mm spherical DSV. Interconnection direction: {routing}. "
    "The separate Halbach research project uses octagonal STL surfaces. "
    "The winding uses 2 × 2 mm copper and requires feed and cooling design."
)
HALBACH_BORE_AXIS = (0.0, 1.0, 0.0)
# Measured on these templates (2-3 Nov 2026): no routing axis is better in
# general. The "centerline intersection" first reported for Gz along B0 was a
# false positive of the contact check (near-duplicate route vertices); the true
# minimum distance is 1.33 mm. For Gx, routing along the bore collides two
# connectors (0.03 mm). Keep the conventional B0 routing; interconnection_axis
# remains available per design. These reference measurements predate the
# mounted-copper clearance check; each new layout is independently swept.
HALBACH_INTERCONNECTION_AXIS = {"x": "b0", "y": "b0", "z": "b0"}
_HALBACH_ROUTING_TEXT = {
    "b0": "along B0 (the interconnection direction can be switched to the bore axis per design)",
    "support_axis": "along the bore axis",
}


def _gradient_description(support: str, axis: str) -> str:
    geometry = (
        "200 mm long cylindrical support of 80 mm radius. "
        if support == "cylinder"
        else "Two 200 × 200 mm planes normal to world x, separated by 100 mm. "
    )
    return geometry + f"Bz = 0.02*{axis} T, with {axis} in metres in the world frame. " + _GRADIENT_WORKFLOW


TEMPLATE_SPECS = (
    TemplateSpec(
        "cylinder_gx", "Cylindrical X gradient (Gx)", "Gradients", _gradient_description("cylinder", "x"), "gradient-x"
    ),
    TemplateSpec("cylinder", EXAMPLE_TITLES["cylinder"], "Gradients", preset_description("cylinder"), "gradient-y"),
    TemplateSpec(
        "cylinder_gz", "Cylindrical Z gradient (Gz)", "Gradients", _gradient_description("cylinder", "z"), "gradient-z"
    ),
    *(
        TemplateSpec(
            f"halbach_g{axis}",
            f"Halbach transverse-field {axis.upper()} gradient (G{axis})",
            "Gradients",
            _HALBACH_DESCRIPTION.format(axis=axis, routing=_HALBACH_ROUTING_TEXT[HALBACH_INTERCONNECTION_AXIS[axis]]),
            f"gradient-{axis}",
        )
        for axis in "xyz"
    ),
    TemplateSpec(
        "shielded_gy",
        "Actively shielded Y gradient (Gy)",
        "Gradients",
        "Two jointly optimized concentric cylindrical current supports: primary radius 80 mm, "
        "shield radius 110 mm, each 300 mm long. Gy = 20 mT/m in a 40 mm spherical DSV. "
        "A separate closed cylinder of radius 150 mm and length 400 mm requests zero external "
        "vector field, with a 20 µT RMS limit at its sampled vertices. "
        "Each support receives a separate winding at the common contour current, using 2 × 2 mm copper. "
        "This sampling constraint does not certify the field everywhere outside, eddy currents, "
        "magnet-interface coupling, inter-support feeds or a buildable assembly.",
        "shielded-gradient",
    ),
    TemplateSpec(
        "biplanar_gx", "Biplanar X gradient (Gx)", "Gradients", _gradient_description("biplanar", "x"), "gradient-x"
    ),
    TemplateSpec("biplanar", EXAMPLE_TITLES["biplanar"], "Gradients", preset_description("biplanar"), "gradient-y"),
    TemplateSpec(
        "biplanar_gz", "Biplanar Z gradient (Gz)", "Gradients", _gradient_description("biplanar", "z"), "gradient-z"
    ),
    TemplateSpec(
        "shim_z0", "Uniform B0 shim (Z0)", "Shim coils", "Uniform Bz = 100 µT. " + _SHIM_DESCRIPTION, "shim-z0"
    ),
    TemplateSpec(
        "shim_z2",
        "Second-order shim — Z2",
        "Shim coils",
        "Peak |Bz| = 10 µT: Z2. " + _SHIM_DESCRIPTION,
        "shim-z2",
    ),
    TemplateSpec(
        "shim_xz",
        "Second-order shim — XZ",
        "Shim coils",
        "Peak |Bz| = 10 µT: XZ. " + _SHIM_DESCRIPTION,
        "shim-xz",
    ),
    TemplateSpec(
        "shim_yz",
        "Second-order shim — YZ",
        "Shim coils",
        "Peak |Bz| = 10 µT: YZ. " + _SHIM_DESCRIPTION,
        "shim-yz",
    ),
    TemplateSpec(
        "shim_x2_y2",
        "Second-order shim — X²−Y²",
        "Shim coils",
        "Peak |Bz| = 10 µT: X²−Y². " + _SHIM_DESCRIPTION,
        "shim-x2-y2",
    ),
    TemplateSpec(
        "shim_xy",
        "Second-order shim — XY",
        "Shim coils",
        "Peak |Bz| = 10 µT: XY. " + _SHIM_DESCRIPTION,
        "shim-xy",
    ),
    TemplateSpec(
        "conical_harmonic",
        EXAMPLE_TITLES["conical_harmonic"],
        "Shim coils",
        preset_description("conical_harmonic"),
        "shim-z2",
    ),
    TemplateSpec(
        "cylindrical_xyz_gradient",
        SYSTEM_EXAMPLE_TITLES["cylindrical_xyz_gradient"],
        "Coil systems",
        system_description("cylindrical_xyz_gradient"),
        "support-cylinder",
    ),
    TemplateSpec(
        "shielded_xyz_gradient",
        SYSTEM_EXAMPLE_TITLES["shielded_xyz_gradient"],
        "Coil systems",
        system_description("shielded_xyz_gradient"),
        "shielded-gradient",
    ),
    TemplateSpec(
        "biplanar_gradient_shim",
        SYSTEM_EXAMPLE_TITLES["biplanar_gradient_shim"],
        "Coil systems",
        system_description(),
        "biplanar-system",
    ),
)


def template_spec(id_: str) -> TemplateSpec:
    for spec in TEMPLATE_SPECS:
        if spec.id == id_:
            return spec
    raise ValueError(f"Unknown design template {id_!r}; choose {', '.join(spec.id for spec in TEMPLATE_SPECS)}.")


def _shielded_project(spec: TemplateSpec, root: Path) -> ProjectDocument:
    config = replace(
        preset_config("cylinder"),
        target_error_tolerance_percent=4.0,
        routing_circuit_scope="per_part",
        symmetry_reduction="off",
        sweep_surface_realization="mesh",
        secondary_target_weight=1.0,
        suppression_field_component="vector",
        suppression_field_rms_limit_T=20e-6,
        suppression_surface_weighting="area",
        layout_manufacturing_intent="Primary and active-shield windings; inter-support feeds and external-field verification require engineering review.",
    )
    # The shared cylindrical geometry records its dimensions with the meshes, so
    # both workbenches can edit radii, lengths and openings of this example.
    geometry = CylindricalGeometry(
        CylinderSurface(0.08, 0.3), CylinderSurface(0.11, 0.3), CylinderSurface(0.15, 0.4, 24, 12)
    )
    project = ProjectDocument.from_config(config, name=spec.title, project_root=root)
    return geometry.apply(project).update_study(preset_study("biplanar"))


def _shim_config(id_: str):
    # Refining the axisymmetric Z2 support introduced a nearly reversing
    # connector in its reviewed wire. Preserve its original ring mesh.
    angular, axial = (32, 16) if id_ == "shim_z2" else (64, 32)
    config = replace(
        preset_config("cylinder"),
        # Resolve the high-curvature shim lobes before extracting level sets.
        # Keep the physical cylinder span at 200 mm with the legacy mesh builder.
        cylinder_mesh_parameter_list=(0.2 * (axial + 1) / axial, 0.08, angular, axial, 1, 0, 0, 0),
        design_context="mr_shim",
        conductor_cross_section_width=0.001,
        conductor_cross_section_height=0.001,
        sweep_min_bend_radius=0.003,
        # The other four cylindrical shims clear without moving a contour
        # tangentially to make room for a cut.
        sweep_corner_cluster_max_source_deviation_m=0.003 if id_ in {"shim_yz", "shim_z2"} else 0.0,
        levels=12,
        target_error_tolerance_percent=4.0,
        target_gradient_normalization="none",
        # Two nearby axial returns of Z2 cross the same turn: reserve a
        # third height without changing its stream function or winding levels.
        normal_shift_length=0.0038 if id_ == "shim_z2" else 0.002,
        symmetry_reduction="off",
    )
    if id_ == "shim_z0":
        return replace(config, field_shape_function="0.0001", target_gradient_strength=100e-6)
    order = {"shim_z2": 0, "shim_xz": 1, "shim_yz": -1, "shim_x2_y2": 2, "shim_xy": -2}[id_]
    coefficient = 10e-6 / math.sqrt(5 / (4 * math.pi)) if order == 0 else 10e-6 / math.sqrt(15 / (16 * math.pi))
    if abs(order) == 1:
        coefficient = -coefficient
    return replace(config, **SolidHarmonicTarget(2, order, coefficient, 0.02).to_config_overrides())


def template_assembly(id_: str, *, project_root=None) -> ProjectAssembly:
    """Instantiate independent owned inputs, ready for the complete workflow.

    All lengths and fields use SI and the world origin. The active shielding
    design fits the stream functions jointly with a separate zero-field target.
    Its support and closed suppression surface are embedded in the project JSON.
    """
    spec = template_spec(id_)
    root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
    if id_ in SYSTEM_EXAMPLE_TITLES:
        return _first_revision(system_assembly(id_, project_root=root))
    if id_ == "shielded_gy":
        project = _shielded_project(spec, root)
    else:
        if id_ in EXAMPLE_TITLES:
            config = preset_config(id_)
            study = preset_study(id_)
        elif id_.startswith("shim_"):
            config = _shim_config(id_)
            study = preset_study("biplanar")
        elif id_.startswith("halbach_g"):
            axis = id_.removeprefix("halbach_g")
            base = preset_config("cylinder")
            length, radius, angular, axial = base.cylinder_mesh_parameter_list[:4]
            config = replace(
                base,
                cylinder_mesh_parameter_list=(length, radius, angular, axial, 1, 0, 0, math.pi / 2),
                field_shape_function=axis,
                target_gradient_axis=axis,
                b_0_strength_T=0.05,
                research_physics={**base.research_physics, "b0_T": (0.0, 0.0, 0.05)},
                interconnection_axis=HALBACH_INTERCONNECTION_AXIS[axis],
            )
            study = preset_study("cylinder")
        else:
            support, axis = id_.split("_g")
            kind = "cylinder" if support == "cylinder" else "biplanar"
            config = replace(preset_config(kind), field_shape_function=axis, target_gradient_axis=axis)
            if id_ == "biplanar_gz":
                # The long crossing cuts pass above another 2 mm winding.
                config = replace(config, normal_shift_length=0.0055, sweep_corner_cluster_max_source_deviation_m=0.003)
            study = preset_study(kind)
        project = ProjectDocument.from_config(config, name=spec.title, project_root=root).update_study(study)
    return _first_revision(ProjectAssembly.new(spec.title, (project,)))


def _first_revision(assembly: ProjectAssembly) -> ProjectAssembly:
    """Present a freshly built example as revision 1 of each channel.

    Building a template chains several internal edits (inputs, owned meshes, study), each of which
    would otherwise count as a user-visible revision. Each channel has a new project id, so
    restarting the count cannot collide with any saved run.
    """
    channels = tuple(replace(channel, project=replace(channel.project, revision=1)) for channel in assembly.channels)
    return replace(assembly, channels=channels)
