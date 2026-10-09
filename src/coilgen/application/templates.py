"""Portable starting designs shared by the GUI, notebooks and batch clients.

Templates specify physical inputs and a complete study; they contain no saved
acceptance claims. The original preset API and numerical defaults are retained.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np

from ..geometry import build_cylinder_mesh, combine_meshes, normalize_mesh
from ..models import Mesh
from .assembly import ProjectAssembly
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
    "Fit the current sheet, extract contours, connect the winding, sweep 2 × 2 mm copper "
    "and independently review fields and manufacturing geometry. Feeds and cooling require design."
)
_SHIM_DESCRIPTION = (
    "Cylindrical shim insert: 200 mm length, 80 mm support radius, 40 mm spherical DSV; "
    "B0 = 3 T along world z. Sweep 1 × 1 mm copper with 0.1 mm insulation per side. "
    "The absolute Bz target uses world coordinates and a fixed 20 mm reference radius. "
    "The full study retains discrete-wire and manufacturing failures for review; feeds and cooling require design."
)


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
        "biplanar_gradient_shim",
        SYSTEM_EXAMPLE_TITLES["biplanar_gradient_shim"],
        "Coil systems",
        system_description(),
        "biplanar-system",
    ),
    TemplateSpec(
        "independent_turns",
        EXAMPLE_TITLES["independent_turns"],
        "Demonstrations",
        preset_description("independent_turns"),
        "independent-turns",
    ),
)


def template_spec(id_: str) -> TemplateSpec:
    for spec in TEMPLATE_SPECS:
        if spec.id == id_:
            return spec
    raise ValueError(f"Unknown design template {id_!r}; choose {', '.join(spec.id for spec in TEMPLATE_SPECS)}.")


def _cylinder(length_m: float, radius_m: float, angular: int = 32, axial: int = 16) -> Mesh:
    # Preserve the legacy builder while giving templates their stated SI span.
    return normalize_mesh(build_cylinder_mesh(length_m * (axial + 1) / axial, radius_m, angular, axial, 1, 0, 0, 0))


def _closed_suppression_cylinder() -> Mesh:
    mesh = _cylinder(0.4, 0.15, angular=24, axial=12)
    angular = 24
    vertices = np.vstack((mesh.vertices, (0, 0, -0.2), (0, 0, 0.2)))
    bottom, top = len(mesh.vertices), len(mesh.vertices) + 1
    last_ring = len(mesh.vertices) - angular
    end_faces = []
    for index in range(angular):
        next_index = (index + 1) % angular
        end_faces.extend(((bottom, index, next_index), (top, last_ring + next_index, last_ring + index)))
    return normalize_mesh(
        Mesh(
            vertices=vertices,
            faces=np.vstack((mesh.faces, np.asarray(end_faces, dtype=np.int64))),
            metadata={
                "geometry_type": "closed_external_suppression_cylinder",
                "units": "SI",
                "radius_m": 0.15,
                "length_m": 0.4,
            },
        )
    )


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
    support = combine_meshes([_cylinder(0.3, 0.08), _cylinder(0.3, 0.11)])
    support.metadata.update(
        {"geometry_type": "active_shielded_cylinder", "units": "SI", "part_roles": ["primary", "active_shield"]}
    )
    return (
        ProjectDocument.from_config(config, name=spec.title, project_root=root)
        .with_mesh(support, field="coil_mesh_file")
        .with_mesh(_closed_suppression_cylinder(), field="secondary_target_mesh_file")
        .update_study(preset_study("biplanar"))
    )


def _shim_config(id_: str):
    config = replace(
        preset_config("cylinder"),
        design_context="mr_shim",
        conductor_cross_section_width=0.001,
        conductor_cross_section_height=0.001,
        sweep_min_bend_radius=0.003,
        levels=12,
        target_error_tolerance_percent=4.0,
        target_gradient_normalization="none",
        normal_shift_length=0.001,
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

    All lengths and fields use SI and the world origin. Active shielding is a
    joint current-sheet solve with an explicit separate zero-field target; its
    support and closed suppression surface are embedded in the project JSON.
    """
    spec = template_spec(id_)
    root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
    if id_ in SYSTEM_EXAMPLE_TITLES:
        return system_assembly(id_, project_root=root)
    if id_ == "shielded_gy":
        project = _shielded_project(spec, root)
    else:
        if id_ in EXAMPLE_TITLES:
            config = preset_config(id_)
            study = preset_study(id_)
        elif id_.startswith("shim_"):
            config = _shim_config(id_)
            study = preset_study("biplanar")
        else:
            support, axis = id_.split("_g")
            kind = "cylinder" if support == "cylinder" else "biplanar"
            config = replace(preset_config(kind), field_shape_function=axis, target_gradient_axis=axis)
            study = preset_study(kind)
        project = ProjectDocument.from_config(config, name=spec.title, project_root=root).update_study(study)
    return ProjectAssembly.new(spec.title, (project,))
