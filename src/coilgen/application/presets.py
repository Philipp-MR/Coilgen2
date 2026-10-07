"""Physical starting models shared by the workbench and headless clients.

These are small insert-coil studies, not released hardware. All lengths,
fields and material values use SI units; original script defaults are intact.
"""

from __future__ import annotations

from dataclasses import replace
import math

from ..config import CoilGenConfig


EXAMPLE_TITLES = {
    "cylinder": "Cylindrical Y gradient (Gy)",
    "biplanar": "Biplanar Y gradient (Gy)",
    "independent_turns": "Independent-turn sweep demo",
    "conical_harmonic": "Conical Z2",
}
PRESET_NAMES = tuple(EXAMPLE_TITLES)

_DESCRIPTIONS = {
    "cylinder": (
        "Y-gradient insert: a 200 mm long cylinder of 80 mm support radius surrounds a 40 mm diameter "
        "spherical imaging volume (DSV). Gy = dBz/dy = 20 mT/m, with the 3 T main field along world z. "
        "The designed Bz perturbation is −0.4, 0 and +0.4 mT at y = −20, 0 and +20 mm. "
        "The study fits a continuous current sheet, extracts 16 contour levels, connects the winding, "
        "sweeps 2 × 2 mm copper (4 mm² area, 6 mm minimum bend radius), and checks its field at the original current. "
        "The 1 mm homogenized sheet thickness is a separate optimization assumption, not the physical wire size. "
        "A 3% sheet design RMS budget reserves headroom for discretization, routing and a 5% independent sheet check. "
        "Connected-wire and manufacturing failures remain visible; feeds, cooling and duty cycle need design."
    ),
    "biplanar": (
        "Open Y-gradient insert: two 200 × 200 mm planes normal to world x, separated by 100 mm, surround "
        "a 40 mm diameter spherical imaging volume (DSV). Gy = dBz/dy = 20 mT/m, "
        "with the 3 T main field along world z. The designed Bz perturbation is −0.4, 0 and +0.4 mT "
        "at y = −20, 0 and +20 mm. Each support receives a separate connected 2 × 2 mm copper circuit "
        "(4 mm² area, 6 mm minimum bend radius) at the modeled current; "
        "direct 3D inter-group connectors "
        "may leave the planes. The 1 mm homogenized sheet thickness is a separate optimization assumption, "
        "not the physical wire size. A 4% sheet design RMS budget reserves sampling margin for a 5% "
        "independent sheet check. Routing and swept-body checks must be inspected. "
        "Inter-plane feeds, mutual inductance, cooling and duty cycle are not qualified."
    ),
    "independent_turns": (
        "B0 shim demonstration: four separate closed turns on a 200 mm long cylinder of 80 mm radius, "
        "targeting uniform Bz = 100 µT in a spherical ROI of 20 mm radius. "
        "The study uses 1 × 1 mm copper, 0.1 mm insulation and a 3 mm minimum bend radius. "
        "A 4% sheet design RMS budget reserves sampling margin for a 5% independent sheet check. "
        "All four solids and all body-pair clearances are reviewed. "
        "The turns have no feed breaks or terminals; a driven assembly and its mutual inductance remain undesigned."
    ),
    "conical_harmonic": (
        "Z2 shim on a 300 mm long conical support, with bottom/top radii 180/140 mm; "
        "50 mm spherical ROI radius. The absolute world-coordinate target is "
        "Bz = 0.002(2z² − x² − y²) T, giving +10 µT at the ROI poles and −5 µT at its equator. "
        "The study uses 12 contour levels and 1 × 1 mm copper in a 3 T main field. "
        "A 4% sheet design RMS budget reserves sampling margin for a 5% independent sheet check. "
        "The continuous sheet passes in the reference run; discrete-wire field and routing need improvement. "
        "Inspect those failed checks before fabrication. Drive feeds and thermal operation remain undesigned."
    ),
}


def _check_kind(kind: str) -> None:
    if kind not in PRESET_NAMES:
        raise ValueError(f"Unknown example {kind!r}; choose {', '.join(PRESET_NAMES)}.")


def preset_description(kind: str = "cylinder") -> str:
    """Explain the starting assumptions without claiming hardware acceptance."""
    _check_kind(kind)
    return _DESCRIPTIONS[kind]


def preset_study(kind: str = "cylinder") -> dict:
    """A full design study with a distinct, denser sheet validation grid.

    The native spherical design grid uses 17 samples across four ROI radii,
    retaining 257 interior samples. Validation uses 18 cell-centered samples
    across two ROI radii, retaining 3112 distinct samples. The cylinder uses a
    3% solve budget and design_margin=0.6; the other examples use 4% and 0.8.
    Both reserve headroom while keeping the independent sheet budget at 5%.
    """
    _check_kind(kind)
    return {
        "pipeline_stage": "evaluated",
        "validation_resolution": 18,
        "design_margin": 0.6 if kind == "cylinder" else 0.8,
        "fabricate": True,
    }


def preset_config(kind: str = "cylinder") -> CoilGenConfig:
    """Return an explicit compact gradient or independent-turn shim model.

    Resistivity is 18 nΩ m, approximately room-temperature copper. The 1 mm
    sheet thickness belongs to the continuous-sheet power model; swept-wire
    resistance and Joule loss use the separate conductor area: 2 × 2 mm for
    Gy gradient inserts and 1 × 1 mm for the two shim demonstrations.
    No thermal, voltage or duty-cycle qualification is inferred from either.
    """
    _check_kind(kind)
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        # The legacy builder realizes height * nz/(nz+1). Compensate here
        # for a physical 200 mm span without changing old script geometry.
        cylinder_mesh_parameter_list=(0.2 * 17 / 16, 0.08, 32, 16, 1, 0, 0, 0),
        field_shape_function="y",
        target_gradient_strength=0.02,
        target_gradient_normalization="linear_fit",
        target_gradient_axis="y",
        target_field_component="Bz",
        target_field_units="T",
        target_region_radius=0.02,
        target_region_resolution=17,
        design_context="mr_gradient",
        b_0_strength_T=3.0,
        sf_opt_method="constrained_quadratic",
        research_physics={"objective": "power", "b0_T": (0.0, 0.0, 3.0)},
        constrained_quadratic_fallback_policy="strict",
        target_error_tolerance_percent=3.0 if kind == "cylinder" else 4.0,
        conductor_thickness=0.001,
        conductor_cross_section_width=0.002 if kind in ("cylinder", "biplanar") else 0.001,
        conductor_cross_section_height=0.002 if kind in ("cylinder", "biplanar") else 0.001,
        specific_conductivity_conductor=18e-9,
        levels=16,
        min_point_loop_number=4,
        gauss_order=2,
        field_evaluation_current_source="stream_function",
        inter_group_connection_path_method="mesh_geodesic",
        joint_group_routing_max_candidates=3,
        conductor_mounting_mode="surface_outer",
        normal_shift_length=0.002,
        interconnection_cut_width=0.006,
        sweep_bend_mode="fillet",
        sweep_min_bend_radius=0.006 if kind in ("cylinder", "biplanar") else 0.003,
        sweep_insulation_thickness=0.0001,
        sweep_manufacturing_tolerance=0.00005,
        sweep_max_segment_length=0.004,
        skip_postprocessing=False,
        skip_sweep=False,
        contour_only_sweep_mode=False,
        skip_inductance_calculation=False,
        parallel_worker_count=1,
        progress_flag=False,
        plot_flag=False,
        save_stl_flag=False,
    )
    if kind == "biplanar":
        return replace(
            config,
            coil_mesh_file="create bi-planary mesh",
            biplanar_mesh_parameter_list=(0.2, 0.2, 16, 16, 1, 0, 0, 0, 0, 0, 0.1),
            surface_is_cylinder_flag=False,
            inter_group_connection_path_method="direct_3d",
            routing_circuit_scope="per_part",
        )
    if kind == "independent_turns":
        return replace(
            config,
            field_shape_function="1e-4",
            target_gradient_strength=1e-4,
            target_gradient_normalization="none",
            design_context="mr_shim",
            pot_offset_factor=0.25,
            levels=4,
            skip_postprocessing=True,
            contour_only_sweep_mode=True,
        )
    if kind == "conical_harmonic":
        from ..problem import PipelineGeometrySpec
        from .targets import SolidHarmonicTarget

        geometry = PipelineGeometrySpec.conical_frustum(
            length_m=0.30,
            radius_bottom_m=0.18,
            radius_top_m=0.14,
            angular_divisions=32,
            length_divisions=12,
        )
        target = SolidHarmonicTarget(2, 0, 10e-6 / math.sqrt(5 / (4 * math.pi)), 0.05)
        config = replace(config, **geometry.to_config_overrides(), **target.to_config_overrides())
        return replace(
            config,
            # Tapered routing must not use cylinder-only geometric assumptions.
            surface_is_cylinder_flag=False,
            target_region_radius=0.05,
            levels=12,
            symmetry_reduction="off",
            sweep_surface_realization="mesh",
            interconnection_cut_width=0.008,
        )
    return config
