from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .diagnostics import (
    DiagnosticCache,
    ManufacturingDiagnostics,
    RegularizationDiagnostics,
    RouteDiagnostics,
    TopologyDiagnostics,
)


def as_float_array(value: Any, shape_last: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape_last is not None and (array.ndim != 2 or array.shape[1] != shape_last):
        raise ValueError(f"Expected a 2D float array with last dimension {shape_last}, got {array.shape}.")
    return array


def as_int_array(value: Any, shape_last: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if shape_last is not None and (array.ndim != 2 or array.shape[1] != shape_last):
        raise ValueError(f"Expected a 2D integer array with last dimension {shape_last}, got {array.shape}.")
    return array


@dataclass
class Mesh:
    """Triangular mesh using Python-native conventions.

    vertices are always ``(n_vertices, 3)``.
    faces are always ``(n_faces, 3)`` and 0-based.
    """

    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray | None = None
    face_normals: np.ndarray | None = None
    boundaries: list[np.ndarray] = field(default_factory=list)
    unique_vert_inds: np.ndarray | None = None
    stream_function: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = as_float_array(self.vertices, 3)
        self.faces = as_int_array(self.faces, 3)
        if self.vertex_normals is not None:
            self.vertex_normals = as_float_array(self.vertex_normals, 3)
        if self.face_normals is not None:
            self.face_normals = as_float_array(self.face_normals, 3)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def copy(self) -> "Mesh":
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            vertex_normals=None if self.vertex_normals is None else self.vertex_normals.copy(),
            face_normals=None if self.face_normals is None else self.face_normals.copy(),
            boundaries=[boundary.copy() for boundary in self.boundaries],
            unique_vert_inds=None if self.unique_vert_inds is None else self.unique_vert_inds.copy(),
            stream_function=None if self.stream_function is None else self.stream_function.copy(),
            metadata=dict(self.metadata),
        )


@dataclass
class SweptBody:
    """One independently closed triangulated conductor body.

    ``contour_index`` is one-based for stable diagnostics and export names.  A
    value of ``None`` denotes the legacy single connected routed-wire body.
    """

    mesh: Mesh
    contour_index: int | None = None
    turn_multiplicity: int = 1
    diagnostics: ManufacturingDiagnostics = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.contour_index is not None and int(self.contour_index) < 1:
            raise ValueError("contour_index must be a positive one-based index or None.")
        if isinstance(self.turn_multiplicity, bool) or int(self.turn_multiplicity) < 1:
            raise ValueError("turn_multiplicity must be a positive integer.")
        self.contour_index = None if self.contour_index is None else int(self.contour_index)
        self.turn_multiplicity = int(self.turn_multiplicity)
        self.diagnostics = dict(self.diagnostics)


@dataclass
class BasisElement:
    triangles: np.ndarray
    one_ring: np.ndarray
    area: np.ndarray
    face_normal: np.ndarray
    triangle_points_abc: np.ndarray
    current: np.ndarray


@dataclass
class ContourLine:
    v: np.ndarray
    potential: float
    current_orientation: float = 1.0
    surface_faces: np.ndarray | None = None
    surface_barycentric: np.ndarray | None = None
    surface_edge_vertices: np.ndarray | None = None
    surface_edge_t: np.ndarray | None = None
    segment_faces: np.ndarray | None = None
    segment_barycentric: np.ndarray | None = None
    turn_multiplicity: int = 1

    def __post_init__(self) -> None:
        self.v = as_float_array(self.v, 3)
        if isinstance(self.turn_multiplicity, bool) or int(self.turn_multiplicity) != self.turn_multiplicity:
            raise ValueError("turn_multiplicity must be a positive integer.")
        self.turn_multiplicity = int(self.turn_multiplicity)
        if self.turn_multiplicity < 1:
            raise ValueError("turn_multiplicity must be a positive integer.")
        if self.surface_faces is not None:
            self.surface_faces = np.asarray(self.surface_faces, dtype=np.int64)
            if self.surface_faces.ndim != 1 or len(self.surface_faces) != len(self.v):
                raise ValueError("surface_faces must be a 1D array matching contour point count.")
        if self.surface_barycentric is not None:
            self.surface_barycentric = as_float_array(self.surface_barycentric, 3)
            if len(self.surface_barycentric) != len(self.v):
                raise ValueError("surface_barycentric must match contour point count.")
        if self.surface_edge_vertices is not None:
            self.surface_edge_vertices = as_int_array(self.surface_edge_vertices, 2)
            if len(self.surface_edge_vertices) != len(self.v):
                raise ValueError("surface_edge_vertices must match contour point count.")
        if self.surface_edge_t is not None:
            self.surface_edge_t = np.asarray(self.surface_edge_t, dtype=float)
            if self.surface_edge_t.ndim != 1 or len(self.surface_edge_t) != len(self.v):
                raise ValueError("surface_edge_t must match contour point count.")
        if self.segment_faces is not None:
            self.segment_faces = np.asarray(self.segment_faces, dtype=np.int64)
            if self.segment_faces.ndim != 1 or len(self.segment_faces) != max(len(self.v) - 1, 0):
                raise ValueError("segment_faces must match contour segment count.")
        if self.segment_barycentric is not None:
            self.segment_barycentric = np.asarray(self.segment_barycentric, dtype=float)
            expected = (max(len(self.v) - 1, 0), 2, 3)
            if self.segment_barycentric.shape != expected:
                raise ValueError(f"segment_barycentric must have shape {expected}.")

    @property
    def length(self) -> float:
        if len(self.v) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.v, axis=0), axis=1).sum())

    @property
    def effective_turn_length(self) -> float:
        """Conductor length represented by this nominal contour."""

        return float(self.turn_multiplicity * self.length)


@dataclass(frozen=True)
class ContourMergeRecord:
    """Trace one attempted replacement of adjacent contours by a weighted contour."""

    representative_contour_index: int
    source_contour_indices: tuple[int, ...]
    source_potentials: tuple[float, ...]
    selected_potential: float
    turn_multiplicity: int
    positioning_method: str
    pair_field_rms_error_T: float
    pair_field_peak_error_T: float
    accepted: bool
    rejection_reason: str = ""
    pass_index: int = 1
    source_turn_multiplicities: tuple[int, ...] = field(default_factory=tuple)
    stage: str = "pre_pruning"


@dataclass
class ContourSparsificationResult:
    """Nominal multi-turn layout provenance and optional physical realization."""

    source_contours: list[ContourLine] = field(default_factory=list)
    merge_records: list[ContourMergeRecord] = field(default_factory=list)
    final_source_contour_indices: list[tuple[int, ...]] = field(default_factory=list)
    realized_turn_contours: list[ContourLine] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WireSegmentOwner:
    """Provenance for one interval of a routed wire centerline."""

    kind: str
    group_index: int = -1
    loop_index: int = -1
    connector_index: int = -1
    group1_index: int = -1
    group2_index: int = -1
    connector_role: str = ""

    def __post_init__(self) -> None:
        self.kind = str(self.kind)
        self.group_index = int(self.group_index)
        self.loop_index = int(self.loop_index)
        self.connector_index = int(self.connector_index)
        self.group1_index = int(self.group1_index)
        self.group2_index = int(self.group2_index)
        self.connector_role = str(self.connector_role)


class WirePathGeometry(str, Enum):
    """Geometric contract for one routed wire-centerline interval."""

    SURFACE = "surface"
    OFFSET_SURFACE = "offset_surface"
    FREE_3D = "free_3d"


@dataclass
class WirePath:
    v: np.ndarray
    segment_owners: list[WireSegmentOwner] = field(default_factory=list)
    segment_geometry: list[WirePathGeometry | str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.v = as_float_array(self.v, 3)
        self.segment_owners = list(self.segment_owners)
        expected = max(len(self.v) - 1, 0)
        if self.segment_owners and len(self.segment_owners) != expected:
            raise ValueError(
                "segment_owners must contain one entry per wire-path segment "
                f"({len(self.segment_owners)} != {expected})"
            )
        if self.segment_geometry:
            if len(self.segment_geometry) != expected:
                raise ValueError(
                    "segment_geometry must contain one entry per wire-path segment "
                    f"({len(self.segment_geometry)} != {expected})"
                )
            self.segment_geometry = [WirePathGeometry(value) for value in self.segment_geometry]
        else:
            self.segment_geometry = [WirePathGeometry.SURFACE] * expected


@dataclass
class LoopGroup:
    loops: list[ContourLine]
    loop_indices: list[int] = field(default_factory=list)
    opened_loops: list[ContourLine] = field(default_factory=list)
    opening_cuts_v: list[np.ndarray] = field(default_factory=list)
    within_group_cut_side: str = ""
    within_group_cut_score: float = 0.0
    within_group_cut_scores: dict[str, float] = field(default_factory=dict)
    within_group_cut_pair_indices: tuple[int, ...] = field(default_factory=tuple)
    within_group_cut_metrics: dict[str, float] = field(default_factory=dict)
    outer_loop_indices: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class ConnectedGroup:
    v: np.ndarray
    return_path_v: np.ndarray
    spiral_in_v: np.ndarray
    spiral_out_v: np.ndarray
    outer_loop_indices: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.v = as_float_array(self.v, 3)
        self.return_path_v = as_float_array(self.return_path_v, 3)
        self.spiral_in_v = as_float_array(self.spiral_in_v, 3)
        self.spiral_out_v = as_float_array(self.spiral_out_v, 3)
        self.outer_loop_indices = tuple(dict.fromkeys(int(index) for index in self.outer_loop_indices))


@dataclass
class InterGroupCut:
    cut1_v: np.ndarray
    cut2_v: np.ndarray
    connection_path_v: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    closing_connection_path_v: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    group1_index: int = -1
    group2_index: int = -1
    selected_point1_v: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    selected_point2_v: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    direct_distance: float = 0.0
    geodesic_distance: float = 0.0
    connector_distance: float = 0.0
    connection_method: str = "direct_3d"
    used_direct_fallback: bool = False
    used_clearance_fallback: bool = False
    is_return_connection: bool = False
    normal_shift_paths_v: list[np.ndarray] = field(default_factory=list)
    group1_variant_name: str = ""
    group2_variant_name: str = ""
    loop1_index: int = -1
    loop2_index: int = -1

    def __post_init__(self) -> None:
        self.cut1_v = as_float_array(self.cut1_v, 3)
        self.cut2_v = as_float_array(self.cut2_v, 3)
        self.connection_path_v = as_float_array(self.connection_path_v, 3)
        self.closing_connection_path_v = as_float_array(self.closing_connection_path_v, 3)
        self.selected_point1_v = as_float_array(self.selected_point1_v, 3)
        self.selected_point2_v = as_float_array(self.selected_point2_v, 3)
        self.group1_index = int(self.group1_index)
        self.group2_index = int(self.group2_index)
        self.loop1_index = int(self.loop1_index)
        self.loop2_index = int(self.loop2_index)
        self.direct_distance = float(self.direct_distance)
        self.geodesic_distance = float(self.geodesic_distance)
        self.connector_distance = float(self.connector_distance)
        self.connection_method = str(self.connection_method)
        self.used_direct_fallback = bool(self.used_direct_fallback)
        self.used_clearance_fallback = bool(self.used_clearance_fallback)
        self.is_return_connection = bool(self.is_return_connection)
        self.normal_shift_paths_v = [as_float_array(path, 3) for path in self.normal_shift_paths_v]
        self.group1_variant_name = str(self.group1_variant_name)
        self.group2_variant_name = str(self.group2_variant_name)


@dataclass
class PcbWirePart:
    centerline: np.ndarray
    track_shape: np.ndarray

    def __post_init__(self) -> None:
        self.centerline = as_float_array(self.centerline, 2)
        self.track_shape = as_float_array(self.track_shape, 2)


@dataclass
class PcbLayout:
    upper_layer: list[PcbWirePart] = field(default_factory=list)
    lower_layer: list[PcbWirePart] = field(default_factory=list)
    boundary: np.ndarray | None = None
    track_width: float = 0.0
    cylinder_radius: float = 0.0

    def __post_init__(self) -> None:
        if self.boundary is not None:
            self.boundary = as_float_array(self.boundary, 2)


@dataclass
class TargetField:
    coords: np.ndarray
    b: np.ndarray
    weights: np.ndarray
    suppressed: np.ndarray
    target_field_group_inds: np.ndarray
    target_gradient_dbdxyz: np.ndarray | None = None
    suppression_normals: np.ndarray | None = None
    suppression_area_weights: np.ndarray | None = None
    target_diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coords = as_float_array(self.coords, 3)
        self.b = as_float_array(self.b, 3)
        self.weights = np.asarray(self.weights, dtype=float)
        self.suppressed = np.asarray(self.suppressed, dtype=bool)
        self.target_field_group_inds = np.asarray(self.target_field_group_inds, dtype=np.int64)
        if self.suppression_normals is not None:
            self.suppression_normals = as_float_array(self.suppression_normals, 3)
            if len(self.suppression_normals) != len(self.coords):
                raise ValueError("suppression_normals must match target-field point count.")
        if self.suppression_area_weights is not None:
            self.suppression_area_weights = np.asarray(self.suppression_area_weights, dtype=float).reshape(-1)
            if len(self.suppression_area_weights) != len(self.coords):
                raise ValueError("suppression_area_weights must match target-field point count.")
        self.target_diagnostics = dict(self.target_diagnostics)


@dataclass
class CoilPart:
    mesh: Mesh
    one_ring_list: list[np.ndarray] = field(default_factory=list)
    node_triangles: list[np.ndarray] = field(default_factory=list)
    basis_elements: list[BasisElement] = field(default_factory=list)
    current_density_mat: np.ndarray | None = None
    sensitivity_matrix: np.ndarray | None = None
    resistance_matrix: np.ndarray | None = None
    stream_function: np.ndarray | None = None
    current_density: np.ndarray | None = None
    potential_level_list: np.ndarray | None = None
    contour_step: float | None = None
    contour_level_diagnostics: dict[str, object] = field(default_factory=dict)
    contour_lines: list[ContourLine] = field(default_factory=list)
    contour_sparsification: ContourSparsificationResult | None = None
    groups: list[LoopGroup] = field(default_factory=list)
    loop_groups: list[list[int]] = field(default_factory=list)
    loop_in_loop_matrix: np.ndarray | None = None
    topology_diagnostics: TopologyDiagnostics = field(default_factory=dict)
    group_levels: list[list[int]] = field(default_factory=list)
    level_positions: list[list[int]] = field(default_factory=list)
    group_centers_v: np.ndarray | None = None
    connected_groups: list[ConnectedGroup] = field(default_factory=list)
    opening_cuts_among_groups: list[InterGroupCut] = field(default_factory=list)
    wire_path: WirePath | None = None
    route_diagnostics: RouteDiagnostics = field(default_factory=dict)
    swept_body: Mesh | None = None
    swept_bodies: list[SweptBody] = field(default_factory=list)
    manufacturing_diagnostics: ManufacturingDiagnostics = field(default_factory=dict)
    pcb_layout: PcbLayout | None = None
    shift_array: np.ndarray | None = None
    points_to_shift: np.ndarray | None = None
    wire_self_contacts: list[tuple[int, int, float]] = field(default_factory=list)
    coil_resistance: float = 0.0
    coil_inductance: float = 0.0
    coil_length: float = 0.0
    coil_cross_section: float = 0.0
    field_by_loops: np.ndarray | None = None
    field_by_layout: np.ndarray | None = None
    loop_significance: np.ndarray | None = None
    combined_loop_length: float = 0.0
    pcb_track_width: float = 0.0
    minimal_contour_distance: float = 0.0
    unconnected_contour_min_wire_distance: float = 0.0
    unconnected_contour_wire_distance_diagnostics: dict[str, float | int | bool] = field(default_factory=dict)
    contour_distance_matrix: np.ndarray | None = None

    @property
    def nominal_contour_count(self) -> int:
        return int(len(self.contour_lines))

    @property
    def physical_turn_count(self) -> int:
        return int(sum(contour.turn_multiplicity for contour in self.contour_lines))

    @property
    def nominal_contour_length(self) -> float:
        return float(sum(contour.length for contour in self.contour_lines))

    @property
    def effective_turn_length(self) -> float:
        return float(sum(contour.effective_turn_length for contour in self.contour_lines))


@dataclass
class CoilGenResult:
    coil_parts: list[CoilPart]
    num_levels: int
    potential_step: float
    primary_surface: int
    target_field: TargetField
    field_by_layout: np.ndarray
    field_by_unconnected_loops: np.ndarray
    field_layout_per1Amp: np.ndarray
    field_loops_per1Amp: np.ndarray
    needed_current_layout: float
    b_field_opt_sf: np.ndarray
    b_field_opt_sf_1A: np.ndarray
    regularization_factor: float
    regularization_lambda: float
    regularization_diagnostics: RegularizationDiagnostics
    target_field_1A: TargetField
    error_vals: dict[str, float]
    input_data: Any
    layout_gradient: Any | None
    combined_mesh: Mesh
    inductance_matrix: np.ndarray | None = None
    resistance: np.ndarray | None = None
    coil_lengths: np.ndarray | None = None
    coil_cross_sections: np.ndarray | None = None
    timing_summary: list[dict[str, float | str]] = field(default_factory=list)
    diagnostic_cache: DiagnosticCache = field(default_factory=dict)
