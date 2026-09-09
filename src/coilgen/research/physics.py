"""Physical surface-current operators in SI units.

Magnetic energy is W = mu0/(8*pi) integral integral K(r).K(r')/|r-r'|
dS dS'. The source-triangle integral is analytic, including self terms;
Gauss-Legendre quadrature integrates the remaining target triangle. No wire
radius, diagonal softening, or smoothing surrogate is used. See the surface
current formulation in https://arxiv.org/abs/2005.10060.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.polynomial.legendre import leggauss

MU0_OVER_4PI = 1.0e-7


@dataclass(frozen=True, slots=True)
class EnergyQuadrature:
    initial_order: int = 4
    maximum_order: int = 32
    relative_tolerance: float = 2.0e-3
    psd_relative_tolerance: float = 1.0e-10
    maximum_faces: int = 768

    def __post_init__(self) -> None:
        for name in ("initial_order", "maximum_order", "maximum_faces"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"Energy {name} must be an integer.")
        if self.initial_order < 2 or self.maximum_order < 2 * self.initial_order:
            raise ValueError("Energy quadrature needs initial_order >= 2 and maximum_order >= 2*initial_order.")
        if not np.isfinite(self.relative_tolerance) or not 0 < self.relative_tolerance < 1:
            raise ValueError("Energy relative_tolerance must be finite and between zero and one.")
        if not np.isfinite(self.psd_relative_tolerance) or not 0 <= self.psd_relative_tolerance <= 1e-6:
            raise ValueError("Energy psd_relative_tolerance must be between zero and 1e-6.")
        if self.maximum_faces < 1:
            raise ValueError("Energy maximum_faces must be positive.")

    def preflight(self, face_count: int) -> dict[str, int | str]:
        """Check the explicit dense-assembly budget before building operators.

        The face cap bounds work, not a promised runtime or peak-memory limit.
        Raising it is explicit; energy is never disabled to accommodate a mesh.
        """
        if isinstance(face_count, bool) or not isinstance(face_count, (int, np.integer)) or face_count < 1:
            raise ValueError("Energy face_count must be a positive integer.")
        if face_count > self.maximum_faces:
            raise ValueError(
                f"Energy assembly has {face_count} faces, exceeding energy.maximum_faces={self.maximum_faces}. "
                "Use a coarser mesh or explicitly raise this O(F^2 q^2) quadrature work budget after benchmarking."
            )
        return {
            "assembly": "dense_analytic_triangle_potential",
            "face_count": int(face_count),
            "maximum_faces": self.maximum_faces,
            "maximum_quadrature_order": self.maximum_order,
            "single_kernel_bytes": int(8 * face_count**2),
            "maximum_source_target_quadrature_pairs": int(face_count**2 * self.maximum_order**2),
        }


@dataclass(frozen=True, slots=True)
class SurfaceEnergy:
    """W = x.T @ matrix_J_per_A2 @ x; L(x,I) = 2 W / I**2.

    The error is an observed operator-norm quadrature convergence estimate,
    not a rigorous error bound or a mesh-discretization error estimate.
    """

    matrix_J_per_A2: np.ndarray
    quadrature_order: int
    relative_change: float
    kernel_relative_change: float
    minimum_eigenvalue_J_per_A2: float
    roundoff_psd_correction_J_per_A2: float

    def energy_J(self, coefficients_A: np.ndarray) -> float:
        x = np.asarray(coefficients_A, dtype=float)
        return float(x @ self.matrix_J_per_A2 @ x)

    def inductance_H(self, coefficients_A: np.ndarray, wire_current_A: float) -> float:
        if not np.isfinite(wire_current_A) or wire_current_A == 0:
            raise ValueError("Inductance requires an explicit nonzero physical wire current in A.")
        return 2.0 * self.energy_J(coefficients_A) / wire_current_A**2


@dataclass(frozen=True, slots=True)
class SurfaceOperators:
    triangles_m: np.ndarray
    areas_m2: np.ndarray
    centroids_m: np.ndarray
    current_density_per_A_m: np.ndarray  # (face, Cartesian component, reduced unknown)
    part_face_counts: tuple[int, ...]


def validate_stream_function_topology(vertices: np.ndarray, faces: np.ndarray) -> tuple[dict[str, int], ...]:
    """Reject nonmanifold/nonorientable meshes and unresolved harmonic modes.

    A separate constant on each boundary loop is retained by the existing
    reducer. A positive-genus component also needs harmonic circulation modes,
    which this nodal research formulation deliberately does not invent.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Research physics requires finite (vertex,3) coordinates in metres.")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Research physics requires a nonempty triangular surface.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Research surface contains out-of-range face indices.")
    parent = np.arange(len(vertices))

    def root(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = int(parent[v])
        return v

    edges: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        if len(set(face)) != 3:
            raise ValueError("Research surface contains a degenerate face.")
        for a, b in zip(face, np.roll(face, -1)):
            parent[root(int(b))] = root(int(a))
            edge = tuple(sorted((int(a), int(b))))
            edges.setdefault(edge, []).append(1 if a < b else -1)
    if any(len(signs) > 2 for signs in edges.values()):
        raise ValueError("Research surface has nonmanifold edges.")
    if any(len(signs) == 2 and sum(signs) != 0 for signs in edges.values()):
        raise ValueError("Research surface has inconsistent face orientation; orient it before solving.")
    used = np.unique(faces)
    if len(used) != len(vertices):
        raise ValueError("Research surface contains isolated vertices.")
    component_ids = np.array([root(int(v)) for v in range(len(vertices))])
    reports = []
    for component in np.unique(component_ids):
        nv = int(np.count_nonzero(component_ids == component))
        ne = sum(component_ids[a] == component for a, _ in edges)
        nf = int(np.count_nonzero(component_ids[faces[:, 0]] == component))
        boundary: dict[int, set[int]] = {}
        for (a, b), signs in edges.items():
            if component_ids[a] == component and len(signs) == 1:
                boundary.setdefault(a, set()).add(b)
                boundary.setdefault(b, set()).add(a)
        if any(len(neighbors) != 2 for neighbors in boundary.values()):
            raise ValueError("Research surface has a nonmanifold/open boundary chain.")
        unseen = set(boundary)
        loops = 0
        while unseen:
            loops += 1
            stack = [unseen.pop()]
            while stack:
                for neighbor in boundary[stack.pop()]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
        twice_genus = 2 - loops - (nv - ne + nf)
        if twice_genus < 0 or twice_genus % 2:
            raise ValueError("Research surface topology is not an orientable two-manifold.")
        if twice_genus:
            raise ValueError(
                f"Positive-genus surface (genus={twice_genus // 2}) requires harmonic circulation modes; "
                "the current nodal stream-function basis is incomplete."
            )
        reports.append({"vertices": nv, "edges": int(ne), "faces": nf, "boundary_loops": loops, "genus": 0})
    return tuple(reports)


def surface_operators(problem: Any) -> SurfaceOperators:
    mesh = problem.combined_mesh
    validate_stream_function_topology(mesh.vertices, mesh.faces)
    triangles = np.asarray(mesh.vertices[mesh.faces], dtype=float)
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if np.any(areas <= 0) or not np.all(np.isfinite(areas)):
        raise ValueError("Research surface contains zero-area or invalid triangles.")
    from ..optimization import _contract_surface_current_density, _reduced_surface_current_density_operators

    peak, count, reason = _contract_surface_current_density(problem, np.zeros(mesh.n_vertices))
    if not np.isfinite(peak) or count != len(triangles):
        raise ValueError("Research physics requires complete face-current operators. " + reason)
    supplied = np.concatenate(_reduced_surface_current_density_operators(problem), axis=0)
    local = (np.roll(triangles, 1, axis=1) - np.roll(triangles, -1, axis=1)) / (2 * areas[:, None, None])
    currents = np.einsum("fva,fvr->far", local, problem.reducer.expansion_matrix[mesh.faces])
    scale = max(float(np.max(np.abs(currents))), 1e-300)
    if supplied.shape != currents.shape or np.max(np.abs(supplied - currents)) > 1e-9 * scale:
        raise ValueError("Supplied current operators disagree with the triangle geometry and stream-function basis.")
    return SurfaceOperators(triangles, areas, np.mean(triangles, axis=1), currents, tuple(map(len, problem.part_faces)))


def force_torque_operators(
    surface: SurfaceOperators, b0_T: np.ndarray, origin_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return total F/x [N/A] and torque/x [N m/A] in a prescribed fixed B0.

    Uses integral K x B0 dS and integral (r-origin) x (K x B0) dS.
    B0 is a uniform (3,) vector or prescribed (face,3) piecewise-constant
    samples. The integrands are then constant/linear on each flat face, so
    centroid integration is exact for this background discretization. Self
    forces and structural stresses are not modeled.
    """
    b0 = np.asarray(b0_T, dtype=float)
    origin = np.asarray(origin_m, dtype=float)
    if (
        b0.shape not in {(3,), (len(surface.areas_m2), 3)}
        or origin.shape != (3,)
        or not np.all(np.isfinite(b0))
        or not np.all(np.isfinite(origin))
    ):
        raise ValueError("B0 [T] must be a finite (3,) or (face,3) array and torque origin [m] a finite (3,) vector.")
    k = np.moveaxis(surface.current_density_per_A_m, 1, -1)
    force_faces = np.cross(k, b0 if b0.ndim == 1 else b0[:, None, :]) * surface.areas_m2[:, None, None]
    torque_faces = np.cross(surface.centroids_m[:, None, :] - origin, force_faces)
    force, torque = np.sum(force_faces, axis=0), np.sum(torque_faces, axis=0)
    # Remove only summation roundoff relative to the summed absolute terms.
    # Otherwise an exact zero-force equality can constrain numerical noise.
    for value, terms in ((force, force_faces), (torque, torque_faces)):
        error_scale = 64 * np.finfo(float).eps * np.sum(np.abs(terms), axis=0)
        value[np.abs(value) <= error_scale] = 0.0
    return force.T, torque.T


def triangle_potential(points_m: np.ndarray, triangle_m: np.ndarray) -> np.ndarray:
    """Analytic integral over a triangle of 1/|r-r'| dS' [m].

    Edge logarithms and solid angle evaluate the singular self integral
    without a softening radius. Points are normally strictly interior outer
    quadrature nodes; exact edge points are handled by the zero-distance limit.
    """
    points = np.asarray(points_m, dtype=float)
    tri = np.asarray(triangle_m, dtype=float)
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Triangle potential requires a nondegenerate triangle.")
    normal /= norm
    relative = tri[None, :, :] - points[:, None, :]
    lengths = np.linalg.norm(relative, axis=2)
    potential = np.zeros(len(points))
    for i in range(3):
        j = (i + 1) % 3
        edge = tri[j] - tri[i]
        edge_length = np.linalg.norm(edge)
        outward = np.cross(edge / edge_length, normal)
        distance = relative[:, i] @ outward
        denominator = lengths[:, i] + lengths[:, j] - edge_length
        safe = np.maximum(denominator, np.finfo(float).eps * edge_length)
        potential += distance * np.log1p(2.0 * edge_length / safe)
    a, b, c = relative[:, 0], relative[:, 1], relative[:, 2]
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        np.prod(lengths, axis=1)
        + np.einsum("ij,ij->i", a, b) * lengths[:, 2]
        + np.einsum("ij,ij->i", b, c) * lengths[:, 0]
        + np.einsum("ij,ij->i", c, a) * lengths[:, 1]
    )
    solid_angle = np.abs(2.0 * np.arctan2(numerator, denominator))
    height = np.abs(relative[:, 0] @ normal)
    potential -= height * solid_angle
    if not np.all(np.isfinite(potential)) or np.min(potential) < -1e-10 * np.sqrt(norm):
        raise ValueError("Triangle-potential evaluation lost numerical accuracy; check coordinate scales.")
    return potential


def triangle_pair_kernel(
    triangles_m: np.ndarray, order: int, *, progress: Callable[[float, str], None] | None = None
) -> np.ndarray:
    """Return symmetric integral integral 1/r dS dS' [m^3], including diagonal."""
    triangles = np.asarray(triangles_m, dtype=float)
    nodes, weights = leggauss(order)
    u, v = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2, indexing="ij")
    w = np.outer(weights / 2, weights / 2) * (1 - u)
    barycentric = np.stack(((1 - u) * (1 - v), u, (1 - u) * v), axis=-1).reshape(-1, 3)
    points = np.einsum("qi,fij->fqj", barycentric, triangles)
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    if np.any(double_areas <= 0):
        raise ValueError("Energy quadrature requires nondegenerate triangles.")
    kernel = np.empty((len(triangles), len(triangles)))
    # One source at a time limits memory to O(face_count * order^2).
    for source, triangle in enumerate(triangles):
        if progress is not None and source % 8 == 0:
            progress(
                100 * source / len(triangles), f"energy quadrature order {order}, source {source + 1}/{len(triangles)}"
            )
        values = triangle_potential(points.reshape(-1, 3), triangle).reshape(len(triangles), -1)
        kernel[:, source] = (values @ w.ravel()) * double_areas
    return (kernel + kernel.T) / 2


def surface_magnetic_energy(
    surface: SurfaceOperators,
    settings: EnergyQuadrature = EnergyQuadrature(),
    *,
    progress: Callable[[float, str], None] | None = None,
) -> SurfaceEnergy:
    """Converge the physical energy matrix, rejecting unresolved quadrature/PSD errors."""
    settings.preflight(len(surface.triangles_m))
    current = surface.current_density_per_A_m
    previous_kernel = previous_matrix = None
    order = settings.initial_order
    while order <= settings.maximum_order:
        kernel = triangle_pair_kernel(surface.triangles_m, order, progress=progress)
        matrix = 0.5 * MU0_OVER_4PI * sum(current[:, axis].T @ kernel @ current[:, axis] for axis in range(3))
        matrix = (matrix + matrix.T) / 2
        if previous_matrix is not None:
            relative = float(np.linalg.norm(matrix - previous_matrix, 2) / max(np.linalg.norm(matrix, 2), 1e-300))
            kernel_relative = float(
                np.linalg.norm(kernel - previous_kernel, "fro") / max(np.linalg.norm(kernel, "fro"), 1e-300)
            )
            if max(relative, kernel_relative) <= settings.relative_tolerance:
                if progress is not None:
                    progress(
                        100.0,
                        f"energy converged at order {order}: matrix change {relative:.3g}, kernel change {kernel_relative:.3g}",
                    )
                eigenvalues, vectors = np.linalg.eigh(matrix)
                scale = max(float(np.max(np.abs(eigenvalues))), 1e-300)
                minimum = float(eigenvalues[0])
                if minimum < -settings.psd_relative_tolerance * scale:
                    raise ValueError(
                        f"Magnetic-energy matrix is indefinite (minimum eigenvalue {minimum:g}); refine quadrature."
                    )
                correction = max(-minimum, 0.0)
                if correction:
                    matrix = (vectors * np.maximum(eigenvalues, 0.0)) @ vectors.T
                return SurfaceEnergy(matrix, order, relative, kernel_relative, minimum, correction)
        previous_kernel, previous_matrix = kernel, matrix
        order *= 2
    raise ValueError(
        f"Magnetic-energy quadrature did not converge through order {settings.maximum_order}; "
        f"matrix change={relative:.3g}, kernel change={kernel_relative:.3g}, required={settings.relative_tolerance:.3g}."
    )
