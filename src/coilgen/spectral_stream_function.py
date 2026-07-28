from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import Mesh


@dataclass(frozen=True)
class SpectralStreamFunctionBasis:
    basis: np.ndarray
    diagnostics: dict[str, Any]


def build_spectral_stream_function_basis(
    mesh: Mesh,
    expansion_matrix: np.ndarray,
    config: Any,
) -> SpectralStreamFunctionBasis:
    """Build a low-frequency Laplace-Beltrami basis for stream-function values.

    The generalized surface eigenproblem is assembled on the physical mesh and
    then reduced through the existing boundary/symmetry expansion E:

        E^T C E v = lambda E^T M E v

    C is the cotangent stiffness matrix and M is the lumped vertex mass matrix.
    The returned basis maps the current reduced optimizer variables to an even
    smaller spectral coefficient vector.
    """

    expansion = np.asarray(expansion_matrix, dtype=float)
    if expansion.ndim != 2:
        raise ValueError("Stream-function spectral expansion matrix must be two-dimensional.")
    base_unknown_count = int(expansion.shape[1])
    if base_unknown_count == 0:
        raise ValueError("Cannot build a spectral stream-function basis for zero unknowns.")

    stiffness, mass = cotangent_stiffness_and_lumped_mass(mesh)
    reduced_stiffness = expansion.T @ (stiffness @ expansion)
    reduced_mass = expansion.T @ (mass[:, None] * expansion)
    reduced_mass = 0.5 * (reduced_mass + reduced_mass.T)
    if not np.all(np.isfinite(reduced_mass)):
        raise ValueError("Cannot build spectral stream-function basis: reduced surface mass contains nonfinite values.")

    cutoff_length = _positive_float(getattr(config, "stream_function_spectral_cutoff_length_m", 0.0))
    min_modes = max(int(getattr(config, "stream_function_spectral_min_modes", 1)), 1)
    max_modes = int(getattr(config, "stream_function_spectral_max_modes", 0))
    if max_modes <= 0:
        max_modes = min(max(base_unknown_count, min_modes), 256)
    max_modes = min(max(max_modes, min_modes), base_unknown_count)

    requested_eigenmodes = max_modes
    eigenvalues, eigenvectors, solver_name = _lowest_mass_normalised_eigenvectors(
        reduced_stiffness,
        reduced_mass,
        requested_eigenmodes,
    )
    if eigenvalues.size == 0:
        raise ValueError("Stream-function spectral basis eigenproblem returned no modes.")

    if cutoff_length > 0.0:
        cutoff_lambda = float((2.0 * np.pi / cutoff_length) ** 2)
        cutoff_count = int(np.count_nonzero(eigenvalues <= cutoff_lambda * (1.0 + 1.0e-10)))
        cutoff_policy = _normalise_cutoff_policy(getattr(config, "stream_function_spectral_cutoff_policy", "hard"))
        if cutoff_policy == "hard":
            if cutoff_count <= 0:
                longest_wavelength = _lambda_to_wavelength(float(eigenvalues[0]))
                raise ValueError(
                    "The hard stream-function spectral cutoff retains no non-gauge modes: "
                    f"cutoff length {cutoff_length:.6g} m, longest available wavelength "
                    f"{longest_wavelength:.6g} m. Relax the cutoff length."
                )
            selected_count = cutoff_count
        else:
            selected_count = max(min_modes, cutoff_count)
    else:
        cutoff_lambda = float("inf")
        cutoff_count = int(eigenvalues.size)
        selected_count = max_modes
        cutoff_policy = _normalise_cutoff_policy(getattr(config, "stream_function_spectral_cutoff_policy", "hard"))
    selected_count = min(max(selected_count, 1), eigenvalues.size, max_modes)

    selected_values = np.asarray(eigenvalues[:selected_count], dtype=float)
    selected_basis = np.asarray(eigenvectors[:, :selected_count], dtype=float)
    selected_basis = _mass_orthonormalize(selected_basis, reduced_mass)
    max_retained_lambda = float(np.max(selected_values)) if selected_values.size else float("nan")
    min_rejected_lambda = float(eigenvalues[selected_count]) if selected_count < eigenvalues.size else float("nan")
    retained_wavelength = _lambda_to_wavelength(max_retained_lambda)
    rejected_wavelength = _lambda_to_wavelength(min_rejected_lambda)

    diagnostics = {
        "stream_function_basis": "cotangent_spectral_lowpass",
        "stream_function_spectral_active": True,
        "stream_function_spectral_solver": solver_name,
        "stream_function_spectral_unknown_count_before": base_unknown_count,
        "stream_function_spectral_unknown_count_after": int(selected_basis.shape[1]),
        "stream_function_spectral_reduction_factor": float(base_unknown_count / max(int(selected_basis.shape[1]), 1)),
        "stream_function_spectral_cutoff_length_m": float(cutoff_length),
        "stream_function_spectral_cutoff_policy": cutoff_policy,
        "stream_function_spectral_min_modes_overrode_cutoff": bool(
            cutoff_length > 0.0 and selected_count > cutoff_count
        ),
        "stream_function_spectral_cutoff_lambda_1_per_m2": cutoff_lambda,
        "stream_function_spectral_requested_eigenmodes": int(requested_eigenmodes),
        "stream_function_spectral_computed_eigenmodes": int(eigenvalues.size),
        "stream_function_spectral_cutoff_mode_count": int(cutoff_count),
        "stream_function_spectral_min_modes": int(min_modes),
        "stream_function_spectral_max_modes": int(max_modes),
        "stream_function_spectral_cutoff_reached": bool(cutoff_length <= 0.0 or cutoff_count < eigenvalues.size),
        "stream_function_spectral_lowest_lambda_1_per_m2": float(eigenvalues[0]),
        "stream_function_spectral_max_retained_lambda_1_per_m2": max_retained_lambda,
        "stream_function_spectral_min_rejected_lambda_1_per_m2": min_rejected_lambda,
        "stream_function_spectral_min_retained_wavelength_m": retained_wavelength,
        "stream_function_spectral_next_rejected_wavelength_m": rejected_wavelength,
    }
    return SpectralStreamFunctionBasis(selected_basis, diagnostics)


def cotangent_stiffness_and_lumped_mass(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_count = int(mesh.n_vertices)
    stiffness = np.zeros((vertex_count, vertex_count), dtype=float)
    mass = np.zeros(vertex_count, dtype=float)
    for face in faces:
        if len(face) != 3:
            continue
        i, j, k = (int(face[0]), int(face[1]), int(face[2]))
        if i < 0 or j < 0 or k < 0 or i >= vertex_count or j >= vertex_count or k >= vertex_count:
            continue
        p_i = vertices[i]
        p_j = vertices[j]
        p_k = vertices[k]
        area2 = float(np.linalg.norm(np.cross(p_j - p_i, p_k - p_i)))
        if not np.isfinite(area2) or area2 <= 0.0:
            continue
        area = 0.5 * area2
        mass[[i, j, k]] += area / 3.0
        _accumulate_cotangent_edge(stiffness, j, k, _cotangent(p_j - p_i, p_k - p_i))
        _accumulate_cotangent_edge(stiffness, k, i, _cotangent(p_k - p_j, p_i - p_j))
        _accumulate_cotangent_edge(stiffness, i, j, _cotangent(p_i - p_k, p_j - p_k))
    stiffness = 0.5 * (stiffness + stiffness.T)
    return stiffness, mass


def _accumulate_cotangent_edge(stiffness: np.ndarray, a: int, b: int, cotangent: float) -> None:
    if not np.isfinite(cotangent):
        return
    weight = 0.5 * float(cotangent)
    stiffness[a, a] += weight
    stiffness[b, b] += weight
    stiffness[a, b] -= weight
    stiffness[b, a] -= weight


def _cotangent(edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    cross_norm = float(np.linalg.norm(np.cross(edge_a, edge_b)))
    if not np.isfinite(cross_norm) or cross_norm <= 0.0:
        return 0.0
    return float(np.dot(edge_a, edge_b) / cross_norm)


def _lowest_mass_normalised_eigenvectors(
    stiffness: np.ndarray,
    mass_matrix: np.ndarray,
    requested_count: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    mass = np.asarray(mass_matrix, dtype=float)
    if mass.ndim == 1:
        mass = np.diag(mass)
    mass = 0.5 * (mass + mass.T)
    try:
        mass_cholesky = np.linalg.cholesky(mass)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Cannot build spectral stream-function basis: reduced surface mass is not positive definite."
        ) from exc
    left_whitened = np.linalg.solve(mass_cholesky, np.asarray(stiffness, dtype=float))
    operator = np.linalg.solve(mass_cholesky, left_whitened.T).T
    operator = 0.5 * (operator + operator.T)
    unknown_count = int(operator.shape[0])
    requested_count = min(max(int(requested_count), 1), unknown_count)

    # ARPACK/eigsh is efficient for a small low-frequency prefix. For near-full
    # low-pass bases it is faster and more stable to solve the dense symmetric
    # problem once.
    if unknown_count <= 2 or requested_count >= max(int(0.4 * unknown_count), 1):
        values, normalised_vectors = np.linalg.eigh(operator)
        solver_name = "dense_eigh"
    else:
        try:
            from scipy.sparse.linalg import eigsh  # type: ignore

            values, normalised_vectors = eigsh(operator, k=requested_count, which="SA")
            solver_name = "scipy_eigsh"
        except Exception:
            values, normalised_vectors = np.linalg.eigh(operator)
            solver_name = "dense_eigh_fallback"

    order = np.argsort(values)
    values = np.asarray(values[order], dtype=float)
    normalised_vectors = np.asarray(normalised_vectors[:, order], dtype=float)
    count = min(requested_count, values.size)
    values = values[:count]
    normalised_vectors = normalised_vectors[:, :count]
    eigenvalue_scale = max(float(np.max(np.abs(values))) if values.size else 1.0, 1.0)
    eigenvalue_tolerance = 1.0e-10 * eigenvalue_scale
    if np.any(values < -eigenvalue_tolerance):
        raise ValueError("Cotangent spectral operator has significantly negative eigenvalues.")
    values = np.where(np.abs(values) <= eigenvalue_tolerance, 0.0, values)
    vectors = np.linalg.solve(mass_cholesky.T, normalised_vectors)
    vectors = _mass_orthonormalize(vectors, mass)
    return values, vectors, solver_name


def _mass_orthonormalize(vectors: np.ndarray, mass_matrix: np.ndarray) -> np.ndarray:
    mass = np.asarray(mass_matrix, dtype=float)
    if mass.ndim == 1:
        mass = np.diag(mass)
    mass_cholesky = np.linalg.cholesky(0.5 * (mass + mass.T))
    weighted = mass_cholesky.T @ np.asarray(vectors, dtype=float)
    q, _r = np.linalg.qr(weighted, mode="reduced")
    return np.linalg.solve(mass_cholesky.T, q)


def _lambda_to_wavelength(lambda_value: float) -> float:
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        return float("inf")
    return float(2.0 * np.pi / np.sqrt(lambda_value))


def _positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number) or number <= 0.0:
        return 0.0
    return number


def _normalise_cutoff_policy(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"minimum_modes", "min_modes", "legacy"}:
        return "minimum_modes"
    if key in {"hard", "strict", "physical"}:
        return "hard"
    raise ValueError("stream_function_spectral_cutoff_policy must be 'hard' or 'minimum_modes'.")
