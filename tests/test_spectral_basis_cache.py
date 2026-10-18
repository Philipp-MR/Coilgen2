from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.models import Mesh
import coilgen.spectral_stream_function as spectral


@pytest.fixture(autouse=True)
def empty_basis_cache():
    spectral._BASIS_CACHE.clear()
    yield
    spectral._BASIS_CACHE.clear()


def _inputs():
    mesh = Mesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]),
        faces=np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
    )
    expansion = np.eye(5)[:, :4]
    config = SimpleNamespace(
        stream_function_spectral_cutoff_length_m=0.0,
        stream_function_spectral_min_modes=1,
        stream_function_spectral_max_modes=3,
        stream_function_spectral_cutoff_policy="hard",
    )
    return mesh, expansion, config


def test_repeated_preparation_reuses_verified_basis_without_operator_or_solver(monkeypatch):
    mesh, expansion, config = _inputs()
    first = spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    assert first.diagnostics["stream_function_spectral_cache_eligible"]
    assert first.diagnostics["stream_function_spectral_eigen_residual_relative"] < 1e-12
    assert first.diagnostics["stream_function_spectral_mass_orthogonality_error"] < 1e-12
    monkeypatch.setattr(
        spectral,
        "cotangent_stiffness_and_lumped_mass",
        lambda _mesh: pytest.fail("Exact repeat must reuse prepared operator/basis"),
    )
    again = spectral.build_spectral_stream_function_basis(mesh, expansion.copy(), config)
    assert again.diagnostics["stream_function_spectral_cache_hit"]
    np.testing.assert_array_equal(again.basis, first.basis)
    first.basis[:] = 0.0
    again.diagnostics["stream_function_spectral_solver"] = "changed"
    third = spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    assert np.linalg.norm(third.basis) > 0.0
    assert third.diagnostics["stream_function_spectral_solver"] != "changed"


@pytest.mark.parametrize("change", ["vertices", "faces", "reducer", "cutoff", "policy", "min_modes", "max_modes"])
def test_changed_operator_mass_reducer_or_settings_miss_cache(change):
    mesh, expansion, config = _inputs()
    spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    if change == "vertices":
        mesh.vertices[0, 0] += 1e-8
    elif change == "faces":
        mesh.faces[:] = mesh.faces[::-1]
    elif change == "reducer":
        expansion[4, 0] = 0.01
    elif change == "cutoff":
        config.stream_function_spectral_cutoff_length_m = 0.1
    elif change == "policy":
        config.stream_function_spectral_cutoff_policy = "minimum_modes"
    elif change == "min_modes":
        config.stream_function_spectral_min_modes = 2
    elif change == "max_modes":
        config.stream_function_spectral_max_modes = 2
    changed = spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    assert not changed.diagnostics["stream_function_spectral_cache_hit"]


def test_cache_respects_byte_budget(monkeypatch):
    monkeypatch.setattr(spectral, "_BASIS_CACHE_MAX_BYTES", 1)
    mesh, expansion, config = _inputs()
    first = spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    second = spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    assert first.diagnostics["stream_function_spectral_cache_eligible"]
    assert not second.diagnostics["stream_function_spectral_cache_hit"]
    assert not spectral._BASIS_CACHE


def test_cache_evicts_oldest_basis_at_entry_budget(monkeypatch):
    monkeypatch.setattr(spectral, "_BASIS_CACHE_MAX_ENTRIES", 1)
    mesh, expansion, config = _inputs()
    spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    config.stream_function_spectral_max_modes = 2
    spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    assert len(spectral._BASIS_CACHE) == 1
    config.stream_function_spectral_max_modes = 3
    result = spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    assert not result.diagnostics["stream_function_spectral_cache_hit"]
    assert len(spectral._BASIS_CACHE) == 1


def test_failed_eigen_residual_check_prevents_reuse(monkeypatch):
    mesh, expansion, config = _inputs()
    original = spectral._lowest_mass_normalised_eigenvectors

    def wrong_eigenvalues(*args):
        values, vectors, solver = original(*args)
        return values + 1.0, vectors, solver

    monkeypatch.setattr(spectral, "_lowest_mass_normalised_eigenvectors", wrong_eigenvalues)
    first = spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    second = spectral.build_spectral_stream_function_basis(mesh, expansion, config)
    assert not first.diagnostics["stream_function_spectral_cache_eligible"]
    assert not second.diagnostics["stream_function_spectral_cache_hit"]
    assert not spectral._BASIS_CACHE
