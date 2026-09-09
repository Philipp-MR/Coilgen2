from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import coilgen.optimization_solvers as solvers


def test_supported_names_and_injected_discovery_do_not_import_cvxpy(monkeypatch) -> None:
    importer = Mock(side_effect=AssertionError("CVXPY should remain optional"))
    monkeypatch.setattr(solvers, "import_module", importer)

    assert solvers.supported_conic_solvers() == ("CLARABEL", "ECOS", "SCS")
    rows = solvers.conic_solver_capabilities(installed_solvers=["scs", "OSQP"])

    assert [(row.name, row.installed, row.auto_priority) for row in rows] == [
        ("CLARABEL", False, 1),
        ("ECOS", False, 2),
        ("SCS", True, 3),
    ]
    importer.assert_not_called()


def test_capability_records_are_immutable() -> None:
    rows = solvers.conic_solver_capabilities(installed_solvers=["CLARABEL"])

    assert isinstance(rows, tuple)
    with pytest.raises(FrozenInstanceError):
        rows[0].installed = False


def test_missing_optional_cvxpy_keeps_supported_names_but_no_available_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        solvers, "import_module", Mock(side_effect=ModuleNotFoundError("No module named cvxpy", name="cvxpy"))
    )

    rows = solvers.conic_solver_capabilities()

    assert len(rows) == 3
    assert not any(row.installed for row in rows)
    assert solvers.conic_solver_order() == ()
    with pytest.raises(RuntimeError, match="'CLARABEL' is not installed"):
        solvers.conic_solver_order("clarabel")


def test_on_demand_discovery_uses_cvxpy_backend_inventory(monkeypatch) -> None:
    backend_inventory = Mock(return_value=["SCIPY", "CLARABEL", "SCS"])
    importer = Mock(return_value=SimpleNamespace(installed_solvers=backend_inventory))
    monkeypatch.setattr(solvers, "import_module", importer)

    assert solvers.conic_solver_order() == ("CLARABEL", "SCS")

    importer.assert_called_once_with("cvxpy")
    backend_inventory.assert_called_once_with()


def test_auto_order_is_stable_and_ignores_unintegrated_backends() -> None:
    installed = iter(["scs", "MOSEK", "ECOS", "CLARABEL", "CLARABEL", "OSQP"])

    assert solvers.conic_solver_order(installed_solvers=installed) == ("CLARABEL", "ECOS", "SCS")


def test_explicit_solver_does_not_silently_fall_back() -> None:
    assert solvers.conic_solver_order(" ecos ", installed_solvers=["ECOS", "CLARABEL"]) == ("ECOS",)
    with pytest.raises(RuntimeError, match="'ECOS' is not installed"):
        solvers.conic_solver_order("ECOS", installed_solvers=["CLARABEL", "SCS"])


def test_installed_qp_only_solver_cannot_be_used_for_general_soc_formulation() -> None:
    with pytest.raises(ValueError, match="OSQP does not support second-order cone constraints"):
        solvers.conic_solver_order("OSQP", installed_solvers=["OSQP"])


def test_unknown_solver_is_rejected_before_optional_dependency_discovery(monkeypatch) -> None:
    importer = Mock(side_effect=AssertionError("Do not discover an unsupported backend"))
    monkeypatch.setattr(solvers, "import_module", importer)

    with pytest.raises(ValueError, match="not supported"):
        solvers.conic_solver_order("MOSEK")

    importer.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError("broken native extension"), ImportError("broken native extension")])
def test_broken_discovery_is_not_misreported_as_no_installed_solver(monkeypatch, error) -> None:
    monkeypatch.setattr(solvers, "import_module", Mock(side_effect=error))

    with pytest.raises(type(error), match="broken native extension"):
        solvers.conic_solver_capabilities()
