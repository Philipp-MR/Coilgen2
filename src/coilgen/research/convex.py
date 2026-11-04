"""Physical stream function optimization with independently checked SI bounds.

The public specification is JSON-compatible via ``to_dict``/``from_dict``.
Custom field matrices and scenario matrices act on *full nodal stream
functions in A*, before the existing boundary/gauge/symmetry expansion.
Only a fixed finite list of scenarios is certified; this is not a guarantee
for every displacement in a continuous uncertainty region.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping

import numpy as np

from .physics import EnergyQuadrature, force_torque_operators, surface_magnetic_energy, surface_operators


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class LinearFieldConstraint:
    """Bound a physical affine field operator in the stated unit.

    ``norm`` is rms, l2, linf, equality, or upper. Equality imposes Ax=target.
    Upper imposes Ax-target <= tolerance componentwise; use -A for lower
    bounds. All other norms bound Ax-target by tolerance. With group_size=3,
    consecutive Cartesian rows form vectors: linf bounds the maximum vector
    norm and rms bounds RMS vector magnitude. Matrix columns correspond to
    the full mesh vertices. Zero tolerance is permitted.
    """

    name: str
    matrix: Any
    target: Any
    tolerance: float = 0.0
    norm: str = "linf"
    unit: str = "T"
    group_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class FieldScenario:
    name: str
    matrix: Any
    target: Any
    rms_limit_T: float | None = None
    peak_limit_T: float | None = None
    group_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class ResearchPhysicsSpec:
    """Physical objectives, finite uncertainty scenarios and mechanical bounds.

    Mixtures are dimensionless sums weight*objective/reference_scale. Missing
    scales are measured at a minimum-sheet-power feasible reference solution,
    computed once at reference_tolerance_scale times the supplied field
    budgets. Explicit scales use keys power [W], magnetic_energy [J],
    minimax [A/m]. Tolerance sweeps reuse those original reference scales.
    Custom formulas use unweighted design-grid samples in SI. Auto uses a
    conic solver for DCP expressions and local SLSQP for nonconvex expressions.
    Its finite differences operate on the native reduced coefficients.
    """

    objective: str = "power"
    objective_weights: Mapping[str, float] = field(default_factory=lambda: {"power": 1.0, "magnetic_energy": 1.0})
    objective_scales: Mapping[str, float] = field(default_factory=dict)
    reference_tolerance_scale: float = 1.0
    evaluate_energy: bool = False
    sheet_power_limit_W: float | None = None
    magnetic_energy_limit_J: float | None = None
    b0_T: tuple[float, float, float] = (0.0, 0.0, 3.0)
    background_B_per_face_T: Any | None = None
    torque_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_limit_N: float | None = None
    torque_limit_Nm: float | None = None
    force_equality_N: tuple[float, float, float] | None = None
    torque_equality_Nm: tuple[float, float, float] | None = None
    field_constraints: tuple[LinearFieldConstraint, ...] = ()
    scenarios: tuple[FieldScenario, ...] = ()
    energy: EnergyQuadrature = field(default_factory=EnergyQuadrature)
    include_default_field_constraints: bool = True
    acceptance_relative_tolerance: float = 1.0e-5
    acceptance_absolute_by_unit: Mapping[str, float] = field(
        default_factory=lambda: {
            "T": 1e-10,
            "T/m": 1e-9,
            "N": 1e-8,
            "N m": 1e-9,
            "A": 1e-8,
            "A/m": 1e-6,
            "W": 1e-10,
            "J": 1e-12,
        }
    )
    custom_objective: str = ""
    custom_constraints: tuple[str, ...] = ()
    custom_solver: str = "auto"

    def __post_init__(self) -> None:
        objectives = {"power", "magnetic_energy", "minimax"}
        if self.objective not in objectives | {"mixture", "custom"}:
            raise ValueError("Research objective must be power, magnetic_energy, minimax, mixture, or custom.")
        if self.custom_solver not in {"auto", "conic", "local"}:
            raise ValueError("The custom solver must be auto, conic, or local.")
        formulas = self.formula_program()
        if self.objective == "custom" and not formulas.objective_text:
            raise ValueError("A custom research objective requires a scalar custom_objective expression.")
        if formulas.objective_text and self.objective != "custom":
            raise ValueError("Set objective to custom when specifying custom_objective.")
        object.__setattr__(self, "custom_constraints", tuple(formulas.constraint_lines))
        if set(self.objective_weights) - objectives or set(self.objective_scales) - objectives:
            raise ValueError("Research objective weight/scale keys must be power, magnetic_energy, or minimax.")
        if any(not np.isfinite(v) or v < 0 for v in self.objective_weights.values()):
            raise ValueError("Research objective weights must be finite and nonnegative.")
        if self.objective == "mixture" and not any(v > 0 for v in self.objective_weights.values()):
            raise ValueError("A research mixture needs at least one positive objective weight.")
        if any(not np.isfinite(v) or v <= 0 for v in self.objective_scales.values()):
            raise ValueError("Research objective scales must be finite and strictly positive, in SI units.")
        if not np.isfinite(self.reference_tolerance_scale) or self.reference_tolerance_scale < 0:
            raise ValueError("reference_tolerance_scale must be finite and nonnegative.")
        for name in ("b0_T", "torque_origin_m", "force_equality_N", "torque_equality_Nm"):
            value = getattr(self, name)
            if value is not None and (np.asarray(value).shape != (3,) or not np.all(np.isfinite(value))):
                raise ValueError(f"{name} must contain three finite Cartesian components.")
        for name in ("force_limit_N", "torque_limit_Nm", "sheet_power_limit_W", "magnetic_energy_limit_J"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not np.isfinite(self.acceptance_relative_tolerance) or not 0 <= self.acceptance_relative_tolerance < 1:
            raise ValueError("Research acceptance_relative_tolerance must be finite and between zero and one.")
        if any(not np.isfinite(v) or v < 0 for v in self.acceptance_absolute_by_unit.values()):
            raise ValueError("Research absolute acceptance tolerances must be finite and nonnegative.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | ResearchPhysicsSpec) -> ResearchPhysicsSpec:
        if isinstance(values, cls):
            return values
        values = dict(values)
        values["field_constraints"] = tuple(
            value if isinstance(value, LinearFieldConstraint) else LinearFieldConstraint(**value)
            for value in values.get("field_constraints", ())
        )
        values["scenarios"] = tuple(
            value if isinstance(value, FieldScenario) else FieldScenario(**value)
            for value in values.get("scenarios", ())
        )
        energy = values.get("energy", {})
        values["energy"] = energy if isinstance(energy, EnergyQuadrature) else EnergyQuadrature(**energy)
        return cls(**values)

    def to_dict(self, *, include_formula_defaults=False) -> dict[str, Any]:
        values = asdict(self)
        if (
            not include_formula_defaults
            and not self.custom_objective
            and not self.custom_constraints
            and self.custom_solver == "auto"
        ):
            # Preserve existing serialized physics specs and their checkpoint
            # identities when no formula controls have been configured.
            for name in ("custom_objective", "custom_constraints", "custom_solver"):
                del values[name]
        return _json_value(values)

    def formula_program(self):
        from .formulas import parse_formulas

        return parse_formulas(self.custom_objective, self.custom_constraints)


def preflight_research_physics(spec: ResearchPhysicsSpec | Mapping[str, Any], face_count: int) -> dict[str, Any]:
    """Validate energy availability explicitly before expensive operator assembly."""
    spec = ResearchPhysicsSpec.from_dict(spec)
    energy_required = (
        spec.evaluate_energy
        or spec.magnetic_energy_limit_J is not None
        or spec.objective == "magnetic_energy"
        or (spec.objective == "mixture" and spec.objective_weights.get("magnetic_energy", 0) > 0)
        or "energy_J" in spec.formula_program().symbols
    )
    return {"energy_requested": energy_required, **(spec.energy.preflight(face_count) if energy_required else {})}


@dataclass(frozen=True, slots=True)
class _Bound:
    name: str
    matrix: np.ndarray
    target: np.ndarray
    limit: float
    norm: str
    unit: str
    group_size: int = 1
    scalable: bool = True

    def measure(self, x: np.ndarray) -> float:
        residual = self.matrix @ x - self.target
        if not np.all(np.isfinite(residual)):
            return float("inf")
        if self.norm == "upper":
            return float(np.max(residual))
        if self.norm in {"linf", "equality"}:
            if self.group_size > 1:
                return float(np.max(np.linalg.norm(residual.reshape(-1, self.group_size), axis=1)))
            return float(np.max(np.abs(residual)))
        return float(np.linalg.norm(residual))


class ParameterizedResearchProblem:
    """Reuse the reduced physical operators across tolerance solves.

    ``solve(tolerance_scale=...)`` scales default field residual budgets and
    custom/scenario error budgets. Exact equalities, one-sided bounds, force,
    torque, current-density and contour-current limits remain fixed.
    No unsupported backend or scalar-search substitution is performed.
    """

    def __init__(self, problem: Any, spec: ResearchPhysicsSpec | Mapping[str, Any], config: Any):
        import cvxpy as cp
        from .. import optimization as core
        from ..engineering_limits import engineering_surface_current_density_limit_A_per_m
        from ..optimization_solvers import conic_solver_order
        from ..progress import ProgressReporter

        self.problem, self.config = problem, config
        # Own a snapshot: caller edits must not alter the compiled objective or
        # the acceptance contract while reusing this model for a sweep.
        self.spec = ResearchPhysicsSpec.from_dict(ResearchPhysicsSpec.from_dict(spec).to_dict())
        self.formulas = self.spec.formula_program()
        self.preflight = preflight_research_physics(self.spec, len(problem.combined_mesh.faces))
        if not conic_solver_order(config.constrained_optimization_solver):
            raise RuntimeError("Physical convex optimization requires an installed supported conic solver.")
        self.progress = ProgressReporter.from_config(config, label="Research optimization")
        self.progress.report(0, "validating surface topology, physical operators and objective")
        self.surface = surface_operators(problem)
        self.expansion = np.asarray(problem.reducer.expansion_matrix, dtype=float)
        self.force_operator, self.torque_operator = force_torque_operators(
            self.surface,
            self.spec.b0_T if self.spec.background_B_per_face_T is None else self.spec.background_B_per_face_T,
            self.spec.torque_origin_m,
        )
        self.weights = (
            dict(self.spec.objective_weights) if self.spec.objective == "mixture" else {self.spec.objective: 1.0}
        )
        self.weights = {key: value for key, value in self.weights.items() if value > 0}
        if self.spec.objective == "custom":
            self.weights = {}
        self.energy = (
            surface_magnetic_energy(
                self.surface,
                self.spec.energy,
                progress=lambda percent, message: self.progress.report(10 + percent * 0.35, message),
            )
            if self.preflight["energy_requested"]
            else None
        )
        self.power_matrix = self._psd_matrix(np.asarray(problem.reduced_resistance, dtype=float), "Sheet resistance")
        self.quadratic_bounds = tuple(
            (name, matrix, limit, unit)
            for name, matrix, limit, unit in (
                ("sheet_power", self.power_matrix, self.spec.sheet_power_limit_W, "W"),
                (
                    "magnetic_energy",
                    self.energy.matrix_J_per_A2 if self.energy else None,
                    self.spec.magnetic_energy_limit_J,
                    "J",
                ),
            )
            if limit is not None
        )
        if config.stream_function_smoothness_factor > 0 or config.turn_spacing_regularization_factor > 0:
            raise ValueError(
                "Research physical objectives require zero legacy smoothness/spacing penalties; choose an explicit physical mixture."
            )
        self.bounds: list[_Bound] = []
        self.standard_blocks = (
            core._target_error_constraint_blocks(problem, config) if self.spec.include_default_field_constraints else ()
        )
        for block in self.standard_blocks:
            self.bounds.append(
                _Bound(
                    "standard:" + block.name,
                    np.asarray(block.matrix),
                    np.asarray(block.target),
                    float(block.allowed_residual),
                    "linf" if block.metric == "linf" else "l2",
                    "T/m" if block.name in {"gradient_strength", "cross_gradient"} else "T",
                    block.group_size,
                )
            )
        if self.spec.include_default_field_constraints:
            availability = core._constrained_solution_contract_diagnostics(
                problem, np.zeros(self.expansion.shape[1]), config
            )
            if not availability["constrained_solver_constraint_data_available"]:
                raise ValueError(
                    "A requested standard physical constraint cannot be constructed: "
                    + str(availability["constrained_solver_failed_constraints"])
                )
        for constraint in self.spec.field_constraints:
            self._add_field_constraint(constraint)
        for scenario in self.spec.scenarios:
            if scenario.rms_limit_T is None and scenario.peak_limit_T is None:
                raise ValueError(f"Scenario {scenario.name!r} needs an RMS or peak field bound.")
            for label, limit, norm in (("rms", scenario.rms_limit_T, "rms"), ("peak", scenario.peak_limit_T, "linf")):
                if limit is not None:
                    self._add_field_constraint(
                        LinearFieldConstraint(
                            "scenario:" + scenario.name + ":" + label,
                            scenario.matrix,
                            scenario.target,
                            limit,
                            norm,
                            group_size=scenario.group_size,
                        )
                    )
        for label, matrix, limit, equality, unit in (
            ("force", self.force_operator, self.spec.force_limit_N, self.spec.force_equality_N, "N"),
            ("torque", self.torque_operator, self.spec.torque_limit_Nm, self.spec.torque_equality_Nm, "N m"),
        ):
            if limit is not None:
                self.bounds.append(_Bound(label, matrix, np.zeros(3), limit, "l2", unit, scalable=False))
            if equality is not None:
                target = np.asarray(equality, dtype=float)
                if target.shape != (3,):
                    raise ValueError(f"{label} equality requires three components in {unit}.")
                self.bounds.append(_Bound(label + "_equality", matrix, target, 0.0, "equality", unit, scalable=False))
        density_limit = engineering_surface_current_density_limit_A_per_m(config)
        if density_limit > 0:
            matrix = self.surface.current_density_per_A_m.reshape(-1, self.expansion.shape[1])
            self.bounds.append(
                _Bound("surface_current_density", matrix, np.zeros(len(matrix)), density_limit, "linf", "A/m", 3, False)
            )
        self.contour_limit = core._constrained_contour_step_limit(config)
        shield = core._shield_gradient_opposition_constraint_data(problem, config)
        if shield["requested"]:
            if not shield["enabled"]:
                raise ValueError("Requested shield-gradient bound is unavailable: " + shield["reason"])
            self.bounds.append(
                _Bound(
                    "shield_gradient_opposition",
                    np.asarray(shield["matrix"]).reshape(1, -1),
                    np.zeros(1),
                    float(shield["limit_T_per_m"]),
                    "upper",
                    "T/m",
                    scalable=False,
                )
            )
        names = [bound.name for bound in self.bounds] + [name for name, *_ in self.quadratic_bounds]
        if len(names) != len(set(names)):
            raise ValueError("Research constraint/scenario names must be unique.")
        if not self.bounds and not (self.spec.custom_constraints or self.spec.objective == "custom"):
            raise ValueError("A research design needs at least one physical field or mechanical constraint.")
        for bound in self.bounds:
            if (
                not np.all(np.isfinite(bound.matrix))
                or not np.all(np.isfinite(bound.target))
                or not np.isfinite(bound.limit)
            ):
                raise ValueError(f"Constraint {bound.name!r} contains nonfinite values.")

        # A physical drive estimate scales variables, without altering the basis.
        reference = np.linalg.lstsq(problem.reduced_sensitivity, problem.target_z, rcond=1e-12)[0]
        drive_scales = [float(np.max(np.abs(reference)))]
        for bound in self.bounds:
            gain = float(np.max(np.abs(bound.matrix)))
            if gain > 0:
                drive_scales.append(float(np.max(np.abs(bound.target))) / gain)
        self.current_scale = max(drive_scales) if max(drive_scales) > 0 else 1.0
        self.variable = cp.Variable(self.expansion.shape[1])
        self.x = self.current_scale * self.variable
        self.tolerance_scale = cp.Parameter(nonneg=True, value=self.spec.reference_tolerance_scale)
        constraints = []
        for bound in self.bounds:
            scale = max(
                float(np.max(np.abs(bound.target))),
                abs(bound.limit),
                float(np.max(np.abs(bound.matrix))) * self.current_scale,
                1e-300,
            )
            residual = (bound.matrix / scale) @ self.x - bound.target / scale
            budget = bound.limit / scale * (self.tolerance_scale if bound.scalable else 1.0)
            if bound.norm == "equality":
                constraints.append(residual == 0)
            elif bound.norm == "upper":
                constraints.append(residual <= budget)
            elif bound.norm == "linf":
                if bound.group_size > 1:
                    grouped = cp.reshape(residual, (len(bound.target) // bound.group_size, bound.group_size), order="C")
                    constraints.append(cp.norm(grouped, 2, axis=1) <= budget)
                else:
                    constraints.append(cp.norm(residual, "inf") <= budget)
            else:
                constraints.append(cp.norm(residual, 2) <= budget)
        if self.contour_limit["contour_step_constraint_enabled"]:
            limit = self.contour_limit["contour_step_constraint_max_range_A"]
            full = self.expansion @ self.x / limit
            constraints.append(cp.max(full) - cp.min(full) <= 1)
        for _name, matrix, limit, _unit in self.quadratic_bounds:
            scale = max(float(np.linalg.norm(matrix, 2)) * self.current_scale**2, limit, 1e-300)
            constraints.append(
                cp.quad_form(self.variable, cp.psd_wrap(matrix * self.current_scale**2 / scale)) <= limit / scale
            )
        self.peak_scale = max(
            float(np.max(np.linalg.norm(self.surface.current_density_per_A_m @ reference, axis=1))), 1.0
        )
        self.peak = cp.Variable(nonneg=True)
        # Native physical constraints already use dimensionless residuals.
        # Keep these same expressions for both the conic seed and local solve.
        self.native_constraints = tuple(constraints)
        if "minimax" in self.weights:
            current = self.surface.current_density_per_A_m.reshape(-1, self.expansion.shape[1]) / self.peak_scale
            grouped = cp.reshape(current @ self.x, (len(self.surface.areas_m2), 3), order="C")
            constraints.append(cp.norm(grouped, 2, axis=1) <= self.peak)
        custom_objective = None
        self.use_local = False
        self.formula_scales = []
        self.formula_objective_scale = 1.0
        if self.formulas.objective_text or self.formulas.constraint_lines:
            from .formulas import NonConvexFormulaError

            self.use_local = self.spec.custom_solver == "local"
            custom_bounds = []
            if not self.use_local:
                try:
                    custom_objective, custom_bounds = self.formulas.compile(self._formula_bindings(self.x, cp))
                except NonConvexFormulaError:
                    if self.spec.custom_solver == "conic":
                        raise
                    self.use_local = True
            # Positive row/objective scaling preserves the problem, while e.g.
            # microtesla inequalities remain visible at conic feasibility precision.
            self.formula_objective_scale, self.formula_scales = self._formula_scaling(reference)
            if not self.use_local:
                for bound, scale in zip(custom_bounds, self.formula_scales, strict=True):
                    residual = bound.expr / (scale or 1.0)
                    constraints.append(residual == 0 if isinstance(bound, cp.constraints.Equality) else residual <= 0)
        self.constraints = constraints
        self.objective_scales = dict(self.spec.objective_scales)
        self.reference_diagnostics: dict[str, Any] = {}
        if self.use_local:
            self.seed_problem = cp.Problem(
                cp.Minimize(self._quadratic_expression(cp, self.power_matrix)), self.native_constraints
            )
            self.least_squares_seed = reference.copy() / self.current_scale
        if self.spec.objective == "mixture" and any(key not in self.objective_scales for key in self.weights):
            if self.use_local:
                self._solve_local(minimize_power=True)
            else:
                reference_problem = cp.Problem(
                    cp.Minimize(self._quadratic_expression(cp, self.power_matrix)), constraints
                )
                self._run_backend(reference_problem)
            reference = np.asarray(self.x.value).reshape(-1)
            rows = self._check_bounds(reference, self.spec.reference_tolerance_scale)
            if not all(row["met"] for row in rows):
                raise RuntimeError(
                    "Minimum-power reference failed physical acceptance; cannot normalize the objective mixture."
                )
            values = self._physical_values(reference)
            mapping = {
                "power": "sheet_power_W",
                "magnetic_energy": "energy_J",
                "minimax": "max_surface_current_density_A_per_m",
            }
            for key in self.weights:
                if key not in self.objective_scales:
                    value = float(values[mapping[key]])
                    if not np.isfinite(value) or value <= 1e-300:
                        raise ValueError(
                            f"Feasible reference has zero {key}; supply an explicit positive objective_scales[{key!r}]."
                        )
                    self.objective_scales[key] = value
            self.reference_diagnostics = {
                "source": "minimum_power_feasible_at_tolerance_scale_1"
                if self.spec.reference_tolerance_scale == 1.0
                else "minimum_power_feasible_at_reference_tolerance_scale",
                "tolerance_scale": self.spec.reference_tolerance_scale,
                **values,
            }
            if self.use_local:
                self.reference_diagnostics["source"] = "locally_minimized_power_feasible_at_reference_tolerance_scale"
        terms = []
        for key, weight in self.weights.items():
            if key == "minimax":
                scale = self.objective_scales.setdefault(key, self.peak_scale)
                terms.append(weight * self.peak * self.peak_scale / scale)
            else:
                matrix = self.power_matrix if key == "power" else self.energy.matrix_J_per_A2
                scale = self.objective_scales.setdefault(
                    key, max(float(np.linalg.norm(matrix, 2)) * self.current_scale**2, 1e-300)
                )
                terms.append(weight * cp.quad_form(self.variable, cp.psd_wrap(matrix * self.current_scale**2 / scale)))
        objective = custom_objective / self.formula_objective_scale if custom_objective is not None else sum(terms)
        self.cvxpy_problem = None if self.use_local else cp.Problem(cp.Minimize(objective), constraints)
        if self.cvxpy_problem is not None and not self.cvxpy_problem.is_dcp(dpp=True):
            raise ValueError("Research formulation must be convex and parameterized without recanonicalization.")
        self.solve_count = 0
        method = "local nonlinear problem" if self.use_local else "reusable convex graph"
        self.progress.report(60, f"compiled {method}: {len(self.bounds)} physical bounds")

    @staticmethod
    def _psd_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} matrix must be finite and square.")
        matrix = (matrix + matrix.T) / 2
        values, vectors = np.linalg.eigh(matrix)
        scale = max(float(np.max(np.abs(values))), 1e-300)
        if values[0] < -1e-10 * scale:
            raise ValueError(f"{name} matrix is not positive semidefinite.")
        return (vectors * np.maximum(values, 0)) @ vectors.T if values[0] < 0 else matrix

    def _quadratic_expression(self, cp: Any, matrix: np.ndarray) -> Any:
        scale = max(float(np.linalg.norm(matrix, 2)) * self.current_scale**2, 1e-300)
        return cp.quad_form(self.variable, cp.psd_wrap(matrix * self.current_scale**2 / scale))

    def _formula_bindings(self, x, cp=None):
        """Same physical operands for the conic graph and independent NumPy evaluation."""
        symbols = self.formulas.symbols
        values = {}
        if "psi_A" in symbols:
            values["psi_A"] = self.expansion @ x
        if symbols.intersection({"field_T", "residual_T", "target_T"}):
            matrix, target = self.problem.unweighted_reduced_sensitivity, self.problem.unweighted_target_z
            if matrix is None or target is None:
                raise ValueError("Custom field expressions require unweighted design-grid sensitivity and target data.")
            values["target_T"] = target
            if symbols.intersection({"field_T", "residual_T"}):
                values["field_T"] = matrix @ x
                values["residual_T"] = values["field_T"] - target
        for name, matrix in (
            ("power_W", self.power_matrix),
            ("energy_J", self.energy.matrix_J_per_A2 if self.energy is not None else None),
        ):
            if name in symbols:
                values[name] = float(x @ matrix @ x) if cp is None else cp.quad_form(x, cp.psd_wrap(matrix))
        if "peak_current_density_A_per_m" in symbols:
            operator = self.surface.current_density_per_A_m
            if cp is None:
                peak = np.max(np.linalg.norm(operator @ x, axis=1))
            else:
                current = cp.reshape(operator.reshape(-1, operator.shape[-1]) @ x, operator.shape[:2], order="C")
                peak = cp.max(cp.norm(current, 2, axis=1))
            values["peak_current_density_A_per_m"] = peak
        if "force_N" in symbols:
            values["force_N"] = self.force_operator @ x
        if "torque_Nm" in symbols:
            values["torque_Nm"] = self.torque_operator @ x
        return values

    def _formula_scaling(self, reference):
        objective_scale = 0.0
        scales = np.zeros(len(self.spec.custom_constraints))
        for x in (np.zeros_like(reference), reference):
            try:
                objective, comparisons = self.formulas.evaluate(self._formula_bindings(x))
            except ValueError:
                # A log, fractional power or reciprocal can have a restricted
                # domain. A scaling probe is not a feasibility decision.
                continue
            if objective is not None:
                objective_scale = max(objective_scale, abs(objective))
            for i, (left, _, right) in enumerate(comparisons):
                scales[i] = max(scales[i], float(np.max(np.abs(left))), float(np.max(np.abs(right))))
        return objective_scale or 1.0, scales

    def _formula_checks(self, x):
        objective, comparisons = self.formulas.evaluate(self._formula_bindings(x))
        rows = []
        for i, (lhs, operator, rhs) in enumerate(comparisons):
            left, right = np.broadcast_arrays(lhs, rhs)
            residual = left - right if operator == "<=" else right - left
            if operator == "==":
                residual = np.abs(residual)
            # Units are determined by the authored expression. The solver's
            # dimensionless precision is converted using this row's SI scale.
            allowance = self.config.constrained_optimization_tolerance * self.formula_scales[i]
            effective = allowance + self.spec.acceptance_relative_tolerance * np.maximum(np.abs(left), np.abs(right))
            worst = int(np.argmax(residual - effective))
            rows.append(
                {
                    "name": f"custom:{i + 1} {self.spec.custom_constraints[i]}",
                    "lhs": float(left.flat[worst]),
                    "rhs": float(right.flat[worst]),
                    "operator": operator,
                    "value": float(residual.flat[worst]),
                    "limit": 0.0,
                    "effective_limit": float(effective.flat[worst]),
                    "unit": "expression units",
                    "met": bool(np.all(residual <= effective)),
                }
            )
        return objective, rows

    def _add_field_constraint(self, item: LinearFieldConstraint) -> None:
        matrix, target = np.asarray(item.matrix, dtype=float), np.asarray(item.target, dtype=float).reshape(-1)
        if matrix.ndim != 2 or matrix.shape != (len(target), self.expansion.shape[0]) or not len(target):
            raise ValueError(
                f"Field constraint {item.name!r} needs one matrix column per full mesh vertex and one target per row."
            )
        if item.norm not in {"l2", "rms", "linf", "equality", "upper"}:
            raise ValueError(f"Unknown field-constraint norm {item.norm!r}.")
        if (
            isinstance(item.group_size, bool)
            or int(item.group_size) != item.group_size
            or item.group_size < 1
            or len(target) % item.group_size
        ):
            raise ValueError("Field group_size must be a positive integer dividing the matrix row count.")
        if item.norm in {"equality", "upper"} and item.group_size != 1:
            raise ValueError("Equalities and componentwise upper bounds require group_size=1.")
        if item.unit not in self.spec.acceptance_absolute_by_unit:
            raise ValueError(f"Specify an explicit acceptance_absolute_by_unit for unit {item.unit!r}.")
        if not np.isfinite(item.tolerance) or item.tolerance < 0:
            raise ValueError("Field tolerances must be finite and nonnegative.")
        if item.norm == "equality" and item.tolerance != 0:
            raise ValueError("Exact field equality requires tolerance=0.")
        matrix = matrix @ self.expansion
        if item.norm == "rms":
            divisor = np.sqrt(len(target) / item.group_size)
            matrix, target = matrix / divisor, target / divisor
        self.bounds.append(
            _Bound(
                item.name,
                matrix,
                target,
                item.tolerance,
                item.norm,
                item.unit,
                item.group_size,
                scalable=item.norm not in {"equality", "upper"},
            )
        )

    def _run_backend(self, cvxpy_problem: Any, *, native_only=False) -> tuple[str, list[str]]:
        from ..optimization import _cvxpy_solver_options
        from ..optimization_solvers import conic_solver_order

        errors = []
        for solver in conic_solver_order(self.config.constrained_optimization_solver):
            self.progress.report(75, f"solving with {solver}; tolerance scale {self.tolerance_scale.value:g}")
            try:
                cvxpy_problem.solve(solver=solver, warm_start=True, **_cvxpy_solver_options(solver, self.config))
            except Exception as exc:
                errors.append(f"{solver}: {exc}")
                continue
            if cvxpy_problem.status in {"optimal", "optimal_inaccurate"} and self.x.value is not None:
                candidate = np.asarray(self.x.value, dtype=float).reshape(-1)
                try:
                    rows = self._check_bounds(candidate, float(self.tolerance_scale.value), native_only=native_only)
                except (ValueError, FloatingPointError) as exc:
                    errors.append(f"{solver}: numerical acceptance could not evaluate the candidate: {exc}")
                    continue
                failures = [row["name"] for row in rows if not row["met"]]
                if not failures:
                    return solver, errors
                errors.append(f"{solver}: physical acceptance failed for {', '.join(failures)}")
                continue
            errors.append(f"{solver}: {cvxpy_problem.status}")
        raise RuntimeError("No research conic solve succeeded. " + "; ".join(errors))

    def _check_bounds(self, x: np.ndarray, scale: float, *, native_only=False) -> list[dict[str, Any]]:
        rows = []
        finite = bool(np.all(np.isfinite(x)))
        for bound in self.bounds:
            measured = bound.measure(x)
            limit = bound.limit * (scale if bound.scalable else 1.0)
            tolerance = self.spec.acceptance_absolute_by_unit.get(bound.unit, 0.0)
            tolerance += self.spec.acceptance_relative_tolerance * max(abs(limit), float(np.max(np.abs(bound.target))))
            rows.append(
                {
                    "name": bound.name,
                    "value": measured,
                    "limit": limit,
                    "effective_limit": limit + tolerance,
                    "unit": bound.unit,
                    "met": bool(finite and np.isfinite(measured) and measured <= limit + tolerance),
                }
            )
        for name, matrix, limit, unit in self.quadratic_bounds:
            value = float(x @ matrix @ x)
            allowance = self.spec.acceptance_absolute_by_unit.get(unit, 0.0)
            allowance += self.spec.acceptance_relative_tolerance * limit
            rows.append(
                {
                    "name": name,
                    "value": value,
                    "limit": limit,
                    "effective_limit": limit + allowance,
                    "unit": unit,
                    "met": bool(finite and np.isfinite(value) and value <= limit + allowance),
                }
            )
        if self.contour_limit["contour_step_constraint_enabled"]:
            limit = float(self.contour_limit["contour_step_constraint_max_range_A"])
            value = float(np.ptp(self.expansion @ x))
            effective = limit * (
                1 + self.spec.acceptance_relative_tolerance
            ) + self.spec.acceptance_absolute_by_unit.get("A", 0)
            rows.append(
                {
                    "name": "contour_step",
                    "value": value,
                    "limit": limit,
                    "effective_limit": effective,
                    "unit": "A",
                    "met": bool(finite and value <= effective),
                }
            )
        if "minimax" in self.weights and not self.use_local:
            value = float(np.max(np.linalg.norm(self.surface.current_density_per_A_m @ x, axis=1)))
            limit = float(self.peak.value) * self.peak_scale if self.peak.value is not None else float("nan")
            effective = limit * (
                1 + self.spec.acceptance_relative_tolerance
            ) + self.spec.acceptance_absolute_by_unit.get("A/m", 0)
            rows.append(
                {
                    "name": "minimax_epigraph",
                    "value": value,
                    "limit": limit,
                    "effective_limit": effective,
                    "unit": "A/m",
                    "met": bool(finite and np.isfinite(limit) and value <= effective),
                }
            )
        if self.spec.custom_constraints and not native_only:
            rows.extend(self._formula_checks(x)[1])
        return rows

    def _physical_values(self, x: np.ndarray) -> dict[str, Any]:
        return {
            "sheet_power_W": float(x @ self.power_matrix @ x),
            "energy_J": self.energy.energy_J(x) if self.energy is not None else None,
            "force_N": (self.force_operator @ x).tolist(),
            "torque_Nm": (self.torque_operator @ x).tolist(),
            "torque_origin_m": list(self.spec.torque_origin_m),
            "b0_T": list(self.spec.b0_T),
            "force_model": "uniform_B0_total_closed_current_force_vanishes"
            if self.spec.background_B_per_face_T is None
            else "prescribed_piecewise_constant_background_B_per_face",
            "max_surface_current_density_A_per_m": float(
                np.max(np.linalg.norm(self.surface.current_density_per_A_m @ x, axis=1))
            ),
            "stream_function_range_A": float(np.ptp(self.expansion @ x)),
            "reduced_coefficient_norm": float(np.linalg.norm(x)),
        }

    def _solve_local(self, *, minimize_power=False):
        """Run SLSQP in the same scaled reduced basis, starting from native feasibility."""
        import cvxpy as cp
        from scipy.optimize import minimize

        seed_solver, attempts = self._run_backend(self.seed_problem, native_only=True)
        start = np.array(self.variable.value, dtype=float, copy=True)
        if minimize_power:
            scale = max(float(np.linalg.norm(self.power_matrix, 2)) * self.current_scale**2, 1e-300)
        # Compile the existing modeled objective for local comparisons too.
        # Its minimax term uses the actual largest face current at each trial.
        physical_terms = {
            "power": lambda x: float(x @ self.power_matrix @ x),
            "magnetic_energy": lambda x: self.energy.energy_J(x),
            "minimax": lambda x: float(np.max(np.linalg.norm(self.surface.current_density_per_A_m @ x, axis=1))),
        }
        last_point, last_result = None, None

        def quantities(point):
            nonlocal last_point, last_result
            if last_point is not None and np.array_equal(point, last_point):
                return last_result
            self.variable.value = point
            x = self.current_scale * point
            authored_objective, comparisons = self.formulas.evaluate(self._formula_bindings(x))
            if minimize_power:
                objective = float(x @ self.power_matrix @ x) / scale
            elif authored_objective is not None:
                objective = authored_objective / self.formula_objective_scale
            else:
                objective = sum(
                    weight * physical_terms[key](x) / self.objective_scales[key] for key, weight in self.weights.items()
                )
            # SLSQP accepts inequalities as g(x) >= 0. Each equality is two
            # inequalities to support redundant rows (e.g. zero net force).
            margins = []
            for constraint in self.native_constraints:
                residual = np.asarray(constraint.expr.value, dtype=float).reshape(-1)
                margins.append(-residual)
                if isinstance(constraint, cp.constraints.Equality):
                    margins.append(residual)
            for (lhs, op, rhs), row_scale in zip(comparisons, self.formula_scales, strict=True):
                residual = np.asarray((lhs - rhs) / (row_scale or 1.0)).reshape(-1)
                margins.append(residual if op == ">=" else -residual)
                if op == "==":
                    margins.append(residual)
            last_point = point.copy()
            last_result = objective, np.concatenate(margins) if margins else np.empty(0)
            return last_result

        initial_guess = "minimum_power_subject_to_native_constraints"
        try:
            _, seed_margins = quantities(start)
        except ValueError as exc:
            # Some valid formula domains (e.g. log of a positive field)
            # exclude the minimum-power solution. Try the physical drive
            # estimate only when it independently passes the native limits.
            start = self.least_squares_seed.copy()
            native = self._check_bounds(self.current_scale * start, self.tolerance_scale.value, native_only=True)
            if not all(row["met"] for row in native):
                raise ValueError(f"Local formula has no finite, native-feasible initial point: {exc}") from exc
            try:
                _, seed_margins = quantities(start)
            except ValueError as second:
                raise ValueError(f"Local formula is undefined at both feasible starting points: {second}") from second
            initial_guess = "native_feasible_weighted_least_squares"

        def trial(point):
            try:
                return quantities(point)
            except (ValueError, FloatingPointError):
                # Allow the line search to reject trial points outside the
                # real domain. The returned candidate is evaluated afresh.
                return np.inf, np.full(seed_margins.shape, -np.inf)

        constraints = [{"type": "ineq", "fun": lambda z: trial(z)[1]}] if len(seed_margins) else []
        options = {
            "maxiter": self.config.constrained_optimization_max_iterations,
            "ftol": self.config.constrained_optimization_tolerance,
        }
        self.progress.report(None, f"Starting local SLSQP from {initial_guess}")
        result = minimize(
            lambda z: trial(z)[0],
            start,
            method="SLSQP",
            constraints=constraints,
            options=options,
            callback=lambda _z: self.progress.report(None, "Checking local SLSQP iteration"),
        )
        if not result.success or not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Local SLSQP did not converge (status {result.status}): {result.message}")
        self.variable.value = result.x
        self.local_diagnostics = {
            "method": "SciPy SLSQP",
            "global_optimality_claim": False,
            "initial_guess": initial_guess,
            "initial_solver": seed_solver,
            "status_code": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective_evaluations": int(result.nfev),
            "objective_value_scaled": float(result.fun),
            "derivatives": "forward finite differences in current-scaled reduced coefficients",
            **options,
        }
        return "SLSQP", attempts

    def solve(self, *, tolerance_scale: float = 1.0) -> Any:
        from ..optimization import StreamFunctionSolution, _residual_constraint_diagnostics

        if not np.isfinite(tolerance_scale) or tolerance_scale < 0:
            raise ValueError("tolerance_scale must be finite and nonnegative.")
        self.tolerance_scale.value = tolerance_scale
        solver, attempts = self._solve_local() if self.use_local else self._run_backend(self.cvxpy_problem)
        x = np.asarray(self.x.value, dtype=float).reshape(-1)
        rows = self._check_bounds(x, tolerance_scale)
        failed = [row for row in rows if not row["met"]]
        if failed:
            raise RuntimeError(
                "Research physical acceptance failed: "
                + "; ".join(
                    f"{row['name']}={row['value']:.6g} {row['unit']} > {row['effective_limit']:.6g}" for row in failed
                )
            )
        self.solve_count += 1
        self.progress.report(95, "independent physical acceptance passed")
        diagnostics = (
            _residual_constraint_diagnostics(
                self.problem,
                x,
                tuple(
                    replace(block, allowed_residual=block.allowed_residual * tolerance_scale)
                    for block in self.standard_blocks
                ),
                self.config,
            )
            if self.standard_blocks
            else {}
        )
        diagnostics.update(
            {
                "stream_function_scheme": "research_local" if self.use_local else "research_convex",
                "research_physics_enabled": True,
                "field_evaluation_current_source": "stream_function",
                "field_evaluation_current_source_requested": self.config.field_evaluation_current_source,
                "physical_current_refit_disabled": True,
                "research_objective": self.spec.objective,
                "research_objective_scales": dict(self.objective_scales),
                "research_reference": dict(self.reference_diagnostics),
                "research_preflight": dict(self.preflight),
                "research_tolerance_scale": tolerance_scale,
                "research_solve_count": self.solve_count,
                "research_parameterized_dpp": not self.use_local,
                "research_constraint_diagnostics": rows,
                "research_scenario_count": len(self.spec.scenarios),
                "constrained_solver_backend": "scipy_local" if self.use_local else "cvxpy_research",
                "constrained_solver_name": solver,
                "constrained_solver_status": "local_solution" if self.use_local else self.cvxpy_problem.status,
                "constrained_solver_previous_attempts": attempts,
                "constrained_solver_contract_met": True,
                "constrained_solver_physical_constraints_met": True,
                "constrained_solver_hard_constraints_enforced": True,
                "constrained_solver_diagnostic_result": False,
                "constrained_solver_failed_constraints": [],
                "constrained_solver_contract_constraint_diagnostics": rows,
                "constrained_solver_objective_value": self.local_diagnostics["objective_value_scaled"]
                if self.use_local
                else float(self.cvxpy_problem.value),
                "regularization_factor": 0.0,
                "regularization_lambda": 0.0,
                "selected_factor": 0.0,
                "selected_lambda": 0.0,
                "selection_method": "research_local" if self.use_local else "research_convex",
                **self._physical_values(x),
            }
        )
        if self.formulas.objective_text or self.formulas.constraint_lines:
            diagnostics.update(
                research_custom_objective=self.spec.custom_objective,
                research_custom_objective_value=self._formula_checks(x)[0],
                research_custom_constraints=list(self.spec.custom_constraints),
                research_custom_solver=self.spec.custom_solver,
                research_custom_solver_used="local" if self.use_local else "conic",
                research_initial_guess=self.local_diagnostics["initial_guess"]
                if self.use_local
                else "conic_solver_default_with_warm_start_on_reuse",
                research_custom_objective_scale=self.formula_objective_scale,
                research_custom_constraint_scales=self.formula_scales.tolist(),
            )
            if self.use_local:
                diagnostics["research_local_solver"] = dict(self.local_diagnostics)
        if self.energy is not None:
            diagnostics["research_energy_quadrature"] = {
                "order": self.energy.quadrature_order,
                "relative_change": self.energy.relative_change,
                "kernel_relative_change": self.energy.kernel_relative_change,
                "minimum_eigenvalue_J_per_A2": self.energy.minimum_eigenvalue_J_per_A2,
                "roundoff_psd_correction_J_per_A2": self.energy.roundoff_psd_correction_J_per_A2,
                "model": "zero_thickness_surface_double_integral_with_analytic_self_potential",
            }
        return StreamFunctionSolution(x, diagnostics)

    def solve_many(self, tolerance_scales: Iterable[float]) -> tuple[Any, ...]:
        """Solve a fixed-operator tolerance sweep using the selected method.

        Reference scales remain those fixed during construction, independent
        of sweep order. Hardware bounds and equalities remain fixed. A failed
        point raises; use the study runtime to record failures independently.
        """
        scales = tuple(float(value) for value in tolerance_scales)
        if any(not np.isfinite(value) or value < 0 for value in scales):
            raise ValueError("Every tolerance_scale must be finite and nonnegative.")
        return tuple(self.solve(tolerance_scale=value) for value in scales)


def solve_research_problem(problem: Any, config: Any) -> Any:
    return ParameterizedResearchProblem(problem, config.research_physics, config).solve()
