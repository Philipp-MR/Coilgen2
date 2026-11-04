"""Free-form scientific expressions share syntax and semantics across solver and check."""

import numpy as np
import pytest

from coilgen.research.formulas import (
    FORMULA_EXAMPLE_CONSTRAINTS,
    FORMULA_EXAMPLE_OBJECTIVE,
    NonConvexFormulaError,
    parse_formulas,
)


def test_shared_immutable_program_compiles_and_checks_the_same_physical_expressions():
    cp = pytest.importorskip("cvxpy")
    program = parse_formulas(
        FORMULA_EXAMPLE_OBJECTIVE + " + 2^2 + norm2(psi_A)",
        FORMULA_EXAMPLE_CONSTRAINTS + "\npsi_A >= -3\n  force_N[-1] = 0",
    )
    x = cp.Variable(3)
    x.value = [-2.0, 0.5, 0]

    def bindings(x):
        return dict(psi_A=x, residual_T=0.001 * x, power_W=5, force_N=x, torque_Nm=x / 2)

    numeric, rows = program.evaluate(bindings(x.value))
    opt, constraints = program.compile(bindings(x))
    assert program.symbols == {"psi_A", "residual_T", "power_W", "force_N", "torque_Nm"}
    assert program.constraint_lines[-1] == "force_N[-1] = 0"
    assert numeric == pytest.approx(0.5 + 4.25 + 4 + np.linalg.norm(x.value))
    assert opt.value == pytest.approx(numeric) and opt.is_convex()
    assert all(c.is_dcp() for c in constraints)
    for (lhs, rel, rhs), c in zip(rows, constraints):
        numerical = abs(lhs - rhs) if rel == "==" else np.maximum(0, lhs - rhs if rel == "<=" else rhs - lhs)
        np.testing.assert_allclose(numerical, c.violation())


@pytest.mark.parametrize(
    "expr",
    [
        "sum(abs(psi_A))",
        "sum_squares(psi_A)",
        "mean(square(psi_A))",
        "square(psi_A[0])",
        "norm1(psi_A)",
        "norm2(psi_A)",
        "norm_inf(psi_A)",
        "rms(psi_A)",
        "norm2(force_N[-1])",
        "max(psi_A)",
        "-min(psi_A)",
        "max(max(psi_A, -1))",
        "-min(min(psi_A, 2))",
        "sum(psi_A * target_T / 3)",
        "-sqrt(power_W)",
        "exp(power_W)",
        "-log(power_W)",
        "sum((psi_A/2)**2)",
    ],
)
def test_numpy_and_cvxpy_use_the_same_full_vectors_and_reductions(expr):
    cp = pytest.importorskip("cvxpy")
    field = np.array([-1.0, 0.25, 0.7])
    variable, power = cp.Variable(3), cp.Variable(nonneg=True)
    variable.value, power.value = field, 0.5
    program = parse_formulas(expr)
    numerical, _ = program.evaluate(dict(psi_A=field, force_N=field, target_T=-field, power_W=0.5))
    objective, _ = program.compile(dict(psi_A=variable, force_N=variable, target_T=-field, power_W=power))
    assert objective.value == pytest.approx(numerical)


@pytest.mark.parametrize(
    "objective",
    [
        "__import__('os').system('echo bad')",
        "open('data')",
        "psi_A.shape",
        "np.sum(psi_A)",
        "[1,2][1]",
        "norm2(psi_A, 2)",
        "sum(psi_A, axis=0)",
        "(lambda x: x)(1)",
        "True",
        "1e999",
        "target_T[psi_A]",
        "psi_A[2:9]",
        "psi_A[100000001]",
        "power_W**target_T[0]",
        "power_W ** 999999",
        "getattr(power_W, 'name')",
        "[x for x in psi_A]",
        "power_W or 0",
        "psi_A",
        "psi_A + 1",
        "(power_W <= 1)",
        "power_W\n+2",
        "unknown",
        "1j",
    ],
)
def test_parser_rejects_unsupported_or_nonscalar_objectives_without_running_them(objective):
    with pytest.raises(ValueError, match="objective"):
        parse_formulas(objective)


@pytest.mark.parametrize("line", ["psi_A", "0 < force_N[1]", "0 <= power_W <= 3", "psi_A != 0", "1 == unknown"])
def test_exact_relations_are_checked_line_by_line(line):
    with pytest.raises(ValueError, match="line 2"):
        parse_formulas("", "norm_inf(psi_A) <= 5\n" + line)


def test_empty_program_budgets_and_detached_source():
    assert parse_formulas().evaluate({}) == (None, [])
    constraints = ["  power_W <= 2  ", ""]
    program = parse_formulas("", constraints)
    constraints.clear()
    assert program.constraint_lines == ("power_W <= 2",)
    with pytest.raises(ValueError):
        parse_formulas("a" * 2049)
    for malformed in ([""] * 65, [True], None):
        with pytest.raises(ValueError):
            parse_formulas("", malformed)


@pytest.mark.parametrize(
    "expression,bindings",
    [
        ("psi_A[3]", {"psi_A": np.ones(3)}),
        ("abs(power_W)", {"power_W": [1, 2]}),
        ("sum(psi_A + target_T)", {"psi_A": [1], "target_T": [1, 2]}),
        ("norm2(force_N)", {"force_N": [1, 2]}),
        ("sqrt(power_W)", {"power_W": -1}),
        ("energy_J", {}),
        ("energy_J", {"energy_J": None}),
        ("sum(psi_A)", {"psi_A": [np.nan]}),
        ("norm2(force_N)", {"force_N": [False, True, True]}),
    ],
)
def test_binding_errors_do_not_yield_an_objective(expression, bindings):
    with pytest.raises(ValueError):
        parse_formulas(expression).evaluate(bindings)


def test_nonconvex_expressions_evaluate_numerically_with_line_specific_convex_diagnostics():
    cp = pytest.importorskip("cvxpy")
    x = cp.Variable(3)
    x.value = [-2, 1, 0]
    for expr in ("psi_A[0]^3", "sin(psi_A[0])", "psi_A[0]*psi_A[1]"):
        program = parse_formulas(expr)
        assert np.isfinite(program.evaluate({"psi_A": x.value})[0])
        with pytest.raises(NonConvexFormulaError, match="Custom objective:.*DCP"):
            program.compile({"psi_A": x})
    program = parse_formulas("0", "norm2(psi_A) == 1")
    with pytest.raises(NonConvexFormulaError, match="Custom constraint.*equalities must be affine"):
        program.compile({"psi_A": x})
    with pytest.raises(ValueError, match="matching shapes"):
        parse_formulas("", "psi_A <= target_T").evaluate({"psi_A": [1, 2, 3], "target_T": [2]})


def test_static_zero_denominator_is_invalid_for_both_solver_routes():
    cp = pytest.importorskip("cvxpy")
    program = parse_formulas("norm1(psi_A/target_T)")
    for run, x in ((program.compile, cp.Variable(2)), (program.evaluate, [0.3, 0.4])):
        with pytest.raises(ValueError) as error:
            run(dict(psi_A=x, target_T=[0, 0.6]))
        assert not isinstance(error.value, NonConvexFormulaError)


def test_rms_is_finite_for_a_representable_result_even_if_squares_overflow():
    value, _ = parse_formulas("rms(field_T)").evaluate({"field_T": np.full(4, 1e308)})
    assert value == pytest.approx(1e308)


def test_even_powers_preserve_negative_feasible_stream_function_coefficients():
    cp = pytest.importorskip("cvxpy")
    x = cp.Variable(1)
    program = parse_formulas("psi_A[0]^6", "psi_A[0] = -1")
    objective, constraints = program.compile({"psi_A": x})
    optimum = cp.Problem(cp.Minimize(objective), constraints).solve(solver="CLARABEL")
    assert optimum == pytest.approx(1.0, rel=1e-6)
