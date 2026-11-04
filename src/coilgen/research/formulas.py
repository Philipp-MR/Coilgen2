"""Engineering formulas for stream function optimization, with SI operands.

Parse once with `parse_formulas`. Supply the same named quantities to
`program.compile(bindings)` for CVXPY and `program.evaluate(bindings)` for
numerical checks. A scalar objective is minimized. Each constraint line uses
<=, >=, or == (= also works); comparisons on vectors apply element by element.
Arithmetic is elementwise, norms and reductions operate over the whole vector,
and indexing selects one zero-based component (negative indices are supported).
Use explicit SI reference scales when combining quantities of different units.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import operator
import re
from sys import float_info
from types import MappingProxyType

import numpy as np


FORMULA_SYMBOLS = MappingProxyType(
    {
        "psi_A": "Full nodal stream function [A]",
        "field_T": "Selected design-grid field [T], including suppression samples",
        "target_T": "Desired field on the same design grid [T]",
        "residual_T": "field_T minus target_T [T]; vector targets interleave Bx, By, Bz at each point",
        "power_W": "Resistive stream function power [W]",
        "energy_J": "Magnetic energy [J]; requires energy quadrature",
        "peak_current_density_A_per_m": "Maximum face current-density magnitude [A/m]",
        "force_N": "Net Cartesian force [N]",
        "torque_Nm": "Net Cartesian torque [N m]",
    }
)
_VECTORS = {"psi_A", "field_T", "target_T", "residual_T", "force_N", "torque_Nm"}
_REDUCTIONS = {"sum", "sum_squares", "mean", "norm1", "norm2", "norm_inf", "rms"}
_ELEMENTWISE = {"abs", "square", "sqrt", "exp", "log", "sin", "cos"}
FORMULA_FUNCTIONS = tuple(sorted(_REDUCTIONS | _ELEMENTWISE | {"min", "max"}))
FORMULA_EXAMPLE_OBJECTIVE = "power_W / 10 + sum_squares(residual_T / 0.001)"
FORMULA_EXAMPLE_CONSTRAINTS = "norm_inf(force_N) <= 1\nabs(torque_Nm[2]) <= 0.1"
_BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
_RELATIONS = {ast.LtE: "<=", ast.GtE: ">=", ast.Eq: "=="}
_MAX_LINE = 2048
_MAX_CONSTRAINTS = 64


class NonConvexFormulaError(ValueError):
    """A valid formula is outside CVXPY's disciplined convex programming rules."""


def _number(node):
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if isinstance(node, ast.Constant) and type(node.value) in (float, int):
        number = node.value * sign
        if -float_info.max <= number <= float_info.max:
            return number
    raise ValueError("Expected a finite real number.")


def _expression(node, symbols):
    """Translate only the specified grammar to an immutable expression tree."""
    if isinstance(node, ast.Constant):
        return float(_number(node))
    if isinstance(node, ast.Name):
        if node.id not in FORMULA_SYMBOLS:
            raise ValueError(f"Unknown physical quantity {node.id!r}.")
        symbols.add(node.id)
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _expression(node.operand, symbols)
        return ("neg", value) if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        operation = _BINARY[type(node.op)]
        left = _expression(node.left, symbols)
        if operation == "**":
            exponent = _number(node.right)
            if abs(exponent) > 8 or (2 * exponent) % 1 != 0:
                raise ValueError("Powers require a literal integer or half-integer between -8 and 8.")
            return (operation, left, float(exponent))
        return (operation, left, _expression(node.right, symbols))
    if isinstance(node, ast.Subscript):
        index = _number(node.slice)
        if type(index) is not int or abs(index) > 10**8:
            raise ValueError("Indices must be bounded integer literals.")
        return ("index", _expression(node.value, symbols), index)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORMULA_FUNCTIONS:
        name = node.func.id
        counts = (1, 2) if name in {"max", "min"} else (1,)
        if node.keywords or len(node.args) not in counts:
            raise ValueError(f"Invalid arguments for {name}().")
        return (name, *(_expression(arg, symbols) for arg in node.args))
    raise ValueError("Unsupported syntax. Use physical quantities, numeric literals and supported math functions.")


def _vector_result(expr):
    if isinstance(expr, str):
        return expr in _VECTORS
    if not isinstance(expr, tuple):
        return False
    op, *args = expr
    if op in _REDUCTIONS or op == "index" or (op in {"max", "min"} and len(args) == 1):
        return False
    return any(_vector_result(arg) for arg in args)


def _parsed_line(text, *, constraint, symbols):
    if len(text) > _MAX_LINE or "\n" in text or "\r" in text:
        raise ValueError(f"Use one expression per line, at most {_MAX_LINE} characters.")
    # A caret denotes mathematical power here, with exponentiation precedence.
    source = re.sub(r"(?<![<>=!])=(?!=)", "==", text.replace("^", "**"))
    try:
        root = ast.parse(source, mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid formula syntax.") from exc
    if sum(1 for _ in ast.walk(root)) > 192:
        raise ValueError("Formula is too complex for one line.")
    if constraint:
        if not isinstance(root, ast.Compare) or len(root.ops) != 1 or type(root.ops[0]) not in _RELATIONS:
            raise ValueError("Use one comparison per constraint: <=, >= or ==.")
        return (
            _expression(root.left, symbols),
            _RELATIONS[type(root.ops[0])],
            _expression(root.comparators[0], symbols),
        )
    expr = _expression(root, symbols)
    if _vector_result(expr):
        raise ValueError("Objective must be scalar; reduce a vector with a norm, sum or indexed component.")
    return expr


def _numeric(value):
    array = np.asarray(value)
    if array.ndim > 1 or array.size == 0 or array.dtype.kind not in "fiu":
        raise ValueError("Expected a nonempty finite real scalar or vector.")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("The formula quantity or result exceeds its finite SI range.")
    return array


def _operation(op, args, cp):
    a = args[0]
    if op == "index":
        index = args[1]
        if len(a.shape) != 1 or not -a.shape[0] <= index < a.shape[0]:
            raise ValueError(f"Index {index} is outside operand shape {a.shape}.")
        return a[index]
    if op == "neg":
        return -a
    if len(args) == 2:
        b = args[1]
        if op != "**" and a.shape and b.shape and a.shape != b.shape:
            raise ValueError(f"Vector operations require matching shapes, received {a.shape} and {b.shape}.")
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return cp.multiply(a, b) if cp else np.multiply(a, b)
        if op == "/":
            if cp and not isinstance(b, cp.Expression) and np.any(b == 0):
                raise ValueError("Division by a recorded zero denominator.")
            return a / b
        if op == "**":
            # CVXPY's power atom restricts the base to >= 0 unless the
            # exponent is zero, one or a positive power of two.
            if cp and b > 0 and b % 2 == 0 and b not in {2, 4, 8}:
                return cp.power(cp.abs(a), b)
            if cp and b != 1 and b.is_integer() and (b < 0 or b % 2) and not a.is_nonneg():
                raise NonConvexFormulaError(
                    "This power of a signed quantity is outside disciplined convex programming (DCP)."
                )
            return cp.power(a, b) if cp else np.power(a, b)
        if op == "min":
            return (cp.minimum if cp else np.minimum)(a, b)
        return (cp.maximum if cp else np.maximum)(a, b)
    if op in {"norm1", "norm2", "norm_inf", "rms"}:
        p = {"norm1": 1, "norm2": 2, "norm_inf": np.inf, "rms": 2}[op]
        if not cp and a.shape and op in {"norm2", "rms"}:
            # Scaling keeps representable norms/RMS finite even when a*a overflows.
            scale = np.max(np.abs(a))
            if scale == 0:
                return 0.0
            scaled_norm = np.linalg.norm(a / scale)
            return scale * (scaled_norm / math.sqrt(a.size) if op == "rms" else scaled_norm)
        if a.shape:
            value = cp.norm(a, p) if cp else np.linalg.norm(a, ord=p)
        else:
            value = cp.abs(a) if cp else np.abs(a)
        return value / math.sqrt(a.size) if op == "rms" else value
    if op == "sum_squares":
        return cp.sum_squares(a) if cp else np.sum(np.square(a))
    if op in {"sin", "cos"} and cp:
        raise NonConvexFormulaError(f"{op}() on a variable is outside disciplined convex programming (DCP).")
    return getattr(cp or np, op)(a)


def _bind(expr, bindings, cp):
    if isinstance(expr, str):
        return bindings[expr]
    if not isinstance(expr, tuple):
        return np.float64(expr)
    op, *nodes = expr
    if op in {"index", "**"}:
        args = [_bind(nodes[0], bindings, cp), nodes[1]]
    else:
        args = [_bind(arg, bindings, cp) for arg in nodes]
    symbolic = cp and any(isinstance(arg, cp.Expression) for arg in args)
    value = _operation(op, args, cp if symbolic else None)
    return value if symbolic else _numeric(value)


@dataclass(frozen=True, slots=True)
class FormulaProgram:
    objective_text: str
    constraint_lines: tuple[str, ...]
    symbols: frozenset[str]
    _objective: object = field(repr=False)
    _relations: tuple = field(repr=False)

    def compile(self, bindings: Mapping):
        """Return a convex scalar CVXPY expression (or None) and DCP constraints."""
        import cvxpy as cp

        return self._run(bindings, cp)

    def evaluate(self, bindings: Mapping):
        """Return a finite scalar (or None) and (lhs, relation, rhs) numerical triples."""
        return self._run(bindings, None)

    def _run(self, bindings, cp):
        quantities = {}
        for name in self.symbols:
            if name not in bindings or bindings[name] is None:
                raise ValueError(f"Formula quantity {name!r} is unavailable.")
            value = bindings[name]
            if cp and isinstance(value, cp.Constant):
                value = value.value
            if not cp or not isinstance(value, cp.Expression):
                value = _numeric(value)
            if len(value.shape) != int(name in _VECTORS) or (name in {"force_N", "torque_Nm"} and value.shape != (3,)):
                raise ValueError(f"Invalid shape {value.shape} for formula quantity {name}.")
            quantities[name] = value
        objective, constraints = None, []
        label = "Custom objective"
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                if self._objective is not None:
                    result = _bind(self._objective, quantities, cp)
                    objective = cp.Expression.cast_to_const(result) if cp else float(result)
                    if cp and not objective.is_convex():
                        raise NonConvexFormulaError("The minimized objective is not convex under DCP rules.")
                for line, (left, op, right) in zip(self.constraint_lines, self._relations):
                    label = f"Custom constraint {line!r}"
                    lhs, rhs = _bind(left, quantities, cp), _bind(right, quantities, cp)
                    if lhs.shape and rhs.shape and lhs.shape != rhs.shape:
                        raise ValueError(
                            f"Vector comparisons require matching shapes, received {lhs.shape} and {rhs.shape}."
                        )
                    if not cp:
                        lhs, rhs = np.broadcast_arrays(lhs, rhs)
                        constraints.append((lhs, op, rhs))
                    else:
                        relation = {"<=": operator.le, ">=": operator.ge, "==": operator.eq}[op](
                            cp.Expression.cast_to_const(lhs), rhs
                        )
                        if not relation.is_dcp():
                            raise NonConvexFormulaError(
                                "The relation is not a DCP constraint (equalities must be affine)."
                            )
                        constraints.append(relation)
        except NonConvexFormulaError as exc:
            raise NonConvexFormulaError(f"{label}: {exc}") from exc
        except (ValueError, IndexError, TypeError, ArithmeticError) as exc:
            raise ValueError(f"{label}: {exc}") from exc
        return objective, constraints


def parse_formulas(objective: str = "", constraints: str | Sequence[str] = ()) -> FormulaProgram:
    """Validate bounded formula syntax without importing a solver or executing Python."""
    if not isinstance(objective, str) or len(objective) > _MAX_LINE:
        raise ValueError("Expected one objective formula of at most 2048 characters.")
    if isinstance(constraints, str):
        if len(constraints) > _MAX_LINE * _MAX_CONSTRAINTS:
            raise ValueError("Constraint text exceeds the formula size limit.")
        constraints = constraints.splitlines()
    if not isinstance(constraints, Sequence) or len(constraints) > _MAX_CONSTRAINTS:
        raise ValueError(f"Provide at most {_MAX_CONSTRAINTS} constraint lines.")
    symbols, lines, relations = set(), [], []
    objective = objective.strip()
    try:
        expr = _parsed_line(objective, constraint=False, symbols=symbols) if objective else None
    except ValueError as exc:
        raise ValueError(f"Custom objective: {exc}") from exc
    for line_no, line in enumerate(constraints, start=1):
        if not isinstance(line, str):
            raise ValueError(f"Constraint line {line_no} must be text.")
        if len(line) > _MAX_LINE:
            raise ValueError(f"Constraint line {line_no} exceeds {_MAX_LINE} characters.")
        if line.strip():
            try:
                relations.append(_parsed_line(line.strip(), constraint=True, symbols=symbols))
            except ValueError as exc:
                raise ValueError(f"Constraint line {line_no}: {exc}") from exc
            lines.append(line.strip())
    return FormulaProgram(objective, tuple(lines), frozenset(symbols), expr, tuple(relations))
