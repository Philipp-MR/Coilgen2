"""Finite-family contour-order benchmark; no CoilGen production imports.

Run with any Python environment containing NumPy. Lengths are in millimetres.
All contours retain their CCW orientation; their order alone is permuted.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


RADII_MM = np.array([12., 18., 24., 30., 36., 42., 48.])
CUT_HALF_ANGLE_RAD = 0.16
HANDLE_FRACTION = 0.35


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def bezier_controls(source: int, target: int) -> np.ndarray:
    a = CUT_HALF_ANGLE_RAD
    end = RADII_MM[source] * np.array([np.cos(a), -np.sin(a)])
    start = RADII_MM[target] * np.array([np.cos(a), np.sin(a)])
    length = HANDLE_FRACTION * np.linalg.norm(start - end)
    # Directed CCW tangents at the retained arc's end and start, respectively.
    t_end = np.array([np.sin(a), np.cos(a)])
    t_start = np.array([-np.sin(a), np.cos(a)])
    return np.array([end, end + length * t_end,
                     start - length * t_start, start])


def bezier_points(control: np.ndarray, samples: int) -> np.ndarray:
    t = np.linspace(0., 1., samples)[:, None]
    return ((1-t)**3 * control[0] + 3*(1-t)**2*t * control[1]
            + 3*(1-t)*t**2 * control[2] + t**3 * control[3])


def bezier_length(control: np.ndarray) -> float:
    # Fixed high-order quadrature, independent of the plotted polyline sampling.
    x, weight = np.polynomial.legendre.leggauss(96)
    t = ((x + 1.) * 0.5)[:, None]
    derivative = (3*(1-t)**2 * (control[1] - control[0])
                  + 6*(1-t)*t * (control[2] - control[1])
                  + 3*t**2 * (control[3] - control[2]))
    return float(np.dot(weight * 0.5, np.linalg.norm(derivative, axis=1)))


def intersections(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Unique transverse intersections of two polylines, including endpoints.

    Parallel/collinear segments are not counted as isolated intersections.
    This benchmark has no overlapping collinear curve segments. Shared intended
    connector/arc joins are removed separately, by coordinates and loop identity.
    """
    p, r = first[:-1], np.diff(first, axis=0)
    q, s = second[:-1], np.diff(second, axis=0)
    pmin, pmax = np.minimum(first[:-1], first[1:]), np.maximum(first[:-1], first[1:])
    qmin, qmax = np.minimum(second[:-1], second[1:]), np.maximum(second[:-1], second[1:])
    broad = ((pmin[:, None, :] <= qmax[None, :, :] + 1e-10)
             & (qmin[None, :, :] <= pmax[:, None, :] + 1e-10)).all(axis=2)
    i, j = np.nonzero(broad)
    if not len(i):
        return np.empty((0, 2))
    denominator = cross2(r[i], s[j])
    good = np.abs(denominator) > 1e-14
    i, j, denominator = i[good], j[good], denominator[good]
    difference = q[j] - p[i]
    along_first = cross2(difference, s[j]) / denominator
    along_second = cross2(difference, r[i]) / denominator
    hit = ((along_first >= -1e-9) & (along_first <= 1+1e-9)
           & (along_second >= -1e-9) & (along_second <= 1+1e-9))
    points = p[i[hit]] + along_first[hit, None] * r[i[hit]]
    if not len(points):
        return np.empty((0, 2))
    # One crossing landing at a sample vertex otherwise appears twice.
    _, unique = np.unique(np.round(points, decimals=7), axis=0, return_index=True)
    return points[np.sort(unique)]


def arc_points(loop: int, samples: int) -> np.ndarray:
    theta = np.linspace(CUT_HALF_ANGLE_RAD, 2*np.pi-CUT_HALF_ANGLE_RAD, samples)
    return RADII_MM[loop] * np.column_stack((np.cos(theta), np.sin(theta)))


def curve_cache(samples: int) -> tuple[dict, list]:
    edges = [(i, j) for i in range(1, 7) for j in range(6) if i != j]
    bridges = {edge: bezier_points(bezier_controls(*edge), samples) for edge in edges}
    arcs = [arc_points(i, samples*3) for i in range(7)]
    return bridges, arcs


def get_arc_hits(edge: tuple, bridge: np.ndarray, arcs: list) -> list:
    hits = []
    for loop, arc in enumerate(arcs):
        points = intersections(bridge, arc)
        for point in points:
            if loop == edge[0] and np.linalg.norm(point - bridge[0]) < 1e-7:
                continue
            if loop == edge[1] and np.linalg.norm(point - bridge[-1]) < 1e-7:
                continue
            hits.append({"loop_radius_mm": float(RADII_MM[loop]),
                         "point_mm": point.tolist()})
    return hits


def evaluate_orders(samples: int = 161) -> tuple[list, dict, list]:
    bridges, arcs = curve_cache(samples)
    lengths = {edge: bezier_length(bezier_controls(*edge)) for edge in bridges}
    arc_hit_cache = {edge: get_arc_hits(edge, bridge, arcs) for edge, bridge in bridges.items()}
    pair_hit_cache = {}
    results = []
    for interior in itertools.permutations(range(1, 6)):
        order = (6, *interior, 0)
        edges = list(zip(order[:-1], order[1:]))
        bridge_hits = []
        for first, second in itertools.combinations(edges, 2):
            key = tuple(sorted((first, second)))
            if key not in pair_hit_cache:
                pair_hit_cache[key] = intersections(bridges[first], bridges[second]).tolist()
            for point in pair_hit_cache[key]:
                bridge_hits.append({"edge_indices": [edges.index(first), edges.index(second)],
                                    "point_mm": point})
        arc_hits = [{"edge_index": index, **hit}
                    for index, edge in enumerate(edges) for hit in arc_hit_cache[edge]]
        results.append({
            "order_indices": list(order),
            "order_radii_mm": RADII_MM[list(order)].tolist(),
            "connector_length_mm": sum(lengths[edge] for edge in edges),
            "radial_travel_mm": float(sum(abs(RADII_MM[a] - RADII_MM[b]) for a, b in edges)),
            "connector_connector_crossings": len(bridge_hits),
            "connector_arc_crossings": len(arc_hits),
            "total_crossings": len(bridge_hits) + len(arc_hits),
            "radial_order_reversals": int(np.count_nonzero(np.diff(np.sign(np.diff(RADII_MM[list(order)]))))),
            "connector_connector_hits": bridge_hits,
            "connector_arc_hits": arc_hits,
        })
    return results, bridges, arcs


def route_data(row: dict, samples: int) -> dict:
    order = row["order_indices"]
    return {"arcs_mm": [arc_points(i, samples*3).tolist() for i in order],
            "bridges_mm": [bezier_points(bezier_controls(a, b), samples).tolist()
                           for a, b in zip(order[:-1], order[1:])],
            "bridge_control_points_mm": [bezier_controls(a, b).tolist()
                                         for a, b in zip(order[:-1], order[1:])]}


def main() -> None:
    rows, _, _ = evaluate_orders(161)
    by_length = sorted(rows, key=lambda row: row["connector_length_mm"])
    feasible = [row for row in by_length if row["total_crossings"] == 0]
    best = feasible[0]
    # A declared, fixed scrambled order, not selected to maximize its penalty.
    scrambled_order = [6, 1, 5, 2, 4, 3, 0]
    scrambled = next(row for row in rows if row["order_indices"] == scrambled_order)

    # Repeat crossing detection at twice the resolution for every finite-family
    # route. This checks numerical crossing classification, not physical clearance.
    refined_rows, _, _ = evaluate_orders(321)
    assert [row["total_crossings"] for row in rows] == [row["total_crossings"] for row in refined_rows]
    assert best["order_radii_mm"] == [48., 42., 36., 30., 24., 18., 12.]
    assert by_length[0] == best
    assert len(rows) == 120

    payload = {
        "units": "mm",
        "parameters": {"radii_mm": RADII_MM.tolist(), "cut_half_angle_rad": CUT_HALF_ANGLE_RAD,
                       "handle_fraction": HANDLE_FRACTION, "outer_start_inner_finish": True,
                       "contour_orientation": "CCW viewed from +z", "orders_enumerated": len(rows)},
        "scope": "Minimum connector length within the 120 fixed-port, fixed-Bezier, CCW-preserving open routes; not a global winding optimum.",
        "zero_crossing_orders": len(feasible),
        "crossing_count_refinement_check": "All 120 counts agree for 161 and 321 bridge samples, with 3x samples per retained arc.",
        "retained_arc_length_mm": float(RADII_MM.sum() * (2*np.pi - 2*CUT_HALF_ANGLE_RAD)),
        "best": {**best, **route_data(best, 161)},
        "scrambled": {**scrambled, **route_data(scrambled, 161)},
        "ranking_by_connector_length": by_length,
        "cautions": [
            "Zero centerline crossings does not verify conductor width, clearance or bend radius.",
            "Open routes have free terminals; closing the circuit on the same surface is a separate topology problem.",
            "A shortest connector in this family is not an optimized free-form connector.",
            "No field-error or manufacturing claim follows from length alone.",
        ],
    }
    destination = Path(__file__).with_name("benchmark_results.json")
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, row in (("best", best), ("scrambled", scrambled)):
        print(name, json.dumps({k: v for k, v in row.items() if not k.endswith("hits")}))
    print("zero_crossing_orders", len(feasible), "of", len(rows))
    print("refinement_check", "all crossing counts agree")


if __name__ == "__main__":
    main()
