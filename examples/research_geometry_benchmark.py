"""Reproduce cold compilation and warm float64 geometry kernel measurements.

Run with the main project's Python and the accelerate extra. No fastmath or parallel reductions are used.
Each invocation starts with an empty temporary Numba cache to separate actual
compilation latency from warm kernel runtime. Timing thresholds are not tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
_CACHE = tempfile.TemporaryDirectory(prefix="coilgen_numba_benchmark_")
os.environ["NUMBA_CACHE_DIR"] = _CACHE.name

# Imports follow thread/cache setup so cold timings include compilation.
import numpy as np  # noqa: E402
import numba  # noqa: E402
from coilgen import CoilGenConfig, CoilPart, ContourLine, Mesh  # noqa: E402
from coilgen.manufacturing import create_swept_contour_body  # noqa: E402
from coilgen.research.kernels import geometry_acceleration, segment_pair_distances, triangle_pair_distances  # noqa: E402


def _seconds(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return time.perf_counter() - start, result


def main():
    rng = np.random.default_rng(7102026)
    rows = []
    for name, kernel, size, vertices in (
        ("segment", segment_pair_distances, 20000, 2),
        ("triangle", triangle_pair_distances, 2000, 3),
    ):
        first, second = rng.normal(scale=0.01, size=(2, size, vertices, 3))
        cold, _ = _seconds(kernel, first[:1], second[:1], backend="numba")
        reference_s, expected = _seconds(kernel, first, second, backend="reference")
        warm_s, actual = _seconds(kernel, first, second, backend="numba")
        np.testing.assert_allclose(actual.values_m, expected.values_m, rtol=2e-12, atol=1e-15)
        rows.append(
            {
                "primitive": name,
                "pair_count": size,
                "cold_compile_and_single_pair_s": cold,
                "reference_batch_s": reference_s,
                "warm_compiled_batch_s": warm_s,
                "warm_speedup": reference_s / warm_s,
                "maximum_absolute_distance_difference_m": float(np.max(np.abs(actual.values_m - expected.values_m))),
            }
        )
    support = Mesh(
        vertices=np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]), faces=np.array([[0, 1, 2], [0, 2, 3]])
    )
    angle = np.linspace(0, 2 * np.pi, 129)
    points = np.column_stack((0.04 * np.cos(angle), 0.03 * np.sin(angle), np.zeros_like(angle)))
    points[-1] = points[0]
    contour = ContourLine(v=points, potential=1.0)
    part = CoilPart(mesh=support, contour_lines=[contour], contour_step=2.0)
    config = CoilGenConfig(conductor_cross_section_width=0.001, conductor_cross_section_height=0.001)
    sweeps = {}
    bodies = {}
    for backend in ("reference", "numba"):
        with geometry_acceleration(backend):
            elapsed, body = _seconds(create_swept_contour_body, part, contour, config, part_index=1, contour_index=1)
        sweeps[backend] = elapsed
        bodies[backend] = body
    np.testing.assert_array_equal(bodies["numba"].mesh.vertices, bodies["reference"].mesh.vertices)
    np.testing.assert_array_equal(bodies["numba"].mesh.faces, bodies["reference"].mesh.faces)
    for key in ("manufacturing_ok", "sweep_mesh_self_intersection_count", "sweep_mesh_collision_candidate_pair_count"):
        assert bodies["numba"].diagnostics[key] == bodies["reference"].diagnostics[key]
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "float_type": "float64",
        "fastmath": False,
        "native_threads": 1,
        "fresh_numba_cache": True,
        "kernels": rows,
        "complete_sweep": {
            "source_points": len(points),
            "faces": bodies["numba"].mesh.n_faces,
            "seconds": sweeps,
            "speedup": sweeps["reference"] / sweeps["numba"],
            "collision_candidate_pairs": bodies["numba"].diagnostics["sweep_mesh_collision_candidate_pair_count"],
            "mesh_and_selected_diagnostics_equal": True,
        },
        "limitations": "Synthetic distance batches; full sweep also spends time in uncompiled projection, frames, BVH and quality checks. Hardware-dependent timings.",
    }
    output = ROOT / "tmp" / "research_fabrication"
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
