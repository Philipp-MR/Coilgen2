from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from coilgen import (
    CoilGenConfig,
    CoilGenResult,
    CoilPart,
    ContourLine,
    Mesh,
    TargetField,
    WirePath,
    save_wire_trace_artifacts,
    save_wire_trace_as_csv,
    save_wire_trace_as_step,
    wire_trace_rows,
)


class WireTraceExportTest(unittest.TestCase):
    def test_wire_trace_rows_and_csv_use_routed_centerline_coordinates(self) -> None:
        result = _minimal_result_with_wire_path()

        rows = wire_trace_rows(result)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["part_index"], 1)
        self.assertEqual(rows[0]["point_index"], 1)
        self.assertAlmostEqual(rows[0]["x_m"], 0.0)
        self.assertAlmostEqual(rows[0]["segment_length_from_previous_m"], 0.0)
        self.assertAlmostEqual(rows[1]["segment_length_from_previous_m"], 1.0)
        self.assertAlmostEqual(rows[2]["segment_length_from_previous_m"], 1.0)
        self.assertAlmostEqual(rows[2]["cumulative_length_m"], 2.0)
        self.assertEqual(rows[0]["segment_geometry_from_previous"], "")
        self.assertEqual(rows[1]["segment_geometry_from_previous"], "surface")

        with TemporaryDirectory() as tmp:
            files = save_wire_trace_as_csv(result, tmp, prefix="trace_case")
            path = Path(tmp) / "trace_case_wire_trace.csv"

            self.assertEqual(files, [path])
            with path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(csv_rows), 3)
        self.assertEqual(csv_rows[0]["part_index"], "1")
        self.assertEqual(csv_rows[2]["z_m"], "0.0")
        self.assertEqual(float(csv_rows[2]["cumulative_length_m"]), 2.0)
        self.assertEqual(csv_rows[1]["segment_geometry_from_previous"], "surface")

    def test_xyz_wire_path_csv_is_headerless_ordered_and_split_by_part(self) -> None:
        result = _minimal_result_with_wire_path()
        second_part = CoilPart(mesh=result.coil_parts[0].mesh.copy())
        second_part.wire_path = WirePath(np.array([[0.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=float))
        result.coil_parts.append(second_part)
        result.input_data = CoilGenConfig(
            save_wire_trace_csv=True,
            wire_trace_csv_format="xyz_no_header",
            wire_trace_csv_length_unit="m",
            wire_trace_csv_decimal_places=3,
        )

        with TemporaryDirectory() as tmp:
            files = save_wire_trace_as_csv(result, tmp, prefix="trace_case")
            names = [path.name for path in files]
            first_lines = files[0].read_text(encoding="utf-8").splitlines()
            second_lines = files[1].read_text(encoding="utf-8").splitlines()

        self.assertEqual(names, ["trace_case_wire_path_part001.csv", "trace_case_wire_path_part002.csv"])
        self.assertEqual(first_lines, ["0.000,0.000,0.000", "1.000,0.000,0.000", "1.000,1.000,0.000"])
        self.assertEqual(second_lines, ["0.000,0.000,1.000", "2.000,3.000,4.000"])

    def test_step_wire_path_is_an_unswept_piecewise_linear_wire_body(self) -> None:
        result = _minimal_result_with_wire_path()

        with TemporaryDirectory() as tmp:
            files = save_wire_trace_as_step(result, tmp, prefix="trace_case")
            path = Path(tmp) / "trace_case_wire_path.step"
            step_text = path.read_text(encoding="ascii")

        self.assertEqual(files, [path])
        self.assertIn("ISO-10303-21", step_text)
        self.assertIn(
            "('CoilGen'),('CoilGen'),'CoilGen2 - Created and maintained by Philipp Amrein (MRI hardware engineer)','CoilGen2','');",
            step_text,
        )
        self.assertIn("SHAPE_REPRESENTATION", step_text)
        self.assertIn("GEOMETRIC_CURVE_SET", step_text)
        self.assertEqual(step_text.count("TRIMMED_CURVE("), 2)
        self.assertLess(
            step_text.index("CARTESIAN_POINT('',(0.,0.,0.))"),
            step_text.index("CARTESIAN_POINT('',(1.,0.,0.))"),
        )
        self.assertLess(
            step_text.index("CARTESIAN_POINT('',(1.,0.,0.))"),
            step_text.index("CARTESIAN_POINT('',(1.,1.,0.))"),
        )
        self.assertNotIn("ADVANCED_FACE", step_text)
        self.assertNotIn("MANIFOLD_SOLID_BREP", step_text)
        self.assertNotIn("SHELL", step_text)
        self.assertNotIn("SURFACE", step_text)
        self.assertIn("SI_UNIT($,.METRE.)", step_text)

    def test_enabled_trace_artifacts_include_csv_and_step(self) -> None:
        result = _minimal_result_with_wire_path()
        result.input_data = CoilGenConfig(
            save_wire_trace_csv=True,
            wire_trace_csv_format="xyz_no_header",
            wire_trace_csv_length_unit="m",
        )

        with TemporaryDirectory() as tmp:
            files = save_wire_trace_artifacts(result, tmp, prefix="trace_case")

        self.assertEqual([path.name for path in files], ["trace_case_wire_path.csv", "trace_case_wire_path.step"])

    def test_contour_step_contains_independent_closed_wire_bodies(self) -> None:
        result = _minimal_result_with_wire_path()
        result.coil_parts[0].wire_path = None
        result.coil_parts[0].contour_lines = [
            ContourLine(
                v=np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=float,
                ),
                potential=-1.0,
            ),
            ContourLine(
                v=np.array(
                    [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
                    dtype=float,
                ),
                potential=1.0,
            ),
        ]
        result.input_data = CoilGenConfig(
            save_wire_trace_csv=True,
            wire_trace_csv_format="xyz_no_header",
        )

        with TemporaryDirectory() as tmp:
            files = save_wire_trace_artifacts(result, tmp, prefix="trace_case")
            step_text = files[0].read_text(encoding="ascii")

        self.assertEqual([path.name for path in files], ["trace_case_contour_traces.step"])
        self.assertEqual(step_text.count("TRIMMED_CURVE("), 6)
        self.assertEqual(step_text.count("GEOMETRIC_CURVE_SET("), 2)
        self.assertNotIn("SHELL", step_text)
        self.assertNotIn("SURFACE", step_text)
        self.assertNotIn("MANIFOLD_SOLID_BREP", step_text)


def _minimal_result_with_wire_path() -> CoilGenResult:
    mesh = Mesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    part = CoilPart(mesh=mesh)
    part.wire_path = WirePath(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    )
    target = TargetField(
        coords=np.array([[0.0, 0.0, 0.1]], dtype=float),
        b=np.zeros((1, 3), dtype=float),
        weights=np.ones(1, dtype=float),
        suppressed=np.zeros(1, dtype=bool),
        target_field_group_inds=np.ones(1, dtype=np.int64),
    )
    field = np.zeros((1, 3), dtype=float)
    return CoilGenResult(
        coil_parts=[part],
        num_levels=0,
        potential_step=0.0,
        primary_surface=0,
        target_field=target,
        field_by_layout=field.copy(),
        field_by_unconnected_loops=field.copy(),
        field_layout_per1Amp=field.copy(),
        field_loops_per1Amp=field.copy(),
        needed_current_layout=1.0,
        b_field_opt_sf=field.copy(),
        b_field_opt_sf_1A=field.copy(),
        regularization_factor=1.0,
        regularization_lambda=1.0,
        regularization_diagnostics={},
        target_field_1A=target,
        error_vals={},
        input_data=CoilGenConfig(save_wire_trace_csv=True),
        layout_gradient=None,
        combined_mesh=mesh,
        inductance_matrix=np.zeros((1, 1), dtype=float),
        resistance=np.ones(1, dtype=float),
        coil_lengths=np.ones(1, dtype=float),
        coil_cross_sections=np.ones(1, dtype=float),
    )


if __name__ == "__main__":
    unittest.main()
