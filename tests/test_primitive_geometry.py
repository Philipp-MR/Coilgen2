from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from coilgen import (
    AnnularDiscCoilDesign,
    CircularDiscCoilDesign,
    CoilGenConfig,
    ConicalFrustumCoilDesign,
    EllipsoidalCapCoilDesign,
    EllipticalCylinderCoilDesign,
    ImportedMeshCoilDesign,
    ParametricSurfaceCoilDesign,
    PartialCylinderCoilDesign,
    RuledSurfaceCoilDesign,
    ShieldedCoilDesign,
    SphericalCapCoilDesign,
    SuperellipseCylinderCoilDesign,
    SurfaceOfRevolutionCoilDesign,
    ToroidalSegmentCoilDesign,
    validate_config,
)
from coilgen.manufacturing import save_mesh_as_stl
from coilgen.models import Mesh
from coilgen.geometry import (
    available_geometry_factories,
    build_elliptical_cylinder_mesh,
    build_partial_cylinder_mesh,
    build_superellipse_cylinder_mesh,
    load_mesh_from_config,
    offset_mesh_along_vertex_normals,
)
from coilgen.surface_topology import mesh_quality_report, mesh_surface_diagnostics


class PrimitiveGeometryTest(unittest.TestCase):
    def test_imported_mesh_length_scale_converts_source_coordinates_to_metres(self) -> None:
        source_mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 50.0, 0.0]]),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        with TemporaryDirectory() as temporary_directory:
            source_root = Path(temporary_directory)
            save_mesh_as_stl(source_mesh, source_root / "surface_mm.stl")
            design = ImportedMeshCoilDesign(
                mesh_file="surface_mm.stl",
                geometry_source_path=source_root,
                mesh_length_scale_to_m=1.0e-3,
            )

            config = design.to_config()
            loaded = load_mesh_from_config(config, project_root=source_root)

        np.testing.assert_allclose(np.ptp(loaded.vertices, axis=0), (0.1, 0.05, 0.0), atol=1.0e-12)
        self.assertEqual(loaded.metadata["source_length_scale_to_m"], 1.0e-3)

    def test_new_primitive_factories_are_registered(self) -> None:
        names = {factory.name for factory in available_geometry_factories()}

        self.assertIn("create annular disc mesh", names)
        self.assertIn("create partial cylinder mesh", names)
        self.assertIn("create elliptical cylinder mesh", names)
        self.assertIn("create superellipse cylinder mesh", names)
        self.assertIn("create spherical cap mesh", names)
        self.assertIn("create ellipsoidal cap mesh", names)
        self.assertIn("create conical frustum mesh", names)
        self.assertIn("create toroidal segment mesh", names)
        self.assertIn("create ruled surface mesh", names)
        self.assertIn("create surface of revolution mesh", names)

    def test_new_primitive_configs_validate_and_build_meshes(self) -> None:
        configs = (
            CoilGenConfig(
                coil_mesh_file="create circular mesh", circular_mesh_parameter_list=(0.2, 8, 1, 0, 0, 0, 0, 0, 0)
            ),
            CoilGenConfig(
                coil_mesh_file="create annular disc mesh",
                annular_disc_mesh_parameter_list=(0.05, 0.2, 3, 12, 0, 0, 1, 0, 0, 0),
            ),
            CoilGenConfig(
                coil_mesh_file="create partial cylinder mesh",
                partial_cylinder_mesh_parameter_list=(0.3, 0.2, -1.0, 1.0, 8, 4, 1, 0, 0, 0, 0, 0, 0),
            ),
            CoilGenConfig(
                coil_mesh_file="create elliptical cylinder mesh",
                elliptical_cylinder_mesh_parameter_list=(0.3, 0.25, 0.16, 10, 4, 1, 0, 0, 0, 0, 0, 0),
            ),
            CoilGenConfig(
                coil_mesh_file="create superellipse cylinder mesh",
                superellipse_cylinder_mesh_parameter_list=(0.3, 0.25, 0.16, 4.0, 12, 4, 1, 0, 0, 0, 0, 0, 0),
            ),
            CoilGenConfig(
                coil_mesh_file="create spherical cap mesh",
                spherical_cap_mesh_parameter_list=(0.2, 1.2, 12, 4, 1, 0, 0, 0, 0, 0, 0),
            ),
            CoilGenConfig(
                coil_mesh_file="create ellipsoidal cap mesh",
                ellipsoidal_cap_mesh_parameter_list=(0.22, 0.18, 0.16, 1.2, 12, 4, 1, 0, 0, 0, 0, 0, 0),
            ),
            CoilGenConfig(
                coil_mesh_file="create conical frustum mesh",
                conical_frustum_mesh_parameter_list=(0.3, 0.2, 0.12, 12, 4, 1, 0, 0, 0, 0, 0, 0),
            ),
            CoilGenConfig(
                coil_mesh_file="create toroidal segment mesh",
                toroidal_segment_mesh_parameter_list=(
                    0.3,
                    0.06,
                    -1.0,
                    1.0,
                    0.0,
                    2.0 * np.pi,
                    8,
                    6,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
            CoilGenConfig(
                coil_mesh_file="create ruled surface mesh",
                ruled_surface_mesh_parameter_list=(
                    2,
                    3,
                    -0.2,
                    -0.1,
                    0.0,
                    0.2,
                    -0.1,
                    0.0,
                    -0.2,
                    0.1,
                    0.0,
                    0.2,
                    0.1,
                    0.0,
                ),
            ),
            CoilGenConfig(
                coil_mesh_file="create surface of revolution mesh",
                surface_of_revolution_mesh_parameter_list=(10, -0.2, 0.15, 0.0, 0.22, 0.2, 0.15),
            ),
        )

        for config in configs:
            with self.subTest(config.coil_mesh_file):
                report = validate_config(config)
                self.assertTrue(report.ok, report.format())
                mesh = load_mesh_from_config(config, project_root=".")
                self.assertGreater(mesh.n_vertices, 0)
                self.assertGreater(mesh.n_faces, 0)
                self.assertTrue(np.all(np.isfinite(mesh.vertices)))

    def test_scientist_design_classes_build_valid_configs(self) -> None:
        designs = (
            CircularDiscCoilDesign(radius_m=0.2, radial_divisions=8),
            AnnularDiscCoilDesign(inner_radius_m=0.05, outer_radius_m=0.2, radial_divisions=3, circular_divisions=12),
            PartialCylinderCoilDesign(radius_m=0.2, length_m=0.3, angular_divisions=8, length_divisions=4),
            EllipticalCylinderCoilDesign(
                radius_x_m=0.24, radius_y_m=0.16, length_m=0.3, angular_divisions=10, length_divisions=4
            ),
            SuperellipseCylinderCoilDesign(
                radius_x_m=0.24, radius_y_m=0.16, length_m=0.3, exponent=4.0, angular_divisions=12, length_divisions=4
            ),
            SphericalCapCoilDesign(radius_m=0.2, cap_polar_angle_rad=1.2, azimuthal_divisions=12, polar_divisions=4),
            EllipsoidalCapCoilDesign(
                radius_x_m=0.22,
                radius_y_m=0.18,
                radius_z_m=0.16,
                cap_polar_angle_rad=1.2,
                azimuthal_divisions=12,
                polar_divisions=4,
            ),
            ConicalFrustumCoilDesign(
                radius_bottom_m=0.2, radius_top_m=0.12, length_m=0.3, angular_divisions=12, length_divisions=4
            ),
            ToroidalSegmentCoilDesign(
                major_radius_m=0.3,
                minor_radius_m=0.06,
                start_angle_rad=-1.0,
                end_angle_rad=1.0,
                major_divisions=8,
                minor_divisions=6,
            ),
            RuledSurfaceCoilDesign(
                curve_a_m=((-0.2, -0.1, 0.0), (0.2, -0.1, 0.0)),
                curve_b_m=((-0.2, 0.1, 0.0), (0.2, 0.1, 0.0)),
                span_divisions=3,
            ),
            ParametricSurfaceCoilDesign(
                surface=lambda u, v: (0.2 * (u - 0.5), 0.2 * (v - 0.5), 0.02 * np.sin(np.pi * u) * np.sin(np.pi * v)),
                u_divisions=4,
                v_divisions=4,
            ),
            SurfaceOfRevolutionCoilDesign(
                circular_divisions=10, z_radius_pairs_m=((-0.2, 0.15), (0.0, 0.22), (0.2, 0.15))
            ),
        )

        for design in designs:
            with self.subTest(type(design).__name__):
                report = validate_config(design.to_config())
                self.assertTrue(report.ok, report.format())

    def test_cylindrical_primitive_face_orientation_is_outward(self) -> None:
        partial = build_partial_cylinder_mesh(0.3, 0.2, -0.5, 0.5, 8, 4, 1, 0, 0, 0, 0, 0, 0)
        elliptical = build_elliptical_cylinder_mesh(0.3, 0.25, 0.16, 10, 4, 1, 0, 0, 0, 0, 0, 0)

        for mesh in (partial, elliptical):
            face = mesh.faces[0]
            vertices = mesh.vertices[face]
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            centroid = vertices.mean(axis=0)
            self.assertGreater(float(np.dot(normal[:2], centroid[:2])), 0.0)

    def test_offset_shield_preserves_primary_topology_and_separation(self) -> None:
        primary = build_superellipse_cylinder_mesh(0.3, 0.25, 0.16, 4.0, 16, 4, 1, 0, 0, 0, 0, 0, 0)

        shield = offset_mesh_along_vertex_normals(primary, 0.02)

        self.assertEqual(shield.vertices.shape, primary.vertices.shape)
        np.testing.assert_array_equal(shield.faces, primary.faces)
        distances = np.linalg.norm(shield.vertices - primary.vertices, axis=1)
        np.testing.assert_allclose(distances, 0.02, rtol=1e-9, atol=1e-12)
        self.assertEqual(shield.metadata["role"], "shield")
        self.assertTrue(shield.metadata["same_topology_as_primary"])

    def test_generic_shielded_design_builds_same_topology_offset_pair(self) -> None:
        primary = AnnularDiscCoilDesign(
            name="annular_primary_for_wrapper",
            inner_radius_m=0.05,
            outer_radius_m=0.2,
            radial_divisions=3,
            circular_divisions=12,
        )
        design = ShieldedCoilDesign(
            name="shielded_annular_wrapper",
            primary_design=primary,
            shield_separation_m=0.02,
            regularization_method="fixed",
            regularization_factor=10.0,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance=True,
        )

        config = design.to_config()
        report = validate_config(config)
        self.assertTrue(report.ok, report.format())
        mesh = load_mesh_from_config(config, project_root=".")

        primary_mesh = load_mesh_from_config(primary.to_config(), project_root=".")
        self.assertEqual(mesh.n_vertices, 2 * primary_mesh.n_vertices)
        self.assertEqual(mesh.n_faces, 2 * primary_mesh.n_faces)
        self.assertEqual(mesh.metadata["geometry_type"], "generic_shielded_offset_pair")
        self.assertTrue(mesh.metadata["same_topology_primary_shield"])

    def test_mesh_quality_report_includes_shape_and_boundary_metrics(self) -> None:
        mesh = build_partial_cylinder_mesh(0.3, 0.2, -0.5, 0.5, 8, 4, 1, 0, 0, 0, 0, 0, 0)

        diagnostics = mesh_surface_diagnostics(mesh)
        report = mesh_quality_report(mesh)

        self.assertGreater(diagnostics["edge_length_mean_m"], 0.0)
        self.assertGreaterEqual(diagnostics["boundary_loop_count"], 1)
        self.assertGreater(diagnostics["triangle_edge_ratio_max"], 0.0)
        self.assertTrue(report.ok)
        self.assertEqual(report.as_dict()["mesh_quality_error_count"], 0)


if __name__ == "__main__":
    unittest.main()
