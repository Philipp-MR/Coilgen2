"""Editable concentric cylindrical supports, captured as portable SI meshes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral, Real

import numpy as np

from ..geometry import build_cylinder_mesh, combine_meshes, load_mesh_from_config, load_secondary_target_mesh
from ..geometry import normalize_mesh, rotation_matrix_by_vector, split_disconnected_mesh
from ..models import Mesh
from .project import ProjectDocument


GEOMETRY_SCHEMA = "coilgen-cylindrical-geometry-v1"


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive in metres.")
    return float(value)


def _divisions(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")
    return int(value)


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite world-coordinate values.")
    return result


def _axis_rotation(axis):
    axis = np.asarray(axis)
    cross = np.cross((0, 0, 1), axis)
    angle = math.acos(float(np.clip(axis[2], -1, 1)))
    return rotation_matrix_by_vector(cross if np.linalg.norm(cross) > 1e-15 else np.array((1, 0, 0)), angle)


@dataclass(frozen=True, slots=True)
class CylinderSurface:
    radius_m: float
    length_m: float
    angular_divisions: int = 32
    axial_divisions: int = 16
    center_m: tuple[float, float, float] = (0, 0, 0)
    axis: tuple[float, float, float] = (0, 0, 1)
    azimuth_rad: float = 0.0

    def __post_init__(self):
        for name in ("radius_m", "length_m"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(self, "angular_divisions", _divisions(self.angular_divisions, "Angular divisions", 4, 512))
        object.__setattr__(self, "axial_divisions", _divisions(self.axial_divisions, "Axial divisions", 1, 256))
        if self.angular_divisions * (self.axial_divisions + 1) > 100_000:
            raise ValueError("A generated cylinder is limited to 100,000 vertices; refine explicitly in a study.")
        object.__setattr__(self, "center_m", tuple(_vector(self.center_m, "Center")))
        axis = _vector(self.axis, "Cylinder axis")
        norm = np.linalg.norm(axis)
        if norm < 1e-15 or not np.isfinite(norm):
            raise ValueError("Cylinder axis must be nonzero.")
        object.__setattr__(self, "axis", tuple(axis / norm))
        if (
            isinstance(self.azimuth_rad, bool)
            or not isinstance(self.azimuth_rad, Real)
            or not math.isfinite(self.azimuth_rad)
        ):
            raise ValueError("Cylinder azimuth must be a finite angle in radians.")

    def mesh(self, *, closed=False, role="primary") -> Mesh:
        """Preserve the native cylinder's rings while enforcing the stated span."""
        n, axial = self.angular_divisions, self.axial_divisions
        mesh = build_cylinder_mesh(self.length_m * (axial + 1) / axial, self.radius_m, n, axial, 1, 0, 0, 0)
        if closed:
            bottom, top = len(mesh.vertices), len(mesh.vertices) + 1
            vertices = np.vstack((mesh.vertices, (0, 0, -self.length_m / 2), (0, 0, self.length_m / 2)))
            last = len(mesh.vertices) - n
            caps = []
            for i in range(n):
                caps.extend(((bottom, i, (i + 1) % n), (top, last + (i + 1) % n, last + i)))
            mesh = Mesh(vertices, np.vstack((mesh.faces, np.asarray(caps, dtype=np.int64))))
        rotation = _axis_rotation(self.axis) @ rotation_matrix_by_vector(np.array((0, 0, 1)), self.azimuth_rad)
        mesh.vertices = mesh.vertices @ rotation.T + np.asarray(self.center_m)
        mesh.metadata = {"geometry_type": "closed_cylinder" if closed else "cylinder", "units": "SI", "role": role}
        return normalize_mesh(mesh)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Cylinder surface specification must be a mapping.")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class CylindricalGeometry:
    primary: CylinderSurface
    shield: CylinderSurface | None = None
    suppression: CylinderSurface | None = None

    def __post_init__(self):
        if not isinstance(self.primary, CylinderSurface):
            raise TypeError("Primary support must be a CylinderSurface.")
        supports = [self.primary]
        for role in ("shield", "suppression"):
            surface = getattr(self, role)
            if surface is None:
                continue
            if not isinstance(surface, CylinderSurface):
                raise TypeError(f"{role.capitalize()} must be a CylinderSurface or None.")
            if not np.allclose(surface.center_m, self.primary.center_m, rtol=0, atol=1e-12) or not np.allclose(
                surface.axis, self.primary.axis, rtol=0, atol=1e-12
            ):
                raise ValueError("Primary, shield and suppression cylinders must share their world center and axis.")
        if self.shield is not None:
            if self.shield.radius_m <= self.primary.radius_m:
                raise ValueError("Shield radius must exceed primary radius.")
            supports.append(self.shield)
        if self.suppression is not None:
            # The finite polygonal enclosure, not its circumscribed circle, must
            # enclose every support. End caps must remain outside the supports.
            inside_radius = self.suppression.radius_m * math.cos(math.pi / self.suppression.angular_divisions)
            if inside_radius <= max(surface.radius_m for surface in supports) or self.suppression.length_m <= max(
                surface.length_m for surface in supports
            ):
                raise ValueError("The closed suppression cylinder must strictly enclose all winding supports.")

    def to_dict(self):
        return {"schema": GEOMETRY_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"schema", "primary", "shield", "suppression"}:
            raise ValueError("Malformed cylindrical geometry specification.")
        if value["schema"] != GEOMETRY_SCHEMA:
            raise ValueError("Unsupported cylindrical geometry schema.")
        return cls(
            **{
                key: CylinderSurface(**value[key]) if value[key] is not None else None
                for key in ("primary", "shield", "suppression")
            }
        )

    def apply(self, project: ProjectDocument, *, expected_fingerprint=None) -> ProjectDocument:
        """Explicitly replace support/suppression geometry in one revision.

        Target coordinates, amplitudes, physical bounds and saved evidence are
        preserved. A new suppression surface uses vector/area semantics, but
        the engineer must configure its field budget explicitly. ``None``
        deliberately removes the corresponding optional generated surface.
        """
        if expected_fingerprint is not None and expected_fingerprint != project.fingerprint:
            raise ValueError("The project changed after this geometry draft was created.")
        surfaces = [self.primary.mesh()]
        if self.shield is not None:
            surfaces.append(self.shield.mesh(role="active_shield"))
        support = combine_meshes(surfaces)
        support.metadata.update(
            geometry_type="active_shielded_cylinder" if self.shield else "cylinder",
            units="SI",
            part_roles=["primary", "active_shield"] if self.shield else ["primary"],
            cylindrical_geometry=self.to_dict(),
        )
        changes = {"surface_is_cylinder_flag": True}
        if self.shield is not None:
            changes.update(routing_circuit_scope="per_part", sweep_surface_realization="mesh")
        if self.suppression is None:
            changes["secondary_target_mesh_file"] = "none"
        elif project.values["secondary_target_mesh_file"] == "none":
            changes.update(suppression_field_component="vector", suppression_surface_weighting="area")
        updated = project.with_mesh(support, field="coil_mesh_file", changes=changes)
        if self.suppression is not None:
            suppression = self.suppression.mesh(closed=True, role="suppression")
            suppression.metadata["cylindrical_geometry"] = self.to_dict()
            updated = updated.with_mesh(suppression, field="secondary_target_mesh_file")
        # Publish the two owned assets together as one user edit. No intermediate
        # document escapes this operation and the original remains immutable.
        return project if updated.values == project.values else replace(updated, revision=project.revision + 1)

    @classmethod
    def from_project(cls, project: ProjectDocument):
        """Read verified authoring metadata; never infer arbitrary imported CAD.

        Native cylinders and the existing named active-shield template are also
        recognized. Unknown support/suppression meshes return None unchanged.
        """
        config = project.config
        if str(config.coil_mesh_file).strip().lower() == "none":
            return None
        mesh = load_mesh_from_config(config, project.project_root)
        value = mesh.metadata.get("cylindrical_geometry")
        if value is not None:
            spec = cls.from_dict(value)
            expected = combine_meshes([spec.primary.mesh()] + ([spec.shield.mesh()] if spec.shield else []))
            if not _same_mesh(mesh, expected):
                raise ValueError("Cylinder authoring dimensions disagree with the owned support mesh.")
            secondary = load_secondary_target_mesh(config, project.project_root, coil_meshes=[mesh])
            if (
                (secondary is None) != (spec.suppression is None)
                or secondary is not None
                and not _same_mesh(secondary, spec.suppression.mesh(closed=True))
            ):
                raise ValueError("Cylinder authoring dimensions disagree with the owned suppression mesh.")
            return spec
        if config.coil_mesh_file == "create cylinder mesh" and config.secondary_target_mesh_file == "none":
            height, radius, angular, axial, x, y, z, angle = config.cylinder_mesh_parameter_list
            rotation = rotation_matrix_by_vector(np.asarray((x, y, z)), angle)
            axis = rotation @ np.array((0, 0, 1))
            roll = _axis_rotation(axis).T @ rotation
            return cls(
                CylinderSurface(
                    radius,
                    height * axial / (axial + 1),
                    int(angular),
                    int(axial),
                    axis=tuple(axis),
                    azimuth_rad=math.atan2(roll[1, 0], roll[0, 0]),
                )
            )
        if mesh.metadata.get("geometry_type") == "active_shielded_cylinder":
            parts = split_disconnected_mesh(mesh)
            secondary = load_secondary_target_mesh(config, project.project_root, coil_meshes=parts)
            if (
                len(parts) == 2
                and secondary is not None
                and secondary.metadata.get("geometry_type") == ("closed_external_suppression_cylinder")
            ):
                primary, shield = (_template_cylinder(part) for part in parts)
                suppression = _template_cylinder(secondary, closed=True)
                if primary and shield and suppression:
                    return cls(primary, shield, suppression)
        return None


def _same_mesh(left, right):
    return (
        left.vertices.shape == right.vertices.shape
        and np.allclose(left.vertices, right.vertices, rtol=1e-12, atol=1e-14)
        and left.faces.shape == right.faces.shape
        and np.array_equal(left.faces[np.lexsort(left.faces.T[::-1])], right.faces[np.lexsort(right.faces.T[::-1])])
    )


def _template_cylinder(mesh, *, closed=False):
    rings = mesh.vertices[:-2] if closed else mesh.vertices
    levels = np.unique(rings[:, 2])
    if len(levels) < 2 or len(rings) % len(levels):
        return None
    spec = CylinderSurface(
        float(np.max(np.linalg.norm(rings[:, :2], axis=1))),
        float(np.ptp(levels)),
        len(rings) // len(levels),
        len(levels) - 1,
    )
    return spec if _same_mesh(mesh, spec.mesh(closed=closed)) else None
