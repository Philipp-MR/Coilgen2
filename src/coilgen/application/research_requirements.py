"""Owned regional contracts and finite geometric studies, independent of Qt.

All field bounds use physical SI residuals. Fitting weights never relax a bound.
Support poses retain the original nodal order and move every support together;
positive dimensional factors act about an explicit centre before rigid motion.
sample poses move points and desired Cartesian vectors in the stated world frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from numbers import Real
from pathlib import Path
from typing import Mapping
from types import MappingProxyType

import numpy as np

from .gradient_studies import MeasuredFieldMap, RigidTransform
from ..field_components import field_component_values, normalize_field_component, sensitivity_component_matrix
from ..models import CoilPart, TargetField

SCHEMA = "coilgen-research-requirements-v1"
MAX_SAMPLES = 100_000
MAX_OPERATOR_BYTES = 512 * 1024**2
MAX_REPORT_BYTES = 512 * 1024**2


class FieldSamplingError(ValueError):
    """A zero-thickness sheet evaluation is singular at a supplied sample."""


def _name(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 120:
        raise ValueError("Names must contain 1–120 characters.")
    return value.strip()


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative SI number.")
    return float(value)


def _signed_current(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("Saved currents must be finite signed real amperes.")
    return float(value)


def _strict(value, allowed, name):
    if not isinstance(value, Mapping) or set(value) - set(allowed):
        raise ValueError(f"Malformed {name}: unsupported fields.")
    return dict(value)


@dataclass(frozen=True, slots=True)
class DesiredDataset:
    """A desired target, with owned data and explicit rigid registration.

    The existing real-vector map carrier supplies unit and integrity checks.
    Its ``independent`` flag means declared independent desired samples here,
    rather than a measurement or a numerical guarantee.
    """

    data: MeasuredFieldMap
    registration: RigidTransform = RigidTransform()
    recipe: Mapping | None = None

    def __post_init__(self):
        if not isinstance(self.data, MeasuredFieldMap) or not isinstance(self.registration, RigidTransform):
            raise TypeError("Desired data requires a real field map and proper rigid registration.")
        points, field = self.world_values()
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(field)):
            raise ValueError("Registered desired data exceeds the finite SI range.")
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Desired dataset contains duplicate registered sample coordinates.")
        if self.recipe is not None:
            from ..target import evaluate_field_expression
            from ..target_regions import TargetRegion, sample_target_region

            recipe = _strict(self.recipe, {"region", "expressions", "resolution"}, "analytic desired recipe")
            region = TargetRegion.from_dict(recipe["region"])
            expressions = tuple(recipe["expressions"])
            resolution = recipe["resolution"]
            if len(expressions) != 3 or not all(isinstance(value, str) for value in expressions):
                raise ValueError("Analytic desired recipe needs three Cartesian field expressions.")
            expected_points = sample_target_region(region, resolution, independent=self.data.independent)
            expected_field = np.column_stack(
                [evaluate_field_expression(value, expected_points) for value in expressions]
            )
            if (
                self.data.length_unit != "m"
                or self.data.field_unit != "T"
                or not np.array_equal(expected_points, self.data.coordinates)
                or not np.array_equal(expected_field, self.data.field)
            ):
                raise ValueError("Analytic recipe and owned desired samples disagree; regenerate the dataset.")
            geometry = {
                key: tuple(value) if isinstance(value, list) else value for key, value in region.to_dict().items()
            }
            object.__setattr__(
                self,
                "recipe",
                MappingProxyType(
                    {"region": MappingProxyType(geometry), "expressions": expressions, "resolution": resolution}
                ),
            )

    def world_values(self):
        return self.data.world_values(self.registration)

    @property
    def digest(self):
        points, field = self.world_values()
        return sha256(np.asarray(points, dtype="<f8").tobytes() + np.asarray(field, dtype="<f8").tobytes()).hexdigest()

    def to_dict(self):
        recipe = (
            None
            if self.recipe is None
            else {
                "region": dict(self.recipe["region"]),
                "expressions": list(self.recipe["expressions"]),
                "resolution": self.recipe["resolution"],
            }
        )
        return {"data": self.data.to_dict(), "registration": self.registration.to_dict(), "recipe": recipe}

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        value = _strict(value, {"data", "registration", "recipe"}, "desired dataset")
        return cls(
            MeasuredFieldMap.from_dict(value["data"]),
            RigidTransform(**value.get("registration", {})),
            value.get("recipe"),
        )

    @classmethod
    def from_file(cls, path, *, independent=False, registration=RigidTransform()):
        """Import native target CSV/TSV/JSON/NPZ, keeping units explicit.

        Native imported-target formats already specify world SI coordinates;
        portable measured-map JSON/NPZ can additionally state source units.
        Registration always rotates vectors as well as coordinates.
        """
        from ..config import CoilGenConfig
        from ..target import load_target_field_definition

        path = Path(path).expanduser().resolve()
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError("Desired-map source exceeds the explicit 32 MiB import limit.")
        if path.suffix.lower() == ".npz":
            from zipfile import BadZipFile, ZipFile

            try:
                with ZipFile(path) as archive:
                    if (
                        len(archive.infolist()) > 32
                        or sum(entry.file_size for entry in archive.infolist()) > 64 * 1024**2
                    ):
                        raise ValueError("Desired-map NPZ expands beyond the explicit 64 MiB import limit.")
            except BadZipFile as exc:
                raise ValueError("Desired-map NPZ is not a valid numerical archive.") from exc
        try:
            data = MeasuredFieldMap.from_file(path, independent=independent)
            data = replace(data, independent=independent)
        except (ValueError, KeyError):
            target = load_target_field_definition(path, None, CoilGenConfig(target_gradient_normalization="none"))
            data = MeasuredFieldMap(
                target.coords, target.b, independent=independent, source=str(path), source_bytes=path.read_bytes()
            )
        return cls(data, registration)

    @classmethod
    def analytic(cls, region, expressions=("0", "0", "x"), *, resolution=8, independent=False):
        from ..target import evaluate_field_expression
        from ..target_regions import sample_target_region

        if len(expressions) != 3 or not all(isinstance(value, str) for value in expressions):
            raise ValueError("Analytic desired field requires three Cartesian expressions in tesla.")
        points = sample_target_region(region, resolution, independent=independent)
        field = np.column_stack([evaluate_field_expression(expression, points) for expression in expressions])
        from ..target_regions import TargetRegion

        return cls(
            MeasuredFieldMap(points, field, independent=independent, source="Analytic: " + "; ".join(expressions)),
            recipe={
                "region": TargetRegion.from_dict(region).to_dict(),
                "expressions": expressions,
                "resolution": resolution,
            },
        )


@dataclass(frozen=True, slots=True)
class RegionalRequirement:
    name: str
    design: DesiredDataset
    component: str = "Bz"
    rms_limit_T: float | None = None
    peak_limit_T: float | None = None
    weight: float = 1.0
    protected: bool = False
    validation: DesiredDataset | None = None

    def __post_init__(self):
        object.__setattr__(self, "name", _name(self.name))
        object.__setattr__(self, "component", normalize_field_component(self.component))
        if not isinstance(self.design, DesiredDataset) or self.design.data.independent:
            raise ValueError("Design data must be a declared design dataset.")
        if not isinstance(self.protected, bool):
            raise ValueError("Protected-region intent must be boolean.")
        for name in ("rms_limit_T", "peak_limit_T"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.rms_limit_T is None and self.peak_limit_T is None:
            raise ValueError("Each region needs an independent RMS or peak physical bound.")
        object.__setattr__(self, "weight", _nonnegative(self.weight, "Fitting weight"))
        if self.validation is not None and (
            not isinstance(self.validation, DesiredDataset) or not self.validation.data.independent
        ):
            raise ValueError("Validation samples must be explicitly declared independent desired data.")
        if self.protected and np.any(self.design.world_values()[1] != 0):
            raise ValueError("Protected regions require a zero desired vector field.")

    def to_dict(self):
        return {
            "name": self.name,
            "design": self.design.to_dict(),
            "component": self.component,
            "rms_limit_T": self.rms_limit_T,
            "peak_limit_T": self.peak_limit_T,
            "weight": self.weight,
            "protected": self.protected,
            "validation": self.validation.to_dict() if self.validation else None,
        }

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        value = _strict(value, cls.__dataclass_fields__, "regional requirement")
        value["design"] = DesiredDataset.from_dict(value["design"])
        if value.get("validation") is not None:
            value["validation"] = DesiredDataset.from_dict(value["validation"])
        return cls(**value)


@dataclass(frozen=True, slots=True)
class PhysicalScenario:
    name: str
    membership: str = "design"
    support_transform: RigidTransform = RigidTransform()
    sample_transform: RigidTransform = RigidTransform()
    support_scale_xyz: tuple = (1.0, 1.0, 1.0)
    support_scale_center_m: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "name", _name(self.name))
        if self.name.casefold() == "nominal":
            raise ValueError("Nominal is the implicit design scenario; use another scenario name.")
        if self.membership not in {"design", "validation"}:
            raise ValueError("Scenario membership must be design or validation.")
        if not all(isinstance(value, RigidTransform) for value in (self.support_transform, self.sample_transform)):
            raise TypeError("Physical scenarios require proper rigid support and sample transforms.")
        for name in ("support_scale_xyz", "support_scale_center_m"):
            raw = np.asarray(getattr(self, name), dtype=object)
            if raw.shape != (3,) or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in raw
            ):
                raise ValueError(f"{name} requires three finite real values.")
            values = np.asarray(raw, dtype=float)
            if not np.all(np.isfinite(values)) or (name == "support_scale_xyz" and np.any(values <= 0)):
                raise ValueError("Support scale factors must be positive; scale centre must be finite world metres.")
            object.__setattr__(self, name, tuple(float(value) for value in values))

    def to_dict(self):
        return {
            "name": self.name,
            "membership": self.membership,
            "support_transform": self.support_transform.to_dict(),
            "sample_transform": self.sample_transform.to_dict(),
            "support_scale_xyz": list(self.support_scale_xyz),
            "support_scale_center_m": list(self.support_scale_center_m),
        }

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        value = _strict(value, cls.__dataclass_fields__, "physical scenario")
        for name in ("support_transform", "sample_transform"):
            value[name] = RigidTransform(**value.get(name, {}))
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ResearchRequirements:
    regions: tuple[RegionalRequirement, ...]
    scenarios: tuple[PhysicalScenario, ...] = ()

    def __post_init__(self):
        regions = tuple(RegionalRequirement.from_dict(value) for value in self.regions)
        scenarios = tuple(PhysicalScenario.from_dict(value) for value in self.scenarios)
        if not 1 <= len(regions) <= 32 or len(scenarios) > 32:
            raise ValueError("Requirements need 1–32 regions and at most 32 explicit finite scenarios.")
        for items in (regions, scenarios):
            if len({item.name.casefold() for item in items}) != len(items):
                raise ValueError("Region and scenario names must be unique within their list.")
        if sum(len(region.design.data.coordinates) for region in regions) > MAX_SAMPLES:
            raise ValueError("Combined design datasets exceed 100,000 samples.")
        if sum(len(region.validation.data.coordinates) for region in regions if region.validation) > MAX_SAMPLES:
            raise ValueError("Combined independent validation datasets exceed 100,000 samples.")
        report_samples = sum(
            len(region.design.data.coordinates) + (len(region.validation.data.coordinates) if region.validation else 0)
            for region in regions
        )
        if report_samples * (1 + len(scenarios)) * 2 * 3 * 3 * 8 > MAX_REPORT_BYTES:
            raise ValueError("Regional/pose saved field maps exceed the explicit 512 MiB report budget.")
        if not any(not region.protected and region.weight > 0 for region in regions):
            raise ValueError("At least one target region requires a positive fitting weight.")

        def pose_key(support, sample, scale=(1.0, 1.0, 1.0), centre=(0.0, 0.0, 0.0)):
            rotation = np.asarray(support.rotation)
            linear = rotation @ np.diag(scale)
            translation = np.asarray(support.translation_m) + rotation @ (np.asarray(centre) * (1 - np.asarray(scale)))
            return json.dumps((linear.tolist(), translation.tolist(), sample.to_dict()), sort_keys=True)

        pose_keys = [pose_key(RigidTransform(), RigidTransform())] + [
            pose_key(item.support_transform, item.sample_transform, item.support_scale_xyz, item.support_scale_center_m)
            for item in scenarios
        ]
        if len(set(pose_keys)) != len(pose_keys):
            raise ValueError(
                "Explicit poses must differ from nominal and from each other; held-out poses cannot repeat training."
            )
        design_points = np.vstack([region.design.world_values()[0] for region in regions])
        from scipy.spatial import cKDTree

        tree = cKDTree(design_points)
        for region in regions:
            if region.validation is None:
                continue
            if region.validation.digest in {item.design.digest for item in regions}:
                raise ValueError("Validation data repeats a fitted desired dataset.")
            points = region.validation.world_values()[0]
            nearest = tree.query(points)[0]
            # Roundoff-equivalent locations do not become independent by changing units.
            tolerance = 64 * np.finfo(float).eps * np.maximum(1, np.max(np.abs(points), axis=1))
            if np.any(nearest <= tolerance):
                raise ValueError("Independent validation coordinates overlap fitted design samples.")
            if region.protected and np.any(region.validation.world_values()[1] != 0):
                raise ValueError("Protected-region validation requires zero desired vectors.")
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "scenarios", scenarios)

    def to_dict(self):
        return {
            "schema": SCHEMA,
            "regions": [item.to_dict() for item in self.regions],
            "scenarios": [item.to_dict() for item in self.scenarios],
        }

    @property
    def digest(self):
        return sha256(
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def evaluation_contract_digest(self):
        """Same regional datasets, bounds and poses, independent of training membership."""
        value = self.to_dict()
        for scenario in value["scenarios"]:
            scenario.pop("membership")
        return sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        value = _strict(value, {"schema", "regions", "scenarios"}, "research requirements")
        if value.get("schema") != SCHEMA:
            raise ValueError("Unsupported research requirements schema.")
        return cls(tuple(value["regions"]), tuple(value.get("scenarios", ())))


def requirements_stage_fingerprints(config, project_root, spec=None):
    from ..pipeline_stages import stage_fingerprints

    if spec is None:
        return stage_fingerprints(config, project_root)
    native = stage_fingerprints(requirements_validation_config(config), project_root)
    digest = ResearchRequirements.from_dict(spec).digest
    return {key: sha256((value + "/" + digest).encode()).hexdigest() for key, value in native.items()}


def requirements_target(spec):
    spec = ResearchRequirements.from_dict(spec)
    data = [region.design.world_values() for region in spec.regions]
    counts = [len(points) for points, _ in data]
    return TargetField(
        np.vstack([points for points, _ in data]),
        np.vstack([field for _, field in data]),
        np.concatenate([np.full(count, region.weight) for region, count in zip(spec.regions, counts)]),
        np.concatenate([np.full(count, region.protected) for region, count in zip(spec.regions, counts)]),
        np.concatenate([np.full(count, index) for index, count in enumerate(counts)]),
        target_diagnostics={
            "requirements_digest": spec.digest,
            "named_region_count": len(spec.regions),
            "target_normalization": "explicit_si_no_rescaling",
        },
    )


def _points_transform(points, transform, scale_xyz=(1.0, 1.0, 1.0), centre_m=(0.0, 0.0, 0.0)):
    points = np.asarray(points)
    if tuple(scale_xyz) != (1.0, 1.0, 1.0):
        points = (points - centre_m) * scale_xyz + centre_m
    return points @ np.asarray(transform.rotation).T + transform.translation_m


def _scenario_data(dataset, transform):
    points, field = dataset.world_values()
    return _points_transform(points, transform), field @ np.asarray(transform.rotation).T


def _field_operator(
    parts, points, config, support=RigidTransform(), scale_xyz=(1.0, 1.0, 1.0), centre_m=(0.0, 0.0, 0.0)
):
    from ..basis import calculate_basis_functions, calculate_one_ring_by_mesh
    from ..geometry import compute_face_normals, compute_vertex_normals
    from ..physics import calculate_sensitivity_matrix

    columns = sum(part.mesh.n_vertices for part in parts)
    if len(points) * columns * 3 * 8 > MAX_OPERATOR_BYTES:
        raise ValueError("One physical scenario operator exceeds the explicit 512 MiB budget.")
    copies = []
    for part in parts:
        mesh = part.mesh.copy()
        mesh.vertices = _points_transform(mesh.vertices, support, scale_xyz, centre_m)
        _reject_sheet_samples(mesh, points)
        mesh.face_normals = compute_face_normals(mesh)
        mesh.vertex_normals = compute_vertex_normals(mesh)
        copies.append(CoilPart(mesh))
    calculate_one_ring_by_mesh(copies)
    calculate_basis_functions(copies)
    target = TargetField(
        points, np.zeros_like(points), np.ones(len(points)), np.zeros(len(points), bool), np.zeros(len(points), int)
    )
    calculate_sensitivity_matrix(copies, target, config)
    return np.concatenate([part.sensitivity_matrix for part in copies], axis=2)


def _reject_sheet_samples(mesh, points):
    from scipy.spatial import cKDTree
    from ..geometry_kernels import _point_triangle_sq_numpy

    triangles = mesh.vertices[mesh.faces]
    centres = triangles.mean(axis=1)
    radius = float(np.max(np.linalg.norm(triangles - centres[:, None], axis=2)))
    scale = max(float(np.max(np.abs(mesh.vertices))), float(np.max(np.abs(points))), 1)
    tolerance = 64 * np.finfo(float).eps * scale
    tree = cKDTree(centres)
    for start in range(0, len(points), 256):
        samples = points[start : start + 256]
        candidates = tree.query_ball_point(samples, radius + tolerance)
        counts = np.array([len(ids) for ids in candidates])
        if not np.any(counts):
            continue
        if int(counts.sum()) > 1_000_000:
            raise ValueError("Sheet-sample proximity candidates exceed the explicit safety budget.")
        ids = np.concatenate([np.asarray(ids, dtype=int) for ids in candidates])
        query = np.repeat(samples, counts, axis=0)
        if np.any(_point_triangle_sq_numpy(query, triangles[ids]) <= tolerance**2):
            raise FieldSamplingError(
                "A desired sample lies on the current-sheet surface; move it off the singular sheet."
            )


def requirements_evaluation_config(config):
    """Set the existing fixed contour-current policy without retaining dense operators."""
    from ..research.convex import ResearchPhysicsSpec

    physics = ResearchPhysicsSpec.from_dict(config.research_physics or {})
    return replace(
        config,
        research_physics=replace(physics, include_default_field_constraints=False).to_dict(),
        target_field_component="Bz",
        target_gradient_normalization="none",
    )


def requirements_validation_config(config):
    """Validate active physics while ignoring the replaced legacy target source."""
    return replace(
        requirements_evaluation_config(config),
        field_shape_function="x",
        target_field_definition_file="none",
        target_field_definition_field_name="none",
        target_mesh_file="none",
        secondary_target_mesh_file="none",
        target_region_geometry=None,
        target_region_radius=0.05,
        target_region_resolution=2,
        target_gradient_normalization="none",
        target_field_component="Bz",
        target_field_units="T",
        target_gradient_strength=1.0,
        use_only_target_mesh_verts=False,
    )


def prepare_requirement_constraints(parts, config, spec):
    from ..research.convex import LinearFieldConstraint, ResearchPhysicsSpec

    spec = ResearchRequirements.from_dict(spec)
    physics = ResearchPhysicsSpec.from_dict(config.research_physics or {})
    constraints = list(physics.field_constraints)
    poses = [("Nominal", RigidTransform(), RigidTransform(), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))] + [
        (item.name, item.support_transform, item.sample_transform, item.support_scale_xyz, item.support_scale_center_m)
        for item in spec.scenarios
        if item.membership == "design"
    ]
    expected_bytes = (
        len(poses)
        * sum(part.mesh.n_vertices for part in parts)
        * 8
        * sum(
            len(region.design.data.coordinates) * (3 if region.component == "vector" else 1) for region in spec.regions
        )
    )
    if expected_bytes > MAX_OPERATOR_BYTES:
        raise ValueError("Combined regional/pose operators exceed the explicit 512 MiB budget.")
    allocated = 0
    for pose_name, support, sample, scale, centre in poses:
        for region in spec.regions:
            points, target = _scenario_data(region.design, sample)
            sensitivity = _field_operator(parts, points, config, support, scale, centre)
            matrix = sensitivity_component_matrix(sensitivity, region.component, b0_direction=config.b_0_direction)
            allocated += matrix.nbytes
            if allocated > MAX_OPERATOR_BYTES:
                raise ValueError("Combined regional/pose operators exceed the explicit 512 MiB budget.")
            desired = field_component_values(target, region.component, b0_direction=config.b_0_direction)
            for label, limit, norm in (("RMS", region.rms_limit_T, "rms"), ("peak", region.peak_limit_T, "linf")):
                if limit is not None:
                    constraints.append(
                        LinearFieldConstraint(
                            f"Region:{region.name}/Pose:{pose_name}/{label}",
                            matrix,
                            desired,
                            limit,
                            norm,
                            group_size=3 if region.component == "vector" else 1,
                        )
                    )
    return replace(
        config,
        research_physics=replace(
            physics, field_constraints=tuple(constraints), include_default_field_constraints=False
        ).to_dict(),
    )


def _metrics(field, target, points, region, config, *, design_margin=1.0):
    from ..research.convex import ResearchPhysicsSpec

    physics = ResearchPhysicsSpec.from_dict(config.research_physics or {})
    actual = field_component_values(field, region.component, b0_direction=config.b_0_direction).reshape(len(points), -1)
    desired = field_component_values(target, region.component, b0_direction=config.b_0_direction).reshape(
        len(points), -1
    )
    residual = actual - desired
    error = np.hypot.reduce(residual, axis=1)
    rms = float(np.linalg.norm(error / np.sqrt(len(error))))
    peak = float(np.max(error))
    if not np.isfinite(rms) or not np.isfinite(peak):
        return {"available": False, "passed": None, "reason": "Nonfinite field residual."}

    def effective(limit):
        if limit is None:
            return None
        budget = limit / design_margin
        return budget * (1 + physics.acceptance_relative_tolerance) + physics.acceptance_absolute_by_unit.get("T", 0)

    rms_effective, peak_effective = effective(region.rms_limit_T), effective(region.peak_limit_T)
    checks = [value <= limit for value, limit in ((rms, rms_effective), (peak, peak_effective)) if limit is not None]
    failed = np.flatnonzero(error > peak_effective) if region.peak_limit_T is not None else np.array([], int)
    peak_index = int(np.argmax(error))
    metrics = {
        "available": True,
        "passed": all(checks),
        "rms_error_T": rms,
        "peak_error_T": peak,
        "peak_point_m": points[peak_index].tolist(),
        "peak_sample_index": peak_index,
        "failed_sample_indices": failed.tolist(),
        "sample_count": len(points),
        "rms_limit_T": region.rms_limit_T,
        "peak_limit_T": region.peak_limit_T,
        "validation_rms_limit_T": region.rms_limit_T / design_margin if region.rms_limit_T is not None else None,
        "validation_peak_limit_T": region.peak_limit_T / design_margin if region.peak_limit_T is not None else None,
        "effective_rms_limit_T": rms_effective,
        "effective_peak_limit_T": peak_effective,
        "design_margin": design_margin,
        "acceptance_relative_tolerance": physics.acceptance_relative_tolerance,
        "acceptance_absolute_T": physics.acceptance_absolute_by_unit.get("T", 0),
        "component": region.component,
        "protected": region.protected,
        "point_weights": "equal_for_physical_bounds",
        "fitting_weight": region.weight,
    }
    if region.protected:
        magnitude = np.hypot.reduce(actual, axis=1)
        vector_magnitude = np.hypot.reduce(field, axis=1)
        metrics.update(
            suppression_rms_T=float(np.linalg.norm(magnitude / np.sqrt(len(points)))),
            suppression_peak_T=float(np.max(magnitude)),
            suppression_bounded_component=region.component,
            suppression_vector_rms_T=float(np.linalg.norm(vector_magnitude / np.sqrt(len(points)))),
            suppression_vector_peak_T=float(np.max(vector_magnitude)),
        )
    # Centre and scale coordinates before the descriptive local affine fit.
    # Rank loss cannot certify a missing out-of-plane gradient component.
    centre = points.mean(axis=0)
    offsets = points - centre
    scale = float(np.max(np.abs(offsets)))
    affine = np.column_stack((np.ones(len(points)), offsets / scale if scale else offsets))
    singular = np.linalg.svd(affine, compute_uv=False)
    rank = int(np.linalg.matrix_rank(affine))
    condition = float(singular[0] / singular[-1]) if rank == 4 else None
    metrics.update(
        regional_gradient_available=rank == 4,
        regional_gradient_rank=rank,
        regional_gradient_condition_number=condition,
        regional_gradient_acceptance_claim=False,
        regional_gradient_fit_center_m=centre.tolist(),
        regional_gradient_fit_scale_m=scale,
    )
    if rank == 4:
        gradient = np.linalg.lstsq(affine, field, rcond=None)[0][1:].T / scale
        desired_gradient = np.linalg.lstsq(affine, target, rcond=None)[0][1:].T / scale
        metrics["field_gradient_T_per_m"] = gradient.tolist()
        metrics["desired_gradient_T_per_m"] = desired_gradient.tolist()
        metrics["gradient_residual_T_per_m"] = (gradient - desired_gradient).tolist()
    else:
        metrics["regional_gradient_reason"] = "Samples cannot identify all three spatial gradient directions."
    return metrics


def evaluate_requirements(
    parts, config, spec, *, current_A=None, fabrication=None, complete_paths=None, design_margin=1.0
):
    """Evaluate saved sheet and routed geometry at unchanged signed currents.

    ``complete_paths`` optionally maps body IDs to (XYZ path, current A), for
    caller-verified complete electrical circuits. Otherwise only saved routed
    centroid paths qualify; unconnected contour copies are unavailable. No
    current fitting, normalization, target interpolation, or mesh remeshing.
    """
    from ..evaluation import layout_field_geometry
    from ..physics import biot_savart_wire_path
    from ..research.validation import _samples_on_wire

    spec = ResearchRequirements.from_dict(spec)
    if (
        isinstance(design_margin, bool)
        or not isinstance(design_margin, Real)
        or not np.isfinite(design_margin)
        or not 0 < design_margin <= 1
    ):
        raise ValueError("design_margin must be a finite fraction in (0,1].")
    design_margin = float(design_margin)
    vectors = [part.stream_function for part in parts]
    sheet = all(value is not None and np.all(np.isfinite(value)) for value in vectors)
    sf = np.concatenate(vectors) if sheet else None
    paths, body_ids, reason = [], [], ""
    partial_circuit = fabrication is not None and complete_paths is None
    if complete_paths is not None:
        for name, (points, current) in complete_paths.items():
            _name(name)
            raw = np.asarray(points)
            if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 3 or not 2 <= len(raw) <= 100_000:
                raise ValueError("Complete circuits require finite world SI paths and signed currents.")
            points = np.asarray(raw, dtype=float)
            if (
                not np.all(np.isfinite(points))
                or not np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-15)
                or np.linalg.norm(points[-1] - points[0]) > 1e-10 * max(float(np.max(np.ptp(points, axis=0))), 0.01)
            ):
                raise ValueError("Complete electrical paths require finite nondegenerate closed geometry.")
            paths.append((points, _signed_current(current)))
            body_ids.append(name)
    elif fabrication is not None:
        paths, reason = _recorded_fabrication_paths(fabrication)
        body_ids = [plan["body_id"] for plan in fabrication.winding_steps] if paths else []
    else:
        for part_index, part in enumerate(parts):
            points, source = layout_field_geometry(part, config)
            current = part.contour_step if current_A is None else current_A
            if points is None or current is None:
                reason = "An actual saved routed centroid and unchanged finite current are required for every support."
                paths = []
                break
            try:
                paths.append((points, _signed_current(current)))
                body_ids.append(f"support_{part_index + 1:04d}")
            except ValueError:
                if current_A is not None:
                    raise
                reason = "The saved contour current is unavailable or nonfinite."
                paths = []
                break
    poses = [("Nominal", "design", RigidTransform(), RigidTransform(), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))] + [
        (
            item.name,
            item.membership,
            item.support_transform,
            item.sample_transform,
            item.support_scale_xyz,
            item.support_scale_center_m,
        )
        for item in spec.scenarios
    ]
    rows, arrays = [], {}
    for pose_name, membership, support, sample, scale, centre in poses:
        for region in spec.regions:
            datasets = [("design", region.design)]
            if region.validation is not None:
                datasets.append(("independent_validation", region.validation))
            for sampling, dataset in datasets:
                points, target = _scenario_data(dataset, sample)
                common = {
                    "region": region.name,
                    "scenario": pose_name,
                    "scenario_membership": membership,
                    "sampling": sampling,
                    "independent_sampling": sampling == "independent_validation",
                    "dataset_sha256": dataset.digest,
                    "source_sha256": dataset.data.source_sha256,
                    "sample_count": len(points),
                    "component": region.component,
                    "support_scale_xyz": list(scale),
                    "support_scale_center_m": list(centre),
                    "validation_margin_applied": sampling == "independent_validation" or membership == "validation",
                }
                sheet_reason = "Saved nodal stream function is unavailable."
                try:
                    sensitivity = _field_operator(parts, points, config, support, scale, centre) if sheet else None
                except FieldSamplingError as exc:
                    sensitivity = None
                    sheet_reason = str(exc)
                fields = [
                    (
                        "continuous_sheet",
                        np.einsum("ctv,v->tc", sensitivity, sf) if sensitivity is not None else None,
                        sheet_reason,
                    )
                ]
                wire = None
                wire_reason = reason
                if paths:
                    moved = [(_points_transform(path, support, scale, centre), current) for path, current in paths]
                    if any(np.any(_samples_on_wire(path, points)) for path, _ in moved):
                        wire_reason = "Sample lies on a conductor; singular thin-wire field cannot certify a bound."
                    else:
                        wire = sum(
                            (current * biot_savart_wire_path(path, points, worker_count=1) for path, current in moved),
                            np.zeros_like(points),
                        )
                fields.append(
                    (
                        "complete_electrical_paths"
                        if complete_paths is not None
                        else "partial_electrical_paths"
                        if partial_circuit
                        else "saved_routed_centroids",
                        wire,
                        wire_reason or "No saved routed winding is available.",
                    )
                )
                for scope, field, missing in fields:
                    row = {**common, "scope": scope, "current_policy": "unchanged_saved_current_no_refit"}
                    if scope != "continuous_sheet":
                        row["currents_A"] = [current for _, current in paths]
                        row["body_ids"] = body_ids if paths else []
                        row["path_sha256"] = [
                            sha256(np.asarray(path, dtype="<f8").tobytes()).hexdigest() for path, _ in paths
                        ]
                    if field is None:
                        row.update(available=False, passed=None, reason=missing)
                    else:
                        margin = design_margin if common["validation_margin_applied"] else 1.0
                        row.update(_metrics(field, target, points, region, config, design_margin=margin))
                        if scope == "partial_electrical_paths":
                            row.update(
                                residual_bounds_met=row["passed"],
                                passed=None,
                                circuit_complete=False,
                                reason="Recorded partial electrical paths omit a verified complete return circuit.",
                            )
                        prefix = f"requirements_{len(rows):04d}"
                        row["array_prefix"] = prefix
                        arrays[prefix + "_points_m"] = points
                        arrays[prefix + "_field_T"] = field
                        arrays[prefix + "_target_T"] = target
                    rows.append(row)
    available = [row for row in rows if row.get("available")]
    wire_rows = [row for row in rows if row["scope"] != "continuous_sheet"]

    def utilization(row):
        return max(
            value / limit if limit > 0 else (np.inf if value else 0)
            for value, limit in (
                (row["rms_error_T"], row["effective_rms_limit_T"]),
                (row["peak_error_T"], row["effective_peak_limit_T"]),
            )
            if limit is not None
        )

    limiting = max(available, key=utilization) if available else None
    from ..research.convex import ResearchPhysicsSpec

    physics = ResearchPhysicsSpec.from_dict(config.research_physics or {})
    evaluation_contract = {
        "regions_and_poses": spec.evaluation_contract_digest,
        "design_margin": design_margin,
        "acceptance_relative_tolerance": physics.acceptance_relative_tolerance,
        "acceptance_absolute_T": physics.acceptance_absolute_by_unit.get("T", 0),
    }
    return {
        "schema": SCHEMA,
        "requirements_sha256": spec.digest,
        "evaluation_contract_sha256": sha256(json.dumps(evaluation_contract, sort_keys=True).encode()).hexdigest(),
        "regional_contract_sha256": spec.evaluation_contract_digest,
        "design_margin": design_margin,
        "rows": rows,
        "b0_direction": list(config.b_0_direction),
        "support_geometry_sha256": [
            sha256(
                np.asarray(part.mesh.vertices, dtype="<f8").tobytes()
                + np.asarray(part.mesh.faces, dtype="<i8").tobytes()
            ).hexdigest()
            for part in parts
        ],
        "stream_function_sha256": sha256(np.asarray(sf, dtype="<f8").tobytes()).hexdigest() if sf is not None else None,
        "coverage": "listed_finite_registered_poses_and_dimensional_perturbations_only",
        "acceptance_scope": "named_regions_on_recorded_samples_and_listed_poses",
        "common_nodal_topology": True,
        "material_and_profile_perturbations_assessed": False,
        "perturbed_resistance_power_assessed": False,
        "current_refitted": False,
        "external_feeds_included": complete_paths is not None,
        "independent_sampling_complete": all(region.validation is not None for region in spec.regions),
        "passed": all(row.get("passed") is True for row in wire_rows)
        if all(row.get("available") for row in wire_rows) and not partial_circuit
        else None,
        "limiting_region": limiting["region"] if limiting else None,
        "limiting_scenario": limiting["scenario"] if limiting else None,
    }, arrays


def _recorded_fabrication_paths(fabrication):
    """Use the authoritative fabricated electrical paths, never earlier centroid fallback."""
    plans = list(getattr(fabrication, "winding_steps", ()))
    bodies = list(getattr(fabrication, "electrical_metrics", {}).get("bodies", ()))
    plan_ids = [row.get("body_id") for row in plans]
    body_ids = [row.get("body_id") for row in bodies]
    if (
        not plans
        or len(plan_ids) != len(set(plan_ids))
        or len(body_ids) != len(set(body_ids))
        or set(plan_ids) != set(body_ids)
        or any(not isinstance(value, str) or not value for value in plan_ids)
    ):
        return [], "Fabricated paths and recorded currents need complete matching unique body identities."
    currents = {body["body_id"]: body.get("current_A") for body in bodies}
    paths = []
    for plan in plans:
        try:
            raw = np.asarray(plan["electrical_points_m"])
            if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 3 or not 2 <= len(raw) <= 100_000:
                raise ValueError("finite real Nx3 electrical points are required")
            path = np.asarray(raw, dtype=float)
            if not np.all(np.isfinite(path)) or not np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-15):
                raise ValueError("nondegenerate finite electrical points are required")
            paths.append((path, _signed_current(currents[plan["body_id"]])))
        except (ValueError, TypeError, KeyError) as exc:
            return [], f"Recorded fabricated body {plan['body_id']} is unavailable: {exc}."
    return paths, ""


def nominal_requirements(spec):
    """Evaluate the same poses, optimizing only nominal for a paired baseline."""
    spec = ResearchRequirements.from_dict(spec)
    return replace(spec, scenarios=tuple(replace(item, membership="validation") for item in spec.scenarios))


def named_independent_sheet_validation(report, spec):
    """Qualify every region on all declared independent samples and held-out cases."""
    spec = ResearchRequirements.from_dict(spec)
    rows = [
        row
        for row in report.get("rows", ())
        if row["scope"] == "continuous_sheet"
        and (row["independent_sampling"] or row["scenario_membership"] == "validation")
    ]
    regions = {region.name for region in spec.regions}
    available = (
        bool(rows) and regions <= {row["region"] for row in rows} and all(row.get("available") is True for row in rows)
    )
    return {
        "available": available,
        "verified": available,
        "passed": all(row.get("passed") is True for row in rows) if available else None,
        "scope": "named_regional_independent_sheet",
        "source": "owned_regional_desired_samples_and_heldout_poses_no_refit",
        "value": None,
        "limit": None,
        "unit": "T",
        "relative_field_error": None,
        "reason": ""
        if available
        else "Every named region requires available independent desired samples or held-out poses; all declared independent cases must be evaluable.",
    }


def requirements_constraint_rows(report, *, include_wire=True):
    """Convert named evidence to scoped, separately interpretable physical checks."""
    checks = []
    for row in report.get("rows", ()):
        if row["scope"] != "continuous_sheet" and not include_wire:
            continue
        name = f"{row['region']} · {row['scenario']} · {row['sampling']}"
        common = {
            "scope": row["scope"],
            "region": row["region"],
            "scenario": row["scenario"],
            "sampling": row["sampling"],
            "unit": "T",
            "reason": row.get("reason", ""),
        }
        if not row.get("available"):
            checks.append(
                {
                    **common,
                    "name": name + " · unavailable",
                    "value": None,
                    "limit": None,
                    "effective_limit": None,
                    "passed": None,
                    "verified": False,
                }
            )
            continue
        for label, value, limit, effective in (
            ("RMS", "rms_error_T", "rms_limit_T", "effective_rms_limit_T"),
            ("peak", "peak_error_T", "peak_limit_T", "effective_peak_limit_T"),
        ):
            if row.get(limit) is None:
                continue
            passed = None if row.get("passed") is None else row[value] <= row[effective]
            checks.append(
                {
                    **common,
                    "name": name + " · " + label,
                    "value": row[value],
                    "limit": row[limit],
                    "effective_limit": row[effective],
                    "passed": passed,
                    "verified": True,
                }
            )
    return checks
