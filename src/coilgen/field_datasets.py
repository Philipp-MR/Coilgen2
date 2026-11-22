"""Owned vector field datasets and rigid registration in physical SI units."""

from __future__ import annotations
import base64
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import numpy as np
from .config_semantics import LENGTH_UNITS_TO_M as _LENGTH_UNITS
from .config_semantics import VECTOR_FIELD_UNITS_TO_T as _FIELD_UNITS


MEASUREMENT_SCHEMA = "coilgen-measured-field-v1"


def _xyz(value, name, minimum=1):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} finite real XYZ rows.")
    array = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class RigidTransform:
    rotation: tuple = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    translation_m: tuple = (0, 0, 0)

    def __post_init__(self):
        raw_rotation, raw_translation = np.asarray(self.rotation), np.asarray(self.translation_m)
        if raw_rotation.dtype.kind not in "iuf" or raw_translation.dtype.kind not in "iuf":
            raise ValueError("Rigid registration requires real numeric rotation and translation values.")
        rotation = np.asarray(raw_rotation, dtype=float)
        translation = np.asarray(raw_translation, dtype=float)
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(translation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-10)
        ):
            raise ValueError("Registration requires a finite proper rigid rotation and translation in metres.")
        object.__setattr__(self, "rotation", tuple(tuple(float(value) for value in row) for row in rotation))
        object.__setattr__(self, "translation_m", tuple(float(value) for value in translation))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MeasuredFieldMap:
    coordinates: np.ndarray
    field: np.ndarray
    length_unit: str = "m"
    field_unit: str = "T"
    independent: bool = False
    source: str = ""
    source_bytes: bytes = b""

    def __post_init__(self):
        if self.length_unit not in _LENGTH_UNITS or self.field_unit not in _FIELD_UNITS:
            raise ValueError("Measured maps require explicit supported length and field units.")
        if (
            not isinstance(self.independent, bool)
            or not isinstance(self.source, str)
            or not isinstance(self.source_bytes, bytes)
        ):
            raise TypeError("Measured provenance requires boolean independence, a source label and owned bytes.")
        coordinates, field = _xyz(self.coordinates, "Measured coordinates"), _xyz(self.field, "Measured vector field")
        if coordinates.shape != field.shape or len(coordinates) > 100_000:
            raise ValueError("Measured coordinates and fields must match, with at most 100,000 samples.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "field", field)

    @property
    def source_sha256(self):
        return sha256(self.source_bytes).hexdigest() if self.source_bytes else None

    @property
    def data_sha256(self):
        digest = sha256(self.coordinates.tobytes() + self.field.tobytes())
        digest.update(f"{self.length_unit}/{self.field_unit}".encode())
        return digest.hexdigest()

    def world_values(self, transform=RigidTransform()):
        if not isinstance(transform, RigidTransform):
            raise TypeError("Measured registration requires a RigidTransform.")
        rotation = np.asarray(transform.rotation)
        points = (self.coordinates * _LENGTH_UNITS[self.length_unit]) @ rotation.T + transform.translation_m
        field = (self.field * _FIELD_UNITS[self.field_unit]) @ rotation.T
        return points, field

    def to_dict(self):
        return {
            "schema": MEASUREMENT_SCHEMA,
            "coordinates": self.coordinates.tolist(),
            "field": self.field.tolist(),
            "length_unit": self.length_unit,
            "field_unit": self.field_unit,
            "independent": self.independent,
            "source": self.source,
            "source_bytes": base64.b64encode(self.source_bytes).decode("ascii"),
            "source_sha256": self.source_sha256,
            "data_sha256": self.data_sha256,
        }

    @classmethod
    def from_dict(cls, value):
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "schema",
                "coordinates",
                "field",
                "length_unit",
                "field_unit",
                "independent",
                "source",
                "source_bytes",
                "source_sha256",
                "data_sha256",
            }
            or value["schema"] != MEASUREMENT_SCHEMA
        ):
            raise ValueError("Malformed measured-map record.")
        measured = cls(
            value["coordinates"],
            value["field"],
            value["length_unit"],
            value["field_unit"],
            value["independent"],
            value["source"],
            base64.b64decode(value["source_bytes"], validate=True),
        )
        if measured.data_sha256 != value["data_sha256"] or measured.source_sha256 != value["source_sha256"]:
            raise ValueError("Measured-map data/provenance integrity check failed.")
        return measured

    @classmethod
    def from_file(cls, path, *, independent=False):
        """CSV columns: x_m,y_m,z_m,Bx_T,By_T,Bz_T; portable JSON/NPZ also supported.

        Independence is an explicit declaration, then checked against the run's
        fitted inputs. Original file bytes survive changes or relocation.
        """
        path = Path(path).expanduser().resolve()
        data = path.read_bytes()
        if path.suffix.lower() == ".csv":
            rows = np.genfromtxt(StringIO(data.decode("utf-8-sig")), delimiter=",", names=True, ndmin=1)
            names = ("x_m", "y_m", "z_m", "Bx_T", "By_T", "Bz_T")
            if not rows.dtype.names or not set(names) <= set(rows.dtype.names):
                raise ValueError("Measured CSV requires x_m,y_m,z_m,Bx_T,By_T,Bz_T columns.")
            return cls(
                np.column_stack([rows[name] for name in names[:3]]),
                np.column_stack([rows[name] for name in names[3:]]),
                independent=independent,
                source=str(path),
                source_bytes=data,
            )
        if path.suffix.lower() == ".json":
            measured = cls.from_dict(json.loads(data))
        elif path.suffix.lower() == ".npz":
            with np.load(BytesIO(data), allow_pickle=False) as arrays:
                measured = cls.from_dict(json.loads(str(arrays["measurement_json"])))
        else:
            raise ValueError("Measured data must be CSV or a portable measured-map JSON/NPZ record.")
        return measured

    def save(self, path):
        from .persistence import atomic_json

        path = Path(path).expanduser().resolve()
        if path.suffix.lower() == ".json":
            atomic_json(path, self.to_dict())
        elif path.suffix.lower() == ".npz":
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, measurement_json=json.dumps(self.to_dict(), allow_nan=False))
        else:
            raise ValueError("Save a portable measured-map .json or .npz file.")
        return path
