"""Exact segment intersections with physical 3D cutting volumes (SI units)."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CutShape:
    kind: str
    center: np.ndarray
    normal: np.ndarray
    size: np.ndarray  # half width, sphere radius, or box half lengths
    tangent: np.ndarray

    @classmethod
    def from_dict(cls, value):
        kind = value["kind"]
        center, normal, size = (np.asarray(value[name], float) for name in ("center", "normal", "size"))
        if kind not in {"plane", "sphere", "box"}:
            raise ValueError("Choose a plane, sphere or box cut.")
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (center, normal, size)):
            raise ValueError("Cut shapes need finite 3D coordinates and dimensions.")
        if np.linalg.norm(normal) < 1e-12 or np.any(size <= 1e-9):
            raise ValueError("Use a nonzero cut normal and positive dimensions.")
        if abs(np.linalg.norm(normal) - 1) > 1e-14:
            normal = normal / np.linalg.norm(normal)
        tangent = value.get("tangent")
        if tangent is None:
            tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
        tangent = np.asarray(tangent, float)
        if tangent.shape != (3,) or not np.isfinite(tangent).all():
            raise ValueError("A cut shape needs a finite orientation vector.")
        if abs(tangent @ normal) > 1e-14:
            tangent = tangent - (tangent @ normal) * normal
        length = np.linalg.norm(tangent)
        if length < 1e-12:
            raise ValueError("The cut shape orientation cannot be parallel to its normal.")
        if abs(length - 1) > 1e-14:
            tangent = tangent / length
        return cls(kind, center.copy(), normal.copy(), size.copy(), tangent.copy())

    def to_dict(self):
        return dict(
            kind=self.kind,
            center=self.center.tolist(),
            normal=self.normal.tolist(),
            size=self.size.tolist(),
            tangent=self.tangent.tolist(),
        )

    @property
    def axes(self):
        return np.array([self.normal, self.tangent, np.cross(self.normal, self.tangent)])

    def intervals(self, arc):
        """Return connected inside intervals along a closed directed contour."""
        points = (arc.points - self.center) @ self.axes.T
        intervals = []
        for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
            direction = b - a
            lo, hi = 0.0, 1.0
            if self.kind == "sphere":
                aa, bb, cc = direction @ direction, 2 * (a @ direction), a @ a - self.size[0] ** 2
                if aa <= 1e-30:
                    continue
                discriminant = bb * bb - 4 * aa * cc
                if discriminant <= 0:
                    continue  # A tangent does not remove a finite contour arc.
                root = np.sqrt(discriminant)
                # Stable quadratic roots, including crossings near an endpoint.
                q = -0.5 * (bb + np.copysign(root, bb))
                roots = sorted((q / aa, cc / q)) if q else (-bb / (2 * aa),) * 2
                lo, hi = max(lo, roots[0]), min(hi, roots[1])
            else:
                for axis in range(1 if self.kind == "plane" else 3):
                    if abs(direction[axis]) <= 1e-15:
                        if abs(a[axis]) > self.size[axis]:
                            lo, hi = 1.0, 0.0
                            break
                    else:
                        p, q = sorted(
                            (
                                (-self.size[axis] - a[axis]) / direction[axis],
                                (self.size[axis] - a[axis]) / direction[axis],
                            )
                        )
                        lo, hi = max(lo, p), min(hi, q)
            if hi > lo and (hi - lo) * arc.lengths[index] > 1e-12:
                start, end = arc.positions[index] + np.array([lo, hi]) * arc.lengths[index]
                if intervals and abs(start - intervals[-1][1]) < 1e-11:
                    intervals[-1][1] = float(end)
                else:
                    intervals.append([float(start), float(end)])
        if len(intervals) > 1 and intervals[0][0] < 1e-11 and arc.perimeter - intervals[-1][1] < 1e-11:
            intervals = [[intervals[-1][0], intervals[0][1] + arc.perimeter], *intervals[1:-1]]
        if any(end - start >= arc.perimeter - 1e-10 for start, end in intervals):
            raise ValueError("The cut shape covers an entire turn. Reduce its size.")
        return intervals

    def moved(self, role, point):
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("A cut handle needs three finite coordinates.")
        value = self.to_dict()
        if role == "center":
            value["center"] = point.tolist()
        elif role == "normal":
            direction = point - self.center
            length = np.linalg.norm(direction)
            if length < 1e-12:
                raise ValueError("Keep the tilt handle away from the cut center.")
            direction /= length
            cross, cosine = np.cross(self.normal, direction), float(self.normal @ direction)
            tangent = self.tangent.copy()
            if cosine > -1 + 1e-10:
                tangent += np.cross(cross, tangent) + np.cross(cross, np.cross(cross, tangent)) / (1 + cosine)
            value["normal"], value["tangent"] = direction.tolist(), tangent.tolist()
        elif role in {"size0", "size1", "size2"}:
            index = int(role[-1])
            axis = self.axes[1] if self.kind == "sphere" else self.axes[index]
            value["size"][index] = abs(float((point - self.center) @ axis))
        else:
            raise ValueError("Unknown cut-shape handle.")
        return self.from_dict(value)

    def display(self, extent):
        axes, lines = self.axes, []
        handles = [dict(role="center", point=self.center.tolist())]
        if self.kind == "sphere":
            angle = np.linspace(0, 2 * np.pi, 65)
            for i, j in ((0, 1), (1, 2), (2, 0)):
                lines.append(
                    self.center + self.size[0] * (np.cos(angle)[:, None] * axes[i] + np.sin(angle)[:, None] * axes[j])
                )
            handles.append(dict(role="size0", point=(self.center + self.size[0] * axes[1]).tolist()))
        else:
            dimensions = self.size.copy()
            if self.kind == "plane":
                dimensions[1:] = extent * 0.45
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            vertices = self.center + (corners * dimensions) @ axes
            for i in range(8):
                for j in range(i):
                    if np.sum(corners[i] != corners[j]) == 1:
                        lines.append(vertices[[i, j]])
            for i in range(1 if self.kind == "plane" else 3):
                handles.append(dict(role=f"size{i}", point=(self.center + axes[i] * self.size[i]).tolist()))
            endpoint = self.center + self.normal * max(extent * 0.2, self.size[0] * 1.5)
            lines.append(np.array([self.center, endpoint]))
            handles.append(dict(role="normal", point=endpoint.tolist()))
        return dict(lines=[p.tolist() for p in lines], handles=handles)
