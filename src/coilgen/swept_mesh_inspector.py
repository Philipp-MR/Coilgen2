"""Interactive inspection of the actual triangulated swept conductor bodies."""

from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import Any, Sequence

import numpy as np


@dataclass
class _BodyView:
    part: Any
    part_index: int
    mesh: Any
    contour_index: int | None
    multiplicity: int
    diagnostics: dict[str, Any]
    fabrication_path: Any
    valid_faces: np.ndarray
    omitted_faces: int
    bounds: tuple[np.ndarray, np.ndarray] | None
    unreferenced_vertices: int
    surface: Any = None
    path_artist: Any = None
    topology_artists: tuple[Any, ...] = ()
    topology_counts: tuple[int, int] | None = None


@dataclass(frozen=True)
class _IssueLocation:
    body_index: int
    label: str
    ring_index: int
    section_index: int | None
    points: np.ndarray
    anchor: np.ndarray
    source: str


class SweptMeshInspector:
    """Display all stored triangles, with body isolation and inspection controls.

    Geometry arrays are read only. Connectivity highlighting is computed on
    demand; manufacturing and collision results are only read from the result.
    """

    MODES = ("Surface + edges", "Surface", "Edges only")
    VIEWS = ("Isometric", "XY", "XZ", "YZ")

    def __init__(self, figure: Any, parts: Sequence[Any], *, part_indices: Sequence[int] | None = None) -> None:
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        self.figure = figure
        self.bodies = self._body_views(parts, part_indices=part_indices)
        if not self.bodies:
            raise ValueError("No nonempty swept conductor meshes are available.")
        self.selected_index: int | None = None
        self.mode = self.MODES[0]
        self.opacity = 1.0
        self.show_support = False
        self.show_path = False
        self.show_topology = False
        self.perspective = False
        self._collapsed_axis: int | None = None
        self._disconnected = False
        self._support_artists: dict[int, Any] = {}
        self._support_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._widgets: list[Any] = []
        self._issue_scope: tuple[int, ...] | None = None
        self._issue_locations: list[_IssueLocation] = []
        self._issue_cursor = -1
        self._issue_message = ""
        self._issue_marker = None
        figure.clf()
        figure.set_size_inches(12.6, 8.4, forward=False)
        self.ax = figure.add_axes([0.025, 0.235, 0.70, 0.685], projection="3d")
        self.ax.set_proj_type("ortho")
        self.empty_label = self.ax.text2D(
            0.5,
            0.5,
            "No drawable triangles in this selection.\nSee omitted-face counts below.",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            color="#b45309",
        )
        self.heading = figure.text(0.04, 0.965, "Swept conductor mesh inspection", fontsize=14, fontweight="bold")
        self.selection_label = figure.text(0.04, 0.925, "", fontsize=10, color="#334155")
        self.failure_label = figure.text(0.04, 0.897, "", fontsize=8.7, color="#b45309", va="top")
        self.summary = figure.text(0.04, 0.218, "", fontsize=8.3, color="#334155", va="top")
        self.mode_buttons = RadioButtons(figure.add_axes([0.765, 0.755, 0.22, 0.13]), self.MODES, active=0)
        self.view_buttons = RadioButtons(figure.add_axes([0.765, 0.535, 0.22, 0.18]), self.VIEWS, active=0)
        self.layers = CheckButtons(
            figure.add_axes([0.745, 0.335, 0.24, 0.17]),
            ("Support mesh", "Fabrication path", "Topology edges", "Perspective"),
            (False, False, False, False),
        )
        self.opacity_slider = Slider(
            figure.add_axes([0.81, 0.28, 0.15, 0.022]), "Opacity", 0.05, 1.0, valinit=self.opacity, valfmt="%.2f"
        )
        self.body_slider = Slider(
            figure.add_axes([0.115, 0.065, 0.525, 0.023]),
            "Body",
            1,
            max(len(self.bodies), 2),
            valinit=1,
            valstep=1,
            valfmt="%d",
        )
        if len(self.bodies) == 1:
            self.body_slider.set_active(False)
        for widget in (self.mode_buttons, self.view_buttons, self.layers):
            for label in widget.labels:
                label.set_fontsize(9)
            for spine in widget.ax.spines.values():
                spine.set_visible(False)
        self._widgets.extend((self.mode_buttons, self.view_buttons, self.layers, self.opacity_slider, self.body_slider))
        for rectangle, label, callback in (
            ([0.045, 0.105, 0.11, 0.035], "All bodies", lambda _event: self.select_body(None)),
            ([0.168, 0.105, 0.10, 0.035], "Previous", lambda _event: self.step_body(-1)),
            ([0.281, 0.105, 0.10, 0.035], "Next", lambda _event: self.step_body(1)),
            ([0.394, 0.105, 0.115, 0.035], "Fit selected", lambda _event: self.fit_selected()),
            ([0.522, 0.105, 0.115, 0.035], "Fit all", lambda _event: self.fit_all()),
            ([0.765, 0.105, 0.20, 0.035], "Inspect issue", lambda _event: self.inspect_issue()),
        ):
            button = Button(figure.add_axes(rectangle), label)
            button.label.set_fontsize(9)
            button.on_clicked(callback)
            self._widgets.append(button)
        figure.text(
            0.04,
            0.025,
            "\n".join(
                textwrap.wrap(
                    "Full stored triangles; rotate, pan and zoom with the mouse/toolbar. "
                    "Fit selected fits the conductor; Fit all includes enabled overlays. "
                    "Topology edges: orange boundary, red nonmanifold. Collision checks are not rerun.",
                    165,
                )
            ),
            fontsize=8,
            color="#64748b",
        )
        self.body_slider.on_changed(lambda value: self.select_body(int(value) - 1, sync_slider=False))
        self.mode_buttons.on_clicked(self.set_mode)
        self.view_buttons.on_clicked(self.set_view)
        self.layers.on_clicked(self._set_layers)
        self.opacity_slider.on_changed(self.set_opacity)
        self._close_id = figure.canvas.mpl_connect("close_event", lambda _event: self.disconnect())
        self._camera_release_id = figure.canvas.mpl_connect("button_release_event", self._camera_released)
        figure._coilgen_swept_mesh_inspector = self
        self.set_view("Isometric", draw=False)
        self.update(draw=False)
        self.fit_view(draw=False)

    @staticmethod
    def _body_views(parts: Sequence[Any], *, part_indices: Sequence[int] | None = None) -> list[_BodyView]:
        output = []
        parts = list(parts)
        indices = list(range(1, len(parts) + 1)) if part_indices is None else list(part_indices)
        if len(indices) != len(parts):
            raise ValueError("part_indices must correspond to the supplied coil parts.")
        for part_index, part in zip(indices, parts):
            bodies = list(getattr(part, "swept_bodies", ()) or ())
            if bodies:
                records = [
                    (body.mesh, body.contour_index, body.turn_multiplicity, body.diagnostics, body.fabrication_path)
                    for body in bodies
                ]
            else:
                records = [
                    (
                        getattr(part, "swept_body", None),
                        None,
                        1,
                        getattr(part, "manufacturing_diagnostics", {}),
                        getattr(part, "fabrication_path", None),
                    )
                ]
            for mesh, contour, multiplicity, diagnostics, path in records:
                if mesh is None or not len(mesh.vertices) or not len(mesh.faces):
                    continue
                vertices = np.asarray(mesh.vertices, dtype=float)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
                valid[valid] &= np.all(np.isfinite(vertices[faces[valid]]), axis=(1, 2))
                referenced = np.unique(faces[valid])
                points = vertices[referenced]
                bounds = (points.min(axis=0), points.max(axis=0)) if len(points) else None
                output.append(
                    _BodyView(
                        part,
                        part_index,
                        mesh,
                        contour,
                        int(multiplicity),
                        dict(diagnostics or {}),
                        path,
                        faces[valid].copy(),
                        int(np.count_nonzero(~valid)),
                        bounds,
                        len(vertices) - len(referenced),
                    )
                )
        return output

    @property
    def visible_indices(self) -> tuple[int, ...]:
        return tuple(range(len(self.bodies))) if self.selected_index is None else (self.selected_index,)

    @property
    def rendered_face_count(self) -> int:
        return sum(len(self.bodies[index].valid_faces) for index in self.visible_indices)

    def select_body(self, index: int | None, *, sync_slider: bool = True, _issue_navigation: bool = False) -> None:
        if index is not None and not 0 <= int(index) < len(self.bodies):
            raise IndexError("Swept body index is out of range.")
        self.selected_index = None if index is None else int(index)
        if not _issue_navigation:
            self._clear_issue_focus()
        if sync_slider and index is not None and int(self.body_slider.val) != index + 1:
            previous = self.body_slider.eventson
            self.body_slider.eventson = False
            self.body_slider.set_val(index + 1)
            self.body_slider.eventson = previous
        self.update(draw=False)
        if not _issue_navigation:
            self.fit_view()

    def _clear_issue_focus(self) -> None:
        self._issue_scope = None
        self._issue_locations = []
        self._issue_cursor = -1
        self._issue_message = ""
        if self._issue_marker is not None:
            self._issue_marker.set_visible(False)

    def inspect_issue(self) -> None:
        """Cycle stored failure locations; fit only a few actual nearby rings."""
        if self._issue_scope is None:
            self._issue_scope = self.visible_indices
            self._issue_locations = [
                location
                for index in self._issue_scope
                for location in _stored_issue_locations(self.bodies[index], index)
            ]
        if not self._issue_locations:
            self._issue_message = (
                "No stored failure locations are available for this selection; inspect the mesh manually."
            )
            self.update()
            return
        self._issue_cursor = (self._issue_cursor + 1) % len(self._issue_locations)
        issue = self._issue_locations[self._issue_cursor]
        self.select_body(issue.body_index, _issue_navigation=True)
        section = f", section {issue.section_index + 1}" if issue.section_index is not None else ""
        self._issue_message = (
            f"Issue {self._issue_cursor + 1} / {len(self._issue_locations)}: {issue.label}; "
            f"body {issue.body_index + 1}, ring {issue.ring_index + 1}{section} ({issue.source})."
        )
        if self._issue_marker is None:
            self._issue_marker = self.ax.plot([], [], [], marker="o", linestyle="none", color="#f97316", markersize=7)[
                0
            ]
        self._issue_marker.set_data_3d(*issue.anchor[:, None])
        self._issue_marker.set_visible(True)
        minimum, maximum = issue.points.min(axis=0), issue.points.max(axis=0)
        center = (minimum + maximum) / 2.0
        radius = max(float(np.max(maximum - minimum)) * 0.65, 1e-6)
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)
        self.update()

    def step_body(self, step: int) -> None:
        current = int(self.body_slider.val) - 1 if self.selected_index is None else self.selected_index
        self.select_body(min(max(current + int(step), 0), len(self.bodies) - 1))

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown mesh display mode: {mode}")
        self.mode = mode
        self.update()

    def set_opacity(self, opacity: float) -> None:
        self.opacity = min(max(float(opacity), 0.05), 1.0)
        self.update()

    def set_view(self, view: str, *, draw: bool = True) -> None:
        angles = {"Isometric": (25.0, -55.0), "XY": (90.0, -90.0), "XZ": (0.0, -90.0), "YZ": (0.0, 0.0)}
        if view not in angles:
            raise ValueError(f"Unknown mesh view: {view}")
        self.ax.view_init(elev=angles[view][0], azim=angles[view][1])
        self._sync_camera_axis_labels()
        if draw:
            self.figure.canvas.draw_idle()

    def _sync_camera_axis_labels(self) -> bool:
        """Hide annotations only along an axis viewed end-on, preserving bounds."""
        from matplotlib.ticker import MaxNLocator, NullLocator

        elevation, azimuth = np.deg2rad((self.ax.elev, self.ax.azim))
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        end_on = np.flatnonzero(np.abs(direction) >= 1.0 - 1e-10)
        collapsed = int(end_on[0]) if len(end_on) else None
        changed = collapsed != self._collapsed_axis
        self._collapsed_axis = collapsed
        for index, axis in enumerate((self.ax.xaxis, self.ax.yaxis, self.ax.zaxis)):
            visible = index != collapsed
            axis.label.set_visible(visible)
            axis.offsetText.set_visible(visible)
            axis.set_major_locator(MaxNLocator(4) if visible else NullLocator())
        return changed

    def _camera_released(self, _event: Any) -> None:
        if self._sync_camera_axis_labels():
            self.figure.canvas.draw_idle()

    def _set_layers(self, _label: str) -> None:
        self.show_support, self.show_path, self.show_topology, self.perspective = self.layers.get_status()
        self.ax.set_proj_type("persp" if self.perspective else "ortho")
        self.update()

    def fit_selected(self) -> None:
        self._clear_issue_focus()
        if self.selected_index is None:
            self.select_body(int(self.body_slider.val) - 1)
        else:
            self.update(draw=False)
            self.fit_view()

    def fit_all(self) -> None:
        self._clear_issue_focus()
        self.selected_index = None
        self.update(draw=False)
        self.fit_view(include_overlays=True)

    def fit_view(self, *, draw: bool = True, include_overlays: bool = False) -> None:
        bounds = [self.bodies[index].bounds for index in self.visible_indices if self.bodies[index].bounds is not None]
        if include_overlays and self.show_support:
            part_indices = {self.bodies[index].part_index for index in self.visible_indices}
            bounds.extend(self._support_bounds[index] for index in part_indices if index in self._support_bounds)
        if include_overlays and self.show_path:
            for index in self.visible_indices:
                path = self.bodies[index].fabrication_path
                if path is not None:
                    points = np.asarray(path.wire_points_m)
                    points = points[np.all(np.isfinite(points), axis=1)]
                    if len(points):
                        bounds.append((points.min(axis=0), points.max(axis=0)))
        if bounds:
            minimum = np.min([bound[0] for bound in bounds], axis=0)
            maximum = np.max([bound[1] for bound in bounds], axis=0)
            center = (minimum + maximum) / 2.0
            radius = max(float((maximum - minimum).max()) * 0.53, 1e-9)
            self.ax.set_xlim(center[0] - radius, center[0] + radius)
            self.ax.set_ylim(center[1] - radius, center[1] + radius)
            self.ax.set_zlim(center[2] - radius, center[2] + radius)
        self.ax.set_box_aspect((1, 1, 1))
        for dimension, axis in zip(("x", "y", "z"), (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis)):
            from matplotlib.ticker import ScalarFormatter

            axis.set_label_text(f"{dimension} [m]")
            axis.set_major_formatter(ScalarFormatter(useOffset=False))
            axis.set_tick_params(labelsize=8)
        self._sync_camera_axis_labels()
        if draw:
            self.figure.canvas.draw_idle()

    def update(self, *, draw: bool = True) -> None:
        from matplotlib import colormaps
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        visible = set(self.visible_indices)
        for index, body in enumerate(self.bodies):
            active = index in visible
            color = colormaps["tab10"](index % 10)
            if body.surface is None and active and len(body.valid_faces):
                body.surface = Poly3DCollection(np.asarray(body.mesh.vertices)[body.valid_faces], linewidths=0.4)
                self.ax.add_collection3d(body.surface)
            if body.surface is not None:
                body.surface.set_visible(active)
                body.surface.set_facecolor((*color[:3], 0.0 if self.mode == "Edges only" else self.opacity))
                body.surface.set_edgecolor(
                    (0.0, 0.0, 0.0, 0.0) if self.mode == "Surface" else (0.20, 0.25, 0.32, self.opacity)
                )
            if active and self.show_path and body.path_artist is None and body.fabrication_path is not None:
                points = np.asarray(body.fabrication_path.wire_points_m)
                body.path_artist = self.ax.plot(*points.T, color="#111827", linewidth=1.2)[0]
            if body.path_artist is not None:
                body.path_artist.set_visible(active and self.show_path)
            if active and self.show_topology and body.topology_counts is None:
                self._prepare_topology(body)
            for artist in body.topology_artists:
                artist.set_visible(active and self.show_topology)
        self._update_support(visible)
        self.empty_label.set_visible(self.rendered_face_count == 0)
        selected = [self.bodies[index] for index in self.visible_indices]
        if self.selected_index is None:
            selection = f"All {len(self.bodies)} {'body' if len(self.bodies) == 1 else 'bodies'}"
        else:
            body = selected[0]
            contour = f", contour {body.contour_index}" if body.contour_index is not None else ", routed conductor"
            selection = f"Body {self.selected_index + 1} / {len(self.bodies)} · part {body.part_index}{contour}"
            if body.multiplicity != 1:
                selection += f" · turn multiplicity {body.multiplicity} (stored metadata)"
        self.selection_label.set_text(selection)
        self.summary.set_text("\n".join(self._summary_lines(selected)))
        failure_lines = textwrap.wrap(self._issue_message, 108, break_long_words=False)
        failure_lines.extend(textwrap.wrap(_failure_reasons_text(selected), 108, break_long_words=False))
        self.failure_label.set_text("\n".join(failure_lines))
        # Reserve space above the mesh for the selected bodies' stored reasons.
        # This only changes viewport size, never the physical coordinate bounds.
        mesh_top = 0.875 - 0.019 * max(len(failure_lines) - 1, 0) if failure_lines else 0.92
        self.ax.set_position([0.025, 0.235, 0.70, max(mesh_top - 0.235, 0.30)])
        if draw:
            self.figure.canvas.draw_idle()

    def _update_support(self, visible: set[int]) -> None:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        parts = {self.bodies[index].part_index: self.bodies[index].part for index in visible}
        if self.show_support:
            for part_index, part in parts.items():
                mesh = getattr(part, "mesh", None)
                if part_index in self._support_artists or mesh is None:
                    continue
                vertices = np.asarray(mesh.vertices, dtype=float)
                faces = np.asarray(mesh.faces, dtype=np.int64)
                valid = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
                faces = faces[valid]
                edges = _mesh_edges(faces)[0]
                segments = vertices[edges]
                segments = segments[np.all(np.isfinite(segments), axis=(1, 2))]
                artist = Line3DCollection(segments, colors="#94a3b8", linewidths=0.35, alpha=0.32)
                self.ax.add_collection3d(artist)
                self._support_artists[part_index] = artist
                if len(segments):
                    self._support_bounds[part_index] = (segments.min(axis=(0, 1)), segments.max(axis=(0, 1)))
        for part_index, artist in self._support_artists.items():
            artist.set_visible(self.show_support and part_index in parts)

    def _prepare_topology(self, body: _BodyView) -> None:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        edges, counts = _mesh_edges(body.valid_faces)
        boundary = edges[counts == 1]
        nonmanifold = edges[counts > 2]
        body.topology_counts = (len(boundary), len(nonmanifold))
        artists = []
        for selected, color in ((boundary, "#f97316"), (nonmanifold, "#dc2626")):
            if len(selected):
                artist = Line3DCollection(np.asarray(body.mesh.vertices)[selected], colors=color, linewidths=1.8)
                self.ax.add_collection3d(artist)
                artists.append(artist)
        body.topology_artists = tuple(artists)

    def _summary_lines(self, bodies: Sequence[_BodyView]) -> list[str]:
        vertices = sum(len(body.mesh.vertices) for body in bodies)
        faces = sum(len(body.mesh.faces) for body in bodies)
        omitted = sum(body.omitted_faces for body in bodies)
        lines = [
            f"Stored geometry: {len(bodies):,} bodies · {vertices:,} vertices · {faces:,} triangles; all drawable triangles shown"
        ]
        if omitted:
            lines[0] += f" · {omitted:,} invalid/nonfinite triangles omitted"
        unreferenced = sum(body.unreferenced_vertices for body in bodies)
        if unreferenced:
            lines[0] += f" · {unreferenced:,} vertices outside drawable faces excluded from fit"
        lines.append(
            "Stored checks — manufacturing: "
            + _status_counts(bodies, "manufacturing_ok")
            + "; watertight: "
            + _status_counts(bodies, "watertight")
            + "; mesh collision: "
            + _collision_status_counts(bodies)
        )
        if self.show_path:
            available = sum(body.fabrication_path is not None for body in bodies)
            lines.append(f"Fabricated profile-centroid paths available: {available} / {len(bodies)} bodies")
        if self.show_topology:
            counts = [body.topology_counts for body in bodies if body.topology_counts is not None]
            lines.append(
                f"Displayed connectivity: {sum(count[0] for count in counts)} boundary edges, "
                f"{sum(count[1] for count in counts)} nonmanifold edges; this is not a collision or solid-validity certificate"
            )
        return ["\n".join(textwrap.wrap(line, 147)) for line in lines]

    def disconnect(self) -> None:
        if self._disconnected:
            return
        self._disconnected = True
        for widget in self._widgets:
            widget.disconnect_events()
        self.figure.canvas.mpl_disconnect(self._close_id)
        self.figure.canvas.mpl_disconnect(self._camera_release_id)


def _mesh_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(faces):
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=int)
    edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    return np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)


def _stored_index(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        return None
    return int(value) if np.isfinite(value) and value >= 0 and int(value) == value else None


def _stored_issue_locations(body: _BodyView, body_index: int) -> list[_IssueLocation]:
    data = body.diagnostics
    ring_count = _stored_index(data.get("sweep_ring_count"))
    section_count = _stored_index(data.get("sweep_section_point_count"))
    path = body.fabrication_path
    path_points = np.asarray(getattr(path, "points_m", np.empty((0, 3))), dtype=float)
    path_valid = path_points.ndim == 2 and path_points.shape[1] == 3 and len(path_points) > 1
    if not ring_count and path_valid:
        ring_count = len(path_points)
    if not ring_count:
        return []
    vertices = np.asarray(body.mesh.vertices)
    rings = None
    if (
        section_count
        and section_count >= 3
        and len(vertices) in {ring_count * section_count, ring_count * section_count + 2}
    ):
        rings = vertices[: ring_count * section_count].reshape((ring_count, section_count, 3))
    if rings is None and (not path_valid or len(path_points) != ring_count):
        return []
    closed = _known_bool(data.get("sweep_closed_path", getattr(path, "closed", False))) is True
    output = []
    for flag, index_key, section_key, label, interval in (
        ("sweep_local_fold_violation", "sweep_worst_quad_ring", "sweep_worst_quad_section", "folded quad", True),
        (
            "sweep_profile_jacobian_violation",
            "sweep_profile_jacobian_worst_ring",
            "sweep_profile_jacobian_worst_section",
            "profile Jacobian",
            False,
        ),
        (
            "sweep_final_frame_rotation_violation",
            "sweep_final_frame_rotation_worst_interval",
            None,
            "frame rotation",
            True,
        ),
    ):
        if _known_bool(data.get(flag)) is not True:
            continue
        ring = _stored_index(data.get(index_key))
        if ring is None or ring >= ring_count - int(interval and not closed):
            continue
        section = _stored_index(data.get(section_key)) if section_key is not None else None
        if section is not None and section_count and section >= section_count:
            section = None
        neighborhood = np.arange(ring - 2, ring + 4)
        neighborhood = (
            np.unique(neighborhood % ring_count)
            if closed
            else neighborhood[(neighborhood >= 0) & (neighborhood < ring_count)]
        )
        if rings is not None:
            points = rings[neighborhood].reshape((-1, 3))
            anchor = rings[ring, section] if section is not None else np.mean(rings[ring], axis=0)
            source = "stored mesh rings"
        else:
            points = path_points[neighborhood]
            anchor = path_points[ring]
            source = "stored fabrication path"
        points = points[np.all(np.isfinite(points), axis=1)]
        if not len(points) or not np.all(np.isfinite(anchor)):
            continue
        output.append(_IssueLocation(body_index, label, ring, section, points.copy(), anchor.copy(), source))
    return output


def _format_statuses(values: Sequence[str]) -> str:
    counts = {value: values.count(value) for value in dict.fromkeys(values)}
    return next(iter(counts)) if len(counts) == 1 else ", ".join(f"{key} {count}" for key, count in counts.items())


def _status_counts(bodies: Sequence[_BodyView], key: str) -> str:
    values = [_known_bool(body.diagnostics.get(key)) for body in bodies]
    return _format_statuses(["unknown" if value is None else "yes" if value else "no" for value in values])


def _known_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and np.isfinite(value) and value in (0, 1):
        return bool(value)
    return None


def _failure_reasons_text(bodies: Sequence[_BodyView]) -> str:
    counts: dict[str, int] = {}
    for body in bodies:
        stored = body.diagnostics.get("manufacturing_failure_reasons")
        reasons = (
            list(dict.fromkeys(reason.strip() for reason in stored.split(";") if reason.strip()))
            if isinstance(stored, str)
            else []
        )
        if not reasons and _known_bool(body.diagnostics.get("manufacturing_ok")) is False:
            reasons = ["reasons unavailable"]
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return ""
    if len(bodies) == 1:
        return "Stored manufacturing failures: " + "; ".join(counts)
    return "Stored manufacturing failures (affected bodies): " + "; ".join(
        f"{reason} ({count})" for reason, count in counts.items()
    )


def _collision_status_counts(bodies: Sequence[_BodyView]) -> str:
    values = []
    for body in bodies:
        data = body.diagnostics
        collision = _known_bool(data.get("sweep_mesh_self_intersection"))
        if collision is True:
            values.append("reported")
        elif _known_bool(data.get("sweep_mesh_collision_check_skipped")) is True or data.get(
            "sweep_mesh_collision_broad_phase"
        ) in {
            "not_certified",
            "skipped_after_upstream_failure",
        }:
            values.append("not checked")
        elif collision is False:
            values.append("none reported")
        else:
            values.append("unknown")
    return _format_statuses(values)
