"""Shared fixtures for interconnection editor."""

from dataclasses import replace
import numpy as np
import pytest
from _group_routing_cases import make_group_routing_case
from coilgen.application.interconnection_editor import WindingEditorSession
from coilgen.layout import cleanup_wire_paths, interconnect_among_groups, interconnect_within_groups
from coilgen.models import CoilPart, ContourLine, Mesh, TargetField, WirePath, WireSegmentOwner
from coilgen.physics import biot_savart_wire_path
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, PipelineState


@pytest.fixture
def routed():
    case = make_group_routing_case("opposed_direction")
    case.config = replace(case.config, joint_group_routing_selection_policy="regular")
    interconnect_within_groups([case.part], case.config)
    interconnect_among_groups([case.part], case.config)
    cleanup_wire_paths([case.part], config=case.config)
    return case


def inner_cut(editor):
    return next(s for s in editor.sections if s.cut is not None and s.owner.loop_index == 1)


def surface_connection(editor):
    return next(s for s in editor.sections if s.editable and {g.value for g in s.geometry} == {"surface"})


def simple_session(tmp_path):
    from coilgen.config import CoilGenConfig

    angles = np.linspace(0, 2 * np.pi, 97)
    points = 0.02 * np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    points[-1] = points[0]
    mesh = Mesh(
        np.array([[-0.06, -0.06, 0], [0.06, -0.06, 0], [0.06, 0.06, 0], [-0.06, 0.06, 0]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    source = ContourLine(points, 1)
    part = CoilPart(
        mesh=mesh,
        stream_function=np.zeros(4),
        contour_lines=[source],
        contour_step=1.0,
        wire_path=WirePath(points, [WireSegmentOwner("contour_loop", loop_index=0)] * (len(points) - 1)),
    )
    coords = np.array([[0, 0, 0.02], [0.002, 0, 0.02], [-0.002, 0, 0.02]])
    field = biot_savart_wire_path(points, coords)
    target = TargetField(coords, field, np.ones(3), np.zeros(3, dtype=bool), np.zeros(3, dtype=int))
    config = CoilGenConfig(
        surface_is_cylinder_flag=False,
        skip_sweep=False,
        skip_normal_shift=True,
        conductor_cross_section_width=0.0003,
        conductor_cross_section_height=0.0003,
        sweep_min_bend_radius=0.001,
        sweep_max_segment_length=0.001,
        sweep_self_intersection_check="capsule_fast",
    )
    checkpoint = PipelineCheckpoint(
        PipelineStage.ROUTED,
        PipelineState(parts=[part], target_field=target, combined_mesh=mesh, sf_b_field=field),
        config,
        tmp_path,
    )
    return WindingEditorSession(checkpoint), checkpoint
