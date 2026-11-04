"""Intermediate display geometry must precede, and never stand in for, a recovery checkpoint."""

from dataclasses import replace

import numpy as np

from coilgen import CoilGenConfig
from coilgen.contours import extract_contours
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from coilgen.models import CoilPart, TargetField
from coilgen.pipeline import CoilGenPipeline
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineState


def test_resumed_cylinder_winding_publishes_each_feature_before_the_routed_boundary(tmp_path):
    support = normalize_mesh(build_cylinder_mesh(0.4, 0.2, 8, 4, 1, 0, 0, 0))
    part = CoilPart(support, stream_function=support.vertices[:, 2].copy())
    part.potential_level_list = np.array([-0.03, 0.03])
    part.contour_step = 0.06
    extract_contours([part], CoilGenConfig(min_point_loop_number=4))
    target = TargetField([[0, 0, 0.03]], [[0, 0, 1e-5]], [1], [False], [0])
    state = PipelineState(parts=[part], target_field=target, combined_mesh=support, sf_b_field=target.b)
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        field_shape_function="y",
        skip_postprocessing=False,
        skip_sweep=False,
        joint_group_routing=True,
        joint_group_routing_max_candidates=1,
        sweep_self_intersection_check="off",
        sweep_surface_realization="mesh",
        progress_flag=False,
    )
    checkpoint = PipelineCheckpoint("contours", state, config, tmp_path)
    events = []

    def feature(name, candidate):
        events.append(name)
        geometry = candidate.parts[0]
        assert geometry.wire_path is not None and len(geometry.wire_path.v) >= 2
        if name == "interconnections":
            assert geometry.fabrication_path is None
        if name == "realized_centerlines":
            assert len(geometry.fabrication_path.wire_points_m) >= 2
        if name == "sweep":
            assert len(geometry.swept_body.faces) > 0
        else:
            assert geometry.swept_body is None

    CoilGenPipeline(config, tmp_path).run_stage(
        "routed",
        checkpoint,
        resume_callback=lambda saved: events.append(saved.stage.value),
        feature_callback=feature,
        checkpoint_callback=lambda saved: events.append(saved.stage.value),
    )
    assert events == ["contours", "interconnections", "realized_centerlines", "sweep", "routed"]
    assert checkpoint.parts[0].wire_path is None

    # A connected-wire study with sweeping disabled has no body to display.
    events.clear()
    wires_only = replace(config, skip_sweep=True)
    CoilGenPipeline(wires_only, tmp_path).run_stage(
        "routed", PipelineCheckpoint("contours", state, wires_only, tmp_path), feature_callback=feature
    )
    assert events == ["interconnections"]
