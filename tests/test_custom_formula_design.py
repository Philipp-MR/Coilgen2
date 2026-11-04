"""Exercise authored formulas through saved projects, job requests and a meshed Gy solve."""

from dataclasses import replace
import json
import time

import numpy as np
import pytest

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.research.jobs import JobManager, RuntimePolicy


def test_cylindrical_gy_custom_power_and_nonlinear_objectives_reach_the_worker(tmp_path):
    config = replace(
        preset_config(),
        cylinder_mesh_parameter_list=(0.3, 0.12, 12, 6, 1, 0, 0, 0),
        target_gradient_strength=0.01,
        target_region_radius=0.045,
        target_region_resolution=8,
        target_error_tolerance_percent=5,
        gauss_order=1,
        skip_inductance_calculation=True,
    )
    baseline = ProjectDocument.from_config(config, name="Custom Gy optimization")
    constraint = "rms(residual_T) <= 1.5e-5"
    formula = {"objective": "custom", "custom_solver": "auto", "custom_constraints": [constraint]}
    specs = {
        "builtin": {"objective": "power"},
        "conic": {**formula, "custom_objective": "power_W"},
        "local": {**formula, "custom_objective": "power_W + 100 * (1 - cos(rms(residual_T) / 1e-5))"},
    }
    power, residual, request_digests, fingerprints = {}, {}, set(), set()
    runtime = tmp_path / "jobs"
    with JobManager(runtime, RuntimePolicy(timeout_seconds=120)) as jobs:
        for case, spec in specs.items():
            project = baseline.update_config({"research_physics": spec})
            restored = ProjectDocument.load(project.save(tmp_path / f"{case}.coilgen.json"))
            assert restored.fingerprint == project.fingerprint
            submitted = DesignService.submit(
                restored,
                jobs,
                pipeline_stage="sheet",
                fabricate=False,
                validation_resolution=5,
                design_margin=1.0,
                kernel_backend="reference",
            )
            directory = runtime / submitted["id"]
            deadline = time.monotonic() + 135
            while time.monotonic() < deadline:
                state = jobs.get(submitted["id"])
                if state["status"] not in {"queued", "running", "cancelling"}:
                    break
                time.sleep(0.05)
            assert state["status"] == "completed", state
            record = DesignService.load_run(directory)
            assert record.metadata["request_sha256"] == submitted["request_sha256"]
            assert record.metadata["project_fingerprint"] == restored.fingerprint
            request_digests.add(submitted["request_sha256"])
            fingerprints.add(restored.fingerprint)
            retained = json.loads((directory / "request.json").read_text())["config"]["research_physics"]
            assert all(retained[key] == value for key, value in spec.items())
            diagnostics = json.loads((directory / "solver_diagnostics.json").read_text())
            measured = record.arrays["sheet_field_T"][:, 2]
            target = record.arrays["target_field_T"][:, 2]
            rms = float(np.sqrt(np.mean((measured - target) ** 2)))
            assert 0 < rms < 1.5e-5
            residual[case] = rms
            power[case] = record.metadata["summary"]["power_w"]
            assert power[case] > 0
            if case != "builtin":
                assert diagnostics["research_custom_solver_used"] == case
                assert diagnostics["research_custom_constraints"] == [constraint]
                objective = power[case] if case == "conic" else power[case] + 100 * (1 - np.cos(rms / 1e-5))
                assert diagnostics["research_custom_objective_value"] == pytest.approx(objective, rel=1e-8)
                assert diagnostics["constrained_solver_contract_met"]
    assert len(request_digests) == len(fingerprints) == 3
    assert power["conic"] == pytest.approx(power["builtin"], rel=1e-4)
    assert residual["local"] < 0.8 * residual["builtin"] and power["local"] > power["builtin"]
