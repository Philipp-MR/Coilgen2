"""Run independent designs in bounded processes with verified restart.

    python examples/research_parallel_study.py

Use research_tolerance_study.py for a repeated fixed-operator solve. Here the
geometries differ, so processes are appropriate and operator reuse is absent.
"""

from dataclasses import replace
import os
from pathlib import Path

from coilgen.research.jobs import RuntimePolicy
from coilgen.research.requests import ResearchRequest
from coilgen.research.studies import ResearchStudy
from research_basic_designs import cylinder_config


if __name__ == "__main__":
    requests = [
        ResearchRequest.from_config(
            replace(cylinder_config(), cylinder_mesh_parameter_list=(0.30, radius, 12, 6, 1, 0, 0, 0)),
            name=f"Cylinder radius {radius:g} m",
            design_margin=0.8,
        )
        for radius in (0.12, 0.14)
    ]
    output = Path(__file__).resolve().parents[1] / "outputs" / "research_parallel"
    policy = RuntimePolicy(max_workers=min(2, os.cpu_count() or 1), threads_per_worker=1)
    result = ResearchStudy(output, policy).run(requests)
    for row in result["candidates"]:
        print(row["name"], row["status"], "reused:", row["reused"], row.get("summary", {}))
    print("Eligible Pareto points:", result["pareto_indices"])
    print("Evidence:", output / "study.json")
