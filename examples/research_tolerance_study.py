"""Reuse one assembled conic problem for a balanced tolerance sweep.

    python examples/research_tolerance_study.py

Power and magnetic-energy normalization stays fixed at the feasible reference
for tolerance_scale=1. Output records sampled sheet constraints; independently
validate and fabricate selected candidates before making winding decisions.
"""

from dataclasses import replace
from pathlib import Path

from threadpoolctl import threadpool_limits

from coilgen import run_coilgen
from coilgen.optimization import StreamFunctionOptimizationProblem
from coilgen.research.convex import ResearchPhysicsSpec
from coilgen.research.studies import ToleranceStudy
from research_basic_designs import cylinder_config


def main() -> None:
    config = replace(cylinder_config(), retain_heavy_fields=True, low_memory_mode=False)
    with threadpool_limits(1):
        seed = run_coilgen(config)
        problem = StreamFunctionOptimizationProblem.from_parts(seed.coil_parts, seed.target_field, config)
    spec = ResearchPhysicsSpec(
        objective="mixture",
        objective_weights={"power": 1, "magnetic_energy": 1},
        reference_tolerance_scale=1.0,
    )
    output = Path(__file__).resolve().parents[1] / "outputs" / "research_tolerance"
    result = ToleranceStudy(output, problem, spec, config, threads=1).run([0.75, 1, 1.5])
    print("Fixed objective scales:", result["normalization_scales"])
    for row in result["candidates"]:
        print(
            "Tolerance scale:",
            row["tolerance_scale"],
            row["status"],
            "reused:",
            row["reused"],
            "metrics:",
            row.get("summary", {}),
        )
    print("Evidence:", output / "study.json")


if __name__ == "__main__":
    main()
