"""Run independent designs in bounded processes with verified restart.

    python examples/research_parallel_study.py [--no-resume]

Use research_tolerance_study.py for a repeated fixed-operator solve. Here the
geometries differ, so processes are appropriate and operator reuse is absent.
"""

from dataclasses import replace
import os

if __package__:
    from ._example_support import example_options, example_runtime
    from .research_basic_designs import cylinder_config
else:
    from _example_support import example_options, example_runtime
    from research_basic_designs import cylinder_config


def main(argv=None):
    options = example_options("outputs/research_parallel", argv, study=True)
    with example_runtime(extra="research", required=("numpy", "cvxpy", "scipy", "threadpoolctl")):
        from coilgen.research.jobs import RuntimePolicy
        from coilgen.research.requests import ResearchRequest
        from coilgen.research.studies import ResearchStudy

        requests = [
            ResearchRequest.from_config(
                replace(cylinder_config(), cylinder_mesh_parameter_list=(0.30, radius, 12, 6, 1, 0, 0, 0)),
                name=f"Cylinder radius {radius:g} m",
                design_margin=0.8,
            )
            for radius in (0.12, 0.14)
        ]
        policy = RuntimePolicy(max_workers=min(2, os.cpu_count() or 1), threads_per_worker=1)
        result = ResearchStudy(options.output_dir, policy).run(requests, resume=options.resume)
        for row in result["candidates"]:
            print(row["name"], row["status"], "reused:", row["reused"], row.get("summary", {}))
        print("Eligible Pareto points:", result["pareto_indices"])
        print("Evidence:", options.output_dir / "study.json")
        if any(
            row["status"] != "completed" or not row.get("summary", {}).get("accepted") for row in result["candidates"]
        ):
            raise RuntimeError("A parallel candidate failed independent physical acceptance; inspect study.json.")


if __name__ == "__main__":
    main()
