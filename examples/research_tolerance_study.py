"""Reuse one assembled conic problem for a balanced tolerance sweep.

    python examples/research_tolerance_study.py [--no-resume]

Power and magnetic-energy normalization stays fixed at the feasible reference
for tolerance_scale=1. Output records sampled sheet constraints; independently
validate and fabricate selected candidates before making winding decisions.
"""

from dataclasses import replace

if __package__:
    from ._example_support import example_options, example_runtime
    from .research_basic_designs import cylinder_config
else:
    from _example_support import example_options, example_runtime
    from research_basic_designs import cylinder_config


def assemble_cylinder_problem(config, project_root):
    """Build this example's fixed operators without solving or making contours."""
    from coilgen import CoilPart, validate_config
    from coilgen.basis import calculate_basis_functions, calculate_one_ring_by_mesh
    from coilgen.geometry import load_mesh_from_config, split_disconnected_mesh
    from coilgen.optimization import StreamFunctionOptimizationProblem
    from coilgen.physics import calculate_resistance_matrix, calculate_sensitivity_matrix
    from coilgen.target import define_target_field

    validate_config(config, project_root=project_root).raise_for_errors()
    parts = [CoilPart(mesh=mesh) for mesh in split_disconnected_mesh(load_mesh_from_config(config, project_root))]
    target = define_target_field([part.mesh for part in parts], None, None, config)
    calculate_one_ring_by_mesh(parts)
    calculate_basis_functions(parts)
    calculate_sensitivity_matrix(parts, target, config)
    calculate_resistance_matrix(parts, config)
    return StreamFunctionOptimizationProblem.from_parts(parts, target, config)


def main(argv=None) -> None:
    options = example_options("outputs/research_tolerance", argv, study=True)
    with example_runtime(extra="research", required=("numpy", "cvxpy", "scipy", "threadpoolctl")) as root:
        from threadpoolctl import threadpool_limits

        from coilgen.research.convex import ResearchPhysicsSpec
        from coilgen.research.studies import ToleranceStudy

        config = replace(cylinder_config(), retain_heavy_fields=True, low_memory_mode=False)
        with threadpool_limits(1):
            problem = assemble_cylinder_problem(config, root)
        spec = ResearchPhysicsSpec(
            objective="mixture",
            objective_weights={"power": 1, "magnetic_energy": 1},
            reference_tolerance_scale=1.0,
        )
        result = ToleranceStudy(options.output_dir, problem, spec, config, threads=1).run(
            [0.75, 1, 1.5], resume=options.resume
        )
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
        print("Evidence:", options.output_dir / "study.json")
        if any(
            row["status"] != "completed" or not row.get("summary", {}).get("accepted") for row in result["candidates"]
        ):
            raise RuntimeError("A tolerance candidate failed sampled physical acceptance; inspect study.json.")


if __name__ == "__main__":
    main()
