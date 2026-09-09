from __future__ import annotations

# Support direct terminal/Spyder execution from any working directory.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    for _path in (_Path(__file__).resolve().parents[3], _Path(__file__).resolve().parents[3] / "src"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

from projects._shared.object_parameters import object_parameters

import csv
from pathlib import Path
from typing import Any

from projects._shared.example_utils import (
    PROJECT_ROOT,
    example_output_dir,
    native_thread_budget,
    parse_example_parameters,
)
from projects._shared.paths import resolve_project_path as resolve_shared_project_path

from coilgen import CoilGenConfig, load_target_field_definition, target_field_symmetry_audit


from projects.general_examples.parameters import (
    AUDIT_CSV_TARGET_FIELD_SYMMETRY_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS,
)


################################################################################
# USER SETTINGS - EDIT THIS BLOCK FIRST IN SPYDER
################################################################################
PARAMETERS = object_parameters(
    {
        "target": {
            # Example default expects that csv_target_field.py has already been run.
            "file": "latest_outputs/general_examples/csv_target_field/y_gradient_target.csv",
            "create_demo_csv_if_missing": True,
            "field_name": "lin_y",
            "target_gradient_strength": 1.0,
        },
        "symmetry": {
            # A y-gradient target is mirror-even across x=0.
            "plane_point_m": (0.0, 0.0, 0.0),
            "plane_normal": (1.0, 0.0, 0.0),
            "coordinate_tolerance_m": None,
            "field_relative_tolerance": 1e-6,
            "component": "Bz",
        },
        "output": {
            "name": "target_field_symmetry_audit",
            "csv_filename": "target_field_symmetry_audit.csv",
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict[str, Any]:
    return parse_example_parameters(
        PARAMETERS,
        DEFAULT_PARAMETERS,
        example_name="audit_csv_target_field_symmetry",
    )


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    target_file = resolve_project_path(parameters["target"]["file"])
    if not target_file.exists() and bool(parameters["target"]["create_demo_csv_if_missing"]):
        write_demo_y_gradient_csv(target_file)
    config = CoilGenConfig(
        target_field_definition_field_name=parameters["target"]["field_name"],
        target_gradient_strength=parameters["target"]["target_gradient_strength"],
    )
    target = load_target_field_definition(target_file, None, config)
    audit = target_field_symmetry_audit(
        target,
        parameters["symmetry"]["plane_point_m"],
        parameters["symmetry"]["plane_normal"],
        coordinate_tolerance_m=parameters["symmetry"]["coordinate_tolerance_m"],
        field_relative_tolerance=parameters["symmetry"]["field_relative_tolerance"],
        component=parameters["symmetry"]["component"],
    )
    output_dir = example_output_dir(parameters["output"]["name"])
    csv_path = output_dir / parameters["output"]["csv_filename"]
    write_audit_csv(csv_path, audit)
    print_audit(audit, target_file, csv_path)
    print("Spyder variables available after run: target, audit, csv_path")
    return target, audit, csv_path


def resolve_project_path(path: str | Path) -> Path:
    return resolve_shared_project_path(PROJECT_ROOT, "general_examples", path)


def write_audit_csv(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("quantity", "value"))
        for key, value in audit.items():
            writer.writerow((key, value))


def write_demo_y_gradient_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    axis = (-0.04, -0.02, 0.0, 0.02, 0.04)
    lines = ["x_m,y_m,z_m,lin_y,weight,suppressed,group,dbz_dx,dbz_dy,dbz_dz"]
    for x in axis:
        for y in axis:
            for z in axis:
                if (x * x + y * y + z * z) ** 0.5 <= 0.04:
                    lines.append(f"{x:.12g},{y:.12g},{z:.12g},{y:.12g},1,false,1,0,1,0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_audit(audit: dict[str, Any], target_file: Path, csv_path: Path) -> None:
    status = "PASS" if audit["target_symmetry_ok"] else "FAIL"
    print("Target-field symmetry audit")
    print(f"target file: {target_file}")
    print(f"status: {status}")
    print(f"samples: {audit['target_symmetry_sample_count']}")
    print(f"component: {audit['target_symmetry_component']}")
    print(f"selected parity: {audit.get('target_symmetry_parity', 'unknown')}")
    print(f"coordinate max error: {audit.get('target_symmetry_max_coordinate_error_m', float('nan')):.6g} m")
    print(f"selected parity RMS error: {audit.get('target_symmetry_selected_normalized_rms', float('nan')):.6g}")
    reason = str(audit.get("target_symmetry_reason", ""))
    if reason:
        print(f"reason: {reason}")
    print(f"saved audit CSV: {csv_path}")


if __name__ == "__main__":
    with native_thread_budget():
        target, audit, csv_path = main()
