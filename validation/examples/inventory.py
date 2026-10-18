"""Generate the current runnable-example inventory without importing examples."""

import ast
from hashlib import sha256
import json
from pathlib import Path


def entrypoints(root):
    root = Path(root)
    rows = []
    for directory in ("examples", "projects"):
        for path in sorted((root / directory).rglob("*.py")):
            source = path.read_bytes()
            tree = ast.parse(source, filename=str(path))
            guarded = any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq)
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__"
                for node in tree.body
            )
            if guarded:
                rows.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(source).hexdigest()})
    return rows


def main():
    root = Path(__file__).resolve().parents[2]
    rows = entrypoints(root)
    output = Path(__file__).with_name("current_inventory.json")
    record = {
        "entrypoint_count": len(rows),
        "entrypoints": rows,
        "scope": "Current guarded Python scripts, discovered without importing or executing them. Inventory is not validation.",
    }
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Recorded {len(rows)} runnable entrypoints in {output}")


if __name__ == "__main__":
    main()
