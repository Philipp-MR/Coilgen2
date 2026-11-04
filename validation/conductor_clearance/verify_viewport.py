"""Reopen each original source revision and its result in the native offscreen viewport."""

import json
from pathlib import Path
import tempfile
import time
import numpy as np
from offscreen_gui import background_workbench
from coilgen.application.session_evidence import revision_path, revision_key
from coilgen.application.evidence import path_array_names


def main():
    import sys

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    runs = json.loads((folder / "new_runs.json").read_bytes())
    app, window = background_workbench(Path(tempfile.mkdtemp(prefix="coilgen-crossing-viewport-")) / "jobs")
    window.show()
    app.processEvents()
    receipts = {}
    while len(receipts) < len(runs):
        for key, path in runs.items():
            if key in receipts or not (folder / "cases" / (key + ".json")).exists():
                continue
            run = Path(path)
            metadata = json.loads((run / "result.json").read_bytes())
            original = revision_path(run.parents[1], revision_key(metadata))
            window.open_project(original)
            assert window.project.fingerprint == metadata["project_fingerprint"]
            window.load_run(run)
            deadline = time.monotonic() + 60
            while window.viewport._preview_worker is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.02)
            app.processEvents()
            view = window.viewport
            initial = view.dataset.currentData()
            if initial != "review_conductor":
                index = view.dataset.findData("review_conductor")
                assert index >= 0
                view.dataset.setCurrentIndex(index)
                app.processEvents()
            assert view._metadata["request_sha256"] == metadata["request_sha256"]
            assert view.dataset.currentData() == "review_conductor"
            names = path_array_names(view._arrays, "review_conductor")
            assert names and len(names) == len(metadata["fabrication"]["paths"])
            rendered = [mesh.points for mesh in view._fit_meshes]
            points = [view._arrays[k] for k in names]
            assert all(any(np.array_equal(p, m) for m in rendered) for p in points)
            assert sum(value == "winding" for value in view._actors.values()) == len(names)
            receipts[key] = dict(
                request_sha256=metadata["request_sha256"],
                platform=app.platformName(),
                initial_dataset=initial,
                verified_dataset=view.dataset.currentData(),
                point_counts=[len(p) for p in points],
                result_badge=window.results_overview.badge.text(),
            )
            (folder / "native_viewport.json").write_text(json.dumps(receipts, indent=2), encoding="utf8")
            print("Verified native displayed winding", key, flush=True)
        if len(receipts) < len(runs):
            app.processEvents()
            time.sleep(2)
    window.undo_stack.setClean()
    window.close()
    app.processEvents()


if __name__ == "__main__":
    main()
