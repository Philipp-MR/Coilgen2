"""Exercise the installed Qt workbench with an isolated VTK framebuffer."""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYVISTA_OFF_SCREEN"] = "true"

from pathlib import Path
import json
import time


def background_workbench(directory, jobs=None):
    import pyvistaqt
    from PySide6.QtWidgets import QApplication
    from pyvista.plotting.render_window_interactor import RenderWindowInteractor
    from vtkmodules.vtkRenderingCore import vtkRenderWindow
    from coilgen.gui.workbench import CoilGenWorkbench

    class PrivateFramebuffer(type(vtkRenderWindow())):
        def SetWindowInfo(self, _qt_id):
            pass

    original = pyvistaqt.QtInteractor

    def interactor(parent):
        surface = PrivateFramebuffer()
        surface.SetOffScreenRendering(1)
        surface.SetShowWindow(False)
        plotter = original(parent=parent, rw=surface, off_screen=True, auto_update=0)
        plotter.iren = RenderWindowInteractor(plotter, interactor=surface.GetInteractor())
        plotter.iren.initialize()
        assert not surface.GetShowWindow()
        return plotter

    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("CoilGen QA")
    app.setApplicationName("Offscreen crossing review")
    assert app.platformName() == "offscreen"
    pyvistaqt.QtInteractor = interactor
    try:
        window = CoilGenWorkbench(jobs_directory=directory, jobs=jobs)
    finally:
        pyvistaqt.QtInteractor = original
    return app, window


def submit():
    from coilgen.application.templates import TEMPLATE_SPECS
    from coilgen.persistence import numerical_source_digest, source_digest
    import coilgen
    import argparse
    from coilgen.research.jobs import JobManager, RuntimePolicy

    source = Path(coilgen.__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--jobs", type=Path, help="Isolated GUI job store (default: OUTPUT/jobs)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compatible-routes", type=Path,
                        help="Optional earlier native job map. Reuse a routed checkpoint only for the identical physical model and numerical source.")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    directory = (args.jobs or args.output / "jobs").resolve()
    info = {
        "source": str(source),
        "source_digest": source_digest(source),
        "numerical_source_digest": numerical_source_digest(source),
        "jobs": str(directory),
    }
    print(info, flush=True)
    (args.output / "snapshot.json").write_text(json.dumps(info, indent=2))
    jobs = JobManager(
        directory, RuntimePolicy(max_workers=4, threads_per_worker=1, max_pending=45, timeout_seconds=7200)
    )
    app, window = background_workbench(directory, jobs)

    def fail(message):
        raise RuntimeError(message)

    window._error = fail
    runs = {}
    reuse = {}
    earlier = json.loads(args.compatible_routes.read_bytes()) if args.compatible_routes else {}
    priority = ("shielded_gy", "biplanar_gz", "shim_z2", "biplanar", "biplanar_gradient_shim", "conical_harmonic")
    if earlier:
        priority = ("biplanar_gradient_shim", "shielded_gy", "biplanar_gz", "biplanar", "shim_z2", "conical_harmonic")
    templates = sorted(
        TEMPLATE_SPECS, key=lambda spec: (priority.index(spec.id) if spec.id in priority else len(priority), spec.id)
    )
    for spec in templates:
        template = spec.id
        if args.only and template not in args.only:
            continue
        assembly = window.load_template(template)
        assert assembly and window.assembly is assembly
        checkpoints = {}
        for i, channel in enumerate(assembly.channels):
            key = f"{template}__{i}"
            if key not in earlier:
                continue
            prior = Path(earlier[key])
            checkpoint = prior / "checkpoints/routed.cgstage"
            if not checkpoint.is_file() or not (prior / "result.json").is_file():
                continue
            if json.loads((prior / "job.json").read_bytes()).get("status") != "completed":
                continue
            request = json.loads((prior / "request.json").read_bytes())
            manifest = json.loads((prior / "manifest.json").read_bytes())
            if (request.get("project_model_fingerprint") == channel.project.model_fingerprint
                and manifest.get("numerical_source_sha256") == info["numerical_source_digest"]):
                checkpoints[channel.id] = checkpoint
                reuse[key] = dict(checkpoint=str(checkpoint), physical_model=channel.project.model_fingerprint,
                                  numerical_source=info["numerical_source_digest"], prior_run=str(prior),
                                  prior_request=manifest["request_sha256"])
        if checkpoints:
            states = {}
            indices = list(range(len(assembly.channels)))
            indices.sort(key=lambda i: assembly.channels[i].id in checkpoints)
            for i in indices:
                channel = assembly.channels[i]
                window.channel_selector.setCurrentIndex(window.channel_selector.findData(channel.id))
                assert window.project.model_fingerprint == channel.project.model_fingerprint
                state = window.compute(resume_checkpoint=str(checkpoints[channel.id]) if channel.id in checkpoints else None)
                assert state is not None, (template, channel.id)
                states[channel.id] = state
        elif len(assembly.channels) > 1:
            states = window.compute_channels()
        else:
            state = window.compute()
            states = {assembly.channels[0].id: state}
        for i, channel in enumerate(assembly.channels):
            state = states[channel.id]
            runs[f"{template}__{i}"] = str(directory / state["id"])
        (args.output / "new_runs.json").write_text(json.dumps(runs, indent=2))
        (args.output / "checkpoint-reuse.json").write_text(json.dumps(reuse, indent=2))
        print("Submitted", template, flush=True)
    previous = None
    while True:
        app.processEvents()
        by_id = {state["id"]: state for state in jobs.list()}
        status = [
            (key, by_id[Path(path).name]["status"], by_id[Path(path).name].get("active_stage"))
            for key, path in runs.items()
        ]
        if previous != status:
            print(status, flush=True)
            previous = status
        if all(item[1] in {"completed", "failed", "cancelled", "timed_out"} for item in status):
            assert all(item[1] == "completed" for item in status), [item for item in status if item[1] != "completed"]
            break
        time.sleep(0.4)
    window.undo_stack.setClean()
    window.close()
    jobs.close()
    app.processEvents()


if __name__ == "__main__":
    submit()
