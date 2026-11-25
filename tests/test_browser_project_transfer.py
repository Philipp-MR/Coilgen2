import asyncio
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote, urlsplit

import pytest

pytest.importorskip("trame")
from coilgen.web.app import create_app  # noqa: E402
from coilgen.web.input_policy import stage_project_upload  # noqa: E402


def stage_file(workspace, filename, content):
    file = SimpleNamespace(name="file", filename=filename, read_chunk=AsyncMock(side_effect=[content, b""]))
    reader = SimpleNamespace(next=AsyncMock(side_effect=[file, None]))
    return asyncio.run(stage_project_upload(workspace, reader))["token"]


@pytest.mark.parametrize(("format_", "suffix"), [("bundle", ".coilgen"), ("json", ".coilgen.json")])
def test_browser_download_reopens_through_native_upload(tmp_path, format_, suffix):
    server, browser = create_app(tmp_path, server_name=f"test-native-upload-{format_}")
    template = server.state.trame__template_main
    assert "<coilgen-project-upload" in template and '@uploaded="trigger(' in template
    assert "project_upload" not in template
    browser.session.load_template("cylinder")
    fingerprint = browser.session.assembly.fingerprint
    if format_ == "bundle":
        browser.save_project()
    else:
        browser.save_project(format_)
    saved = tmp_path / "exports" / Path(server.state.project_download).name
    assert saved.name.endswith(suffix) and saved.is_file()
    browser.session.new_project()
    token = stage_file(tmp_path, saved.name, saved.read_bytes())
    browser.open_uploaded_project(token)
    assert server.state.project_open_result["ok"]
    assert server.state.project_open_result["token"] == token
    assert browser.session.assembly.fingerprint == fingerprint
    browser.open_uploaded_project(token)
    assert not server.state.project_open_result["ok"]
    assert browser.session.assembly.fingerprint == fingerprint


def test_invalid_native_upload_preserves_active_design_and_reports_error(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-native-upload-invalid")
    prior = browser.session.load_template("biplanar")
    token = stage_file(tmp_path, "broken.coilgen.json", b"[]")
    browser.open_uploaded_project(token)
    assert browser.session.assembly is prior
    assert not server.state.project_open_result["ok"]
    assert server.state.project_open_result["message"] and server.state.message_color == "error"


def test_download_is_published_when_windows_holds_the_empty_staging_folder(tmp_path, monkeypatch):
    server, browser = create_app(tmp_path, server_name="test-publish-locked-staging")
    browser.session.new_project()
    original = TemporaryDirectory._rmtree
    locked_folders = []

    def cleanup(path, ignore_errors=False, **kwargs):
        if Path(path).parent == browser.exports:
            assert not list(Path(path).iterdir())
            locked_folders.append(path)
            if not ignore_errors:
                raise PermissionError("Windows is holding the empty download staging folder")
        else:
            original(path, ignore_errors=ignore_errors, **kwargs)

    monkeypatch.setattr(TemporaryDirectory, "_rmtree", staticmethod(cleanup))
    try:
        browser.save_project()
        assert locked_folders
        saved = browser.session.project_path
        assert server.state.project_download == f"downloads/{saved.name}"
        assert (browser.exports / saved.name).read_bytes() == saved.read_bytes()
        assert server.state.message_color == "success"
    finally:
        browser.plotter.close()
        browser.session.close()


def test_download_paths_preserve_literal_filename_characters(tmp_path):
    server, browser = create_app(tmp_path, server_name="test-escaped-export-name")
    output = browser.exports / "Gz #3 · 50%.zip"
    browser._offer_download(output, "Download")
    url = urlsplit(server.state.download_url)
    assert not url.fragment and not url.query
    assert unquote(url.path) == f"downloads/{output.name}"


def test_python_download_applies_setup_and_needs_no_companion(tmp_path):
    import runpy
    from coilgen.web.menus import menu_action

    server, browser = create_app(tmp_path, server_name="test-python-download")
    try:
        browser.session.load_template("cylinder")
        state = server.state
        state.setup_values = {**state.setup_values, "cylinder_radius_mm": "93", "validation_resolution": "7"}
        browser.setup._changed()
        asyncio.run(menu_action(browser, "python"))
        assert state.file_dialog and state.file_section == "save"
        script = browser.exports / Path(unquote(state.python_download)).name
        exported = runpy.run_path(str(script))["build_project"]()
        assert exported.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.093)
        assert exported.study["validation_resolution"] == 7
        assert exported.fingerprint == browser.session.project.fingerprint
        assert "Download Python script" in state.trame__template_main
        assert not script.with_suffix(".coilgen.json").exists()
        state.setup_values = {**state.setup_values, "cylinder_radius_mm": "invalid"}
        browser.setup._changed()
        browser.export_python()
        assert not state.python_download
        assert len(list(browser.exports.glob("*.py"))) == 1
    finally:
        browser.plotter.close()
        browser.session.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is optional for client-component tests")
def test_native_upload_javascript_progress_and_errors():
    subprocess.run(
        [shutil.which("node"), "--test", str(Path(__file__).with_name("browser_project_upload.test.cjs"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
