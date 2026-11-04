"""Browser inputs cannot open host files, evaluate Python, or claim another user's upload."""

import asyncio
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pytest

pytest.importorskip("aiohttp")
from aiohttp import FormData  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from coilgen.application.templates import template_assembly  # noqa: E402
from coilgen.web.hub import Hub  # noqa: E402
from coilgen.web.input_policy import (  # noqa: E402
    _numeric_payload_size,
    check_browser_inputs,
    claim_project_upload,
    load_browser_assembly,
    upload_name,
)


def test_advanced_and_model_builder_accept_only_owned_inputs(tmp_path):
    project = template_assembly("shielded_gy", project_root=tmp_path).channels[0].project
    check_browser_inputs(project.values, project.study)
    check_browser_inputs({"coil_mesh_file": project.config.coil_mesh_file}, current_project=project)
    check_browser_inputs({"field_shape_function": "(sin(x) + 2y).^2 + solid_harmonic(2,0,1,x,y,z)"})
    for source in (
        str(tmp_path / "another-user.stl"),
        "../other/asset.stl",
        "create ../../secret.stl",
        "//host/private",
    ):
        with pytest.raises(ValueError, match="owned project asset"):
            check_browser_inputs({"coil_mesh_file": source}, current_project=project)
    for key in ("sf_source_file", "preoptimized_stream_function_file", "target_field_definition_file"):
        with pytest.raises(ValueError, match="owned project asset"):
            check_browser_inputs({key: "C:/another/user/secret.npz"})
    for key in ("output_directory", "geometry_source_path"):
        with pytest.raises(ValueError, match="browser manages"):
            check_browser_inputs({key: "../other"})
    with pytest.raises(ValueError, match="numerical"):
        check_browser_inputs({"field_shape_function": "x.__class__"})
    with pytest.raises(ValueError, match="numerical"):
        check_browser_inputs({}, {"requirements": {"recipe": {"expressions": ["0", "0", "open('x')"]}}})
    with pytest.raises(ValueError, match="only runs"):
        check_browser_inputs({}, {"array_study": {"channels": [{"run_directory": "C:/elsewhere"}]}}, workspace=tmp_path)


def test_browser_opens_inline_json_and_compact_owned_mesh_but_never_loads_legacy_sidecars(tmp_path, monkeypatch):
    project = template_assembly("shielded_gy", project_root=tmp_path).channels[0].project
    for suffix in (".json", ".coilgen"):
        path = project.save(tmp_path / f"shield{suffix}")
        restored = load_browser_assembly(path)
        assert restored.channels[0].project.fingerprint == project.fingerprint
    data = project.to_dict()
    data["schema"] = "coilgen-project-v1"
    for key in data["assets"]:
        data["assets"][key] = {"path": "../other/private.stl", "sha256": key, "source": "", "kind": "mesh"}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(
        "coilgen.application.assets.InputAsset.from_file", lambda *args, **kw: pytest.fail("read sidecar")
    )
    with pytest.raises(ValueError, match="sidecar"):
        load_browser_assembly(path)


def test_live_browser_actions_reject_foreign_stl_and_preserve_valid_import(tmp_path):
    pytest.importorskip("trame")
    from coilgen.web.app import create_app

    stl = (
        b"solid sample\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 10 0 0\nvertex 0 10 0\n"
        b"endloop\nendfacet\nendsolid sample\n"
    )
    other = tmp_path / "owner"
    other.mkdir()
    external = other / "private.stl"
    external.write_bytes(stl)
    server, browser = create_app(tmp_path / "visitor", user="visitor", server_name="test-browser-private-files")
    state = server.state
    try:
        state.template = "cylinder"
        browser.load_template()
        original = browser.session.project.fingerprint
        browser.open_advanced()
        config = json.loads(state.advanced_config)
        config["coil_mesh_file"] = str(external)
        state.advanced_config = json.dumps(config)
        browser.apply_advanced()
        assert "owned project asset" in state.message
        assert browser.session.project.fingerprint == original
        state.settings_values = {"coil_mesh_file": str(external)}
        browser.apply_settings()
        assert "owned project asset" in state.message
        assert browser.session.project.fingerprint == original
        assert browser._open_stored_project(external) is None
        assert "uploaded into this workspace" in state.message
        browser._refresh_settings()
        state.mesh_upload = {"name": "uploaded.stl", "content": stl}
        browser.apply_settings()
        assert browser.session.project.values["coil_mesh_file"].startswith("asset:")
        browser.open_advanced()
        accepted = json.loads(state.advanced_config)
        assert accepted["coil_mesh_file"].startswith("asset:")
        accepted["field_shape_function"] = "cos(x) + sin(y)"
        state.advanced_config = json.dumps(accepted)
        browser.apply_advanced()
        assert browser.session.project.values["field_shape_function"] == "cos(x) + sin(y)"
        assert browser.session.project.values["coil_mesh_file"] == accepted["coil_mesh_file"]
    finally:
        browser.plotter.close()
        browser.session.close()


def test_compact_and_json_expression_validation_precedes_native_study_construction(tmp_path, monkeypatch):
    project = template_assembly("cylinder", project_root=tmp_path).channels[0].project
    original = project.save(tmp_path / "original.coilgen")
    with ZipFile(original) as archive:
        envelope = json.loads(archive.read("manifest.json"))
    envelope["document"]["study"]["requirements"] = {"recipe": {"expressions": ["0", "x.__class__", "0"]}}
    compact = tmp_path / "untrusted.coilgen"
    with ZipFile(compact, "w") as archive:
        archive.writestr("manifest.json", json.dumps(envelope))
    monkeypatch.setattr(
        "coilgen.web.input_policy.load_bundle", lambda *args, **kw: pytest.fail("constructed unsafe model")
    )
    with pytest.raises(ValueError, match="numerical"):
        load_browser_assembly(compact)
    data = project.to_dict()
    data["study"] = envelope["document"]["study"]
    raw = tmp_path / "untrusted.json"
    raw.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="numerical"):
        load_browser_assembly(raw)


def test_numeric_payload_rejects_forged_shapes_before_allocation():
    raw = BytesIO()
    np.save(raw, np.array([1.0]), allow_pickle=False)
    forged = raw.getvalue().replace(b"(1,)", b"(9,)")
    data = BytesIO()
    with ZipFile(data, "w") as zip_:
        zip_.writestr("data.npy", forged)
    with pytest.raises(ValueError, match="dimensions"):
        _numeric_payload_size(data.getvalue())


@pytest.mark.parametrize(
    "name", ("../other.stl", "C:other.json", "nul.stl", "file.stl\r\n", "x\\y.json", "x:stream.json")
)
def test_unsafe_browser_upload_names_are_rejected(name):
    with pytest.raises(ValueError):
        upload_name(name)


def test_authenticated_stream_upload_is_limited_and_claimed_once_in_its_own_workspace(tmp_path, monkeypatch):
    import coilgen.web.input_policy as policy

    hub = Hub(tmp_path, password="team")

    async def start(name):
        return None

    hub.ensure_user = start

    async def scenario():
        async with TestClient(TestServer(hub.application())) as client:

            async def post(url, name="shield.coilgen", data=b"payload", headers=None):
                form = FormData()
                form.add_field("file", data, filename=name, content_type="application/octet-stream")
                return await client.post(url, data=form, headers=headers)

            assert (await post("/u/unclaimed/upload/project")).status == 403
            login = await client.post(
                "/login",
                data={"action": "create", "name": "anna", "password": "team", "passphrase": "Private test passphrase"},
                allow_redirects=False,
            )
            url = login.headers["Location"] + "upload/project"
            assert (await post(url, headers={"Origin": "http://foreign.example"})).status == 403
            assert (await post(url, name="fake.txt")).status == 400
            result = await post(url)
            assert result.status == 200
            ticket = await result.json()
            assert ticket["name"] == "shield.coilgen"
            other = tmp_path / "users" / "other"
            other.mkdir()
            with pytest.raises(ValueError):
                claim_project_upload(other, ticket["token"])
            file = claim_project_upload(tmp_path / "users" / "anna", ticket["token"])
            assert file.read_bytes() == b"payload"
            with pytest.raises(ValueError):
                claim_project_upload(tmp_path / "users" / "anna", ticket["token"])
            monkeypatch.setattr(policy, "PROJECT_UPLOAD_LIMIT", 6)
            assert (await post(url, data=b"1234567")).status == 413

    asyncio.run(scenario())
