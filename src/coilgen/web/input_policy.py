"""Browser file transfer and input validation before native code sees user-supplied sources."""

from __future__ import annotations

import ast
import asyncio
from io import BytesIO
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import secrets
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile

from numpy.lib import format as npy

from ..application.assembly import ASSEMBLY_SCHEMA, LEGACY_ASSEMBLY_SCHEMAS, ProjectAssembly
from ..application.bundles import SOURCE_PROJECT_FILE
from ..application.portable import _MAX_MANIFEST_BYTES, is_bundle, load_bundle
from ..application.project import _INPUT_FIELDS, _json_object, ProjectDocument
from ..geometry import geometry_factory
from ..target import _ALLOWED_FUNCTIONS, _normalize_field_expression

PROJECT_UPLOAD_LIMIT = 512 * 1024**2
_EXPANDED_LIMIT = 2 * 1024**3
_TOKEN_TTL_SECONDS = 3600


def upload_name(name):
    if (
        not isinstance(name, str)
        or not 0 < len(name) <= 200
        or name in {".", ".."}
        or PureWindowsPath(name).is_reserved()
        or name.endswith((" ", "."))
        or any(ord(char) < 32 or char in '/\\:<>"|?*' for char in name)
    ):
        raise ValueError("Use a plain upload filename without directories, device names or control characters.")
    return name


def _upload_root(workspace):
    root = Path(workspace).resolve() / "uploads"
    if root.resolve() != root:
        raise ValueError("Invalid uploads directory.")
    root.mkdir(exist_ok=True)
    return root


def checked_upload_path(workspace, path):
    root = _upload_root(workspace)
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(root) or path.resolve() != path or not path.is_file():
        raise ValueError("Choose a file uploaded into this workspace.")
    return path


async def stage_project_upload(workspace, reader):
    part = await reader.next()
    if part is None or getattr(part, "name", None) != "file":
        raise ValueError("Choose one project in the file field.")
    name = upload_name(part.filename)
    if not name.lower().endswith((".json", ".coilgen")):
        raise ValueError("Choose a .json or .coilgen project.")
    root = _upload_root(workspace)
    ticket_root = root / ".pending"
    if ticket_root.resolve() != ticket_root:
        raise ValueError("Invalid upload ticket directory.")
    ticket_root.mkdir(exist_ok=True)
    for stale in ticket_root.iterdir():
        try:
            if (
                re.fullmatch(r"[a-f0-9]{48}", stale.name)
                and stale.is_dir()
                and stale.resolve() == stale
                and stale.stat().st_mtime < time.time() - _TOKEN_TTL_SECONDS
            ):
                _remove_ticket(stale)
        except FileNotFoundError:
            pass  # Another request claimed or expired the ticket first.
    token = secrets.token_hex(24)
    with TemporaryDirectory(dir=root) as staging:
        folder = Path(staging) / "project"
        folder.mkdir()
        size = 0
        with (folder / name).open("xb") as stream:
            while chunk := await part.read_chunk(size=256 * 1024):
                size += len(chunk)
                if size > PROJECT_UPLOAD_LIMIT:
                    raise OverflowError("Project exceeds the 512 MiB upload limit.")
                await asyncio.to_thread(stream.write, chunk)
        if not size:
            raise ValueError("The uploaded project is empty.")
        if await reader.next() is not None:
            raise ValueError("Send one project file per upload.")
        os.rename(folder, ticket_root / token)
    return {"token": token, "name": name}


def _remove_ticket(folder):
    for item in folder.iterdir():
        if item.is_file() and item.resolve() == item:
            item.unlink(missing_ok=True)
    folder.rmdir()


def claim_project_upload(workspace, token):
    if not isinstance(token, str) or not re.fullmatch(r"[a-f0-9]{48}", token):
        raise ValueError("Invalid project upload token.")
    root = _upload_root(workspace)
    pending = root / ".pending" / token
    if (
        pending.resolve() != pending
        or not pending.is_dir()
        or pending.stat().st_mtime < time.time() - _TOKEN_TTL_SECONDS
    ):
        raise ValueError("This upload has expired, was already opened, or belongs to a different workspace.")
    claimed = root / secrets.token_hex(24)
    try:
        os.rename(pending, claimed)
    except FileNotFoundError as exc:
        raise ValueError("This upload has already been opened.") from exc
    files = list(claimed.iterdir())
    if len(files) != 1:
        raise ValueError("Incomplete project upload.")
    return checked_upload_path(workspace, files[0])


_NUMERIC_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.Invert,
)


def _check_expression(expression):
    if not isinstance(expression, str) or len(expression) > 8192:
        raise ValueError("Enter a numerical target expression of at most 8192 characters.")
    try:
        nodes = ast.walk(ast.parse(_normalize_field_expression(expression), mode="eval"))
        for node in nodes:
            if (
                not isinstance(node, _NUMERIC_NODES)
                or isinstance(node, ast.Name)
                and node.id not in {"x", "y", "z", *_ALLOWED_FUNCTIONS}
                or isinstance(node, ast.Call)
                and (not isinstance(node.func, ast.Name) or not callable(_ALLOWED_FUNCTIONS.get(node.func.id)))
                or isinstance(node, ast.Constant)
                and (type(node.value) not in (int, float) or not math.isfinite(node.value))
            ):
                raise ValueError
    except (SyntaxError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError(
            "Use x, y, z, finite numbers, arithmetic and supported numerical functions in target expressions."
        ) from exc


def check_browser_inputs(config, study=None, *, current_project=None, workspace=None):
    """Allow owned assets/generated supports and numeric recipes only, before materialization or preview."""
    for key in _INPUT_FIELDS.intersection(config):
        value = config[key]
        if isinstance(value, str) and (
            value.lower() in ("", "none")
            or re.fullmatch(r"asset:[a-f0-9]{64}", value)
            or key.endswith("mesh_file")
            and geometry_factory(value) is not None
        ):
            continue
        # The shared model builder displays the *current* owned asset's private materialization.
        if (
            current_project
            and str(current_project.values[key]).startswith("asset:")
            and value == getattr(current_project.config, key)
        ):
            continue
        raise ValueError(
            f"{key} requires an owned project asset or generated support. Use the upload controls for files."
        )
    for key in ("geometry_source_path", "output_directory"):
        if key in config and config[key] not in (None, "", "."):
            raise ValueError(f"The browser manages {key}. Set it to null in portable projects.")
    if "field_shape_function" in config:
        _check_expression(config["field_shape_function"])
    if study:
        _check_recipes(study)
        array = study.get("array_study") if isinstance(study, dict) else None
        if isinstance(array, dict):
            for channel in array.get("channels", ()):
                source = channel.get("run_directory") if isinstance(channel, dict) else None
                if not isinstance(source, str) or workspace is None:
                    raise ValueError("An array study may reference only runs in this browser workspace.")
                path = Path(source)
                roots = (Path(workspace) / "jobs", Path(workspace) / "embedded-runs")
                if (
                    path.parent not in roots
                    or not re.fullmatch("[0-9a-f]{32}|[0-9a-f]{64}", path.name)
                    or path.resolve() != path
                    or not path.is_dir()
                ):
                    raise ValueError("An array study may reference only runs in this browser workspace.")


def _check_recipes(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "expressions":
                if not isinstance(item, (list, tuple)):
                    raise ValueError("Target expressions must be a list of numerical expressions.")
                for expression in item:
                    _check_expression(expression)
            elif key == "field_shape_function":
                _check_expression(item)
            else:
                _check_recipes(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                _check_recipes(item)


def _check_document(value, workspace):
    if not isinstance(value, dict):
        raise ValueError("A project upload must contain a project or channel assembly.")
    if "channels" in value:
        for channel in value["channels"]:
            if not isinstance(channel, dict) or not isinstance(channel.get("project"), dict):
                raise ValueError("Upload a self-contained project with its channels embedded, without sidecar files.")
            _check_document(channel["project"], workspace)
        return
    config = value.get("config")
    if not isinstance(config, dict) or not isinstance(value.get("assets"), dict):
        raise ValueError("Upload a self-contained CoilGen project or channel assembly.")
    if any(not isinstance(asset, dict) or "path" in asset for asset in value["assets"].values()):
        raise ValueError("Upload a self-contained project with embedded asset data, without sidecar files.")
    check_browser_inputs(config, value.get("study"), workspace=workspace)


def _numeric_payload_size(data):
    """Reject forged array dimensions before NumPy allocates a mesh or scientific result."""
    with ZipFile(BytesIO(data)) as archive:
        size = sum(item.file_size for item in archive.infolist())
        if size > _EXPANDED_LIMIT:
            raise ValueError("Expanded numerical data exceeds the 2 GiB browser limit.")
        for item in archive.infolist():
            if not item.filename.endswith(".npy"):
                continue
            with archive.open(item) as stream:
                version = npy.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("Unsupported browser numerical array format.")
                header = npy.read_array_header_1_0 if version == (1, 0) else npy.read_array_header_2_0
                shape, _, dtype = header(stream)
                if (
                    dtype.hasobject
                    or dtype.itemsize < 1
                    or any(n < 0 for n in shape)
                    or math.prod(shape) * max(8, dtype.itemsize) > _EXPANDED_LIMIT
                    or math.prod(shape) * dtype.itemsize != item.file_size - stream.tell()
                ):
                    raise ValueError("Untrusted numerical array dimensions, type, or payload length.")
    return size


def load_browser_assembly(path, *, workspace=None):
    path = Path(path)
    if path.stat().st_size > PROJECT_UPLOAD_LIMIT:
        raise ValueError("Project exceeds the 512 MiB browser file limit.")
    if is_bundle(path):
        # A saved study can evaluate an analytic recipe during its native construction.
        # Inspect only the bounded manifest first; the native loader verifies all blobs and hashes.
        with ZipFile(path) as archive:
            info = archive.getinfo("manifest.json")
            if info.file_size > _MAX_MANIFEST_BYTES:
                raise ValueError("Compact project manifest exceeds the supported size.")
            envelope = json.loads(archive.read(info), object_pairs_hook=_json_object)
            _check_document(envelope["document"], workspace)
        document = load_bundle(path, max_bytes=_EXPANDED_LIMIT)
    else:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_json_object)
        _check_document(value, workspace)
        loader = (
            ProjectAssembly if value.get("schema") in {ASSEMBLY_SCHEMA, *LEGACY_ASSEMBLY_SCHEMAS} else ProjectDocument
        )
        document = loader.from_dict(value, project_root=path.parent)
    assembly = document if isinstance(document, ProjectAssembly) else ProjectAssembly.new(document.name, (document,))
    if not assembly.channels:
        raise ValueError("This project has no channels.")
    expanded = 0
    for channel in assembly.channels:
        for asset in channel.project._assets.values():
            if asset.suffix == ".npz":
                expanded += _numeric_payload_size(asset.data)
        for run in channel.project.saved_runs:
            _check_recipes(run.metadata)
            for name, payload in run._files:
                if name.lower().endswith(".npz"):
                    expanded += _numeric_payload_size(payload)
            if expanded > _EXPANDED_LIMIT:
                raise ValueError("Expanded numerical data exceeds the 2 GiB browser limit.")
            # The optional source-project companion can later be opened as part of a CAD export.
            source = dict(run._files).get(SOURCE_PROJECT_FILE)
            if source is not None:
                companion = json.loads(source, object_pairs_hook=_json_object)
                _check_document(companion, workspace)
                if companion.get("results"):
                    raise ValueError("Archived source companions must contain only their model inputs.")
    if expanded > _EXPANDED_LIMIT:
        raise ValueError("Expanded numerical data exceeds the 2 GiB browser limit.")
    return assembly
