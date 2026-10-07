"""Independent channel provenance without changing the scientific problem."""

import json

import pytest

from coilgen import CoilProject, CylinderCoilDesign
from coilgen.application.assembly import ChannelDocument, ProjectAssembly
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.persistence import atomic_json


def test_same_owned_project_cannot_be_attached_to_two_independent_channels():
    first = ProjectDocument.new("A", preset_config())
    renamed_revision = first.rename("B").update_config({"levels": 12})
    with pytest.raises(ValueError, match=r"unique ProjectDocument ID.*duplicate\(name\)"):
        ProjectAssembly.new("Ambiguous channel ownership", (first, renamed_revision))
    with pytest.raises(ValueError, match="unique ProjectDocument ID"):
        ProjectAssembly(
            "assembly",
            "Set",
            1,
            (ChannelDocument("channel-a", "A", first), ChannelDocument("channel-b", "B", first, False)),
        )


def test_duplicate_establishes_independent_ids_with_identical_config_study_and_fingerprint():
    first = ProjectDocument.new("A", preset_config()).update_study({"validation_resolution": 16})
    independent = first.duplicate("B")
    assembly = ProjectAssembly.new("Set", (first,)).add_channel(independent)
    assert [channel.project.id for channel in assembly.channels] == [first.id, independent.id]
    assert independent.id != first.id
    assert independent.values == first.values
    assert independent.study == first.study
    assert independent.fingerprint == first.fingerprint
    assert independent.model_fingerprint == first.model_fingerprint
    assert first.revision == 2 and independent.revision == 1
    assert independent.update_config({"levels": 12}).id == independent.id


def test_add_and_update_reject_ambiguous_ownership_without_mutating_existing_collection():
    first = ProjectDocument.new("A", preset_config())
    second = first.duplicate("B")
    original = ProjectAssembly.new("Set", (first, second))
    before = original.fingerprint
    with pytest.raises(ValueError, match="unique ProjectDocument ID"):
        original.add_channel(first, name="C")
    with pytest.raises(ValueError, match="unique ProjectDocument ID"):
        original.update_channel(original.channels[1].id, project=first.update_config({"levels": 12}))
    assert original.fingerprint == before
    assert original.channels[1].project.id == second.id


def test_collection_load_rejects_shared_project_ids_without_silently_rewriting_provenance(tmp_path):
    first = ProjectDocument.new("A", preset_config())
    assembly = ProjectAssembly.new("Set", (first, first.duplicate("B")))
    path = assembly.save(tmp_path / "set.json")
    value = json.loads(path.read_text())
    value["channels"][1]["project"] = value["channels"][0]["project"]
    value["channels"][1]["sha256"] = value["channels"][0]["sha256"]
    atomic_json(path, value)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unique ProjectDocument ID"):
        ProjectAssembly.load(path)
    assert path.read_bytes() == before


def test_existing_script_project_conversion_retains_config_and_distinct_channel_ownership():
    script = CoilProject("Axes", template=CylinderCoilDesign())
    script.add_channel("Gx", target_bz_expression="x")
    script.add_channel("Gy", target_bz_expression="y")
    expected = script.resolved_designs()
    assembly = ProjectAssembly.from_coil_project(script)
    assert len({channel.project.id for channel in assembly.channels}) == 2
    for channel in assembly.channels:
        assert channel.project.config.field_shape_function == expected[channel.name].to_config().field_shape_function
        assert tuple(channel.project.config.cylinder_mesh_parameter_list) == tuple(
            expected[channel.name].to_config().cylinder_mesh_parameter_list
        )
