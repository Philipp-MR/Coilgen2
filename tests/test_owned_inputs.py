"""Python configurations own independent inputs without retaining dead designs."""

from copy import deepcopy
from dataclasses import asdict
import gc
import pickle
import weakref

import numpy as np
import pytest

from coilgen import CoilGenConfig, CoilGenPipeline, ParametricSurfaceCoilDesign
from coilgen.models import Mesh, TargetField
from coilgen import object_registry as registry


def test_same_name_surface_designs_keep_independent_geometry():
    first = ParametricSurfaceCoilDesign(surface=lambda u, v: (0.1 * u, 0.1 * v, 0), u_divisions=2, v_divisions=2)
    second = ParametricSurfaceCoilDesign(surface=lambda u, v: (0.2 * u, 0.2 * v, 0), u_divisions=2, v_divisions=2)
    config = first.to_config()
    pipeline = CoilGenPipeline(config)
    before = registry.get_in_memory_mesh(config.coil_mesh_file).vertices
    other = second.to_config()
    assert config.coil_mesh_file != other.coil_mesh_file
    np.testing.assert_array_equal(registry.get_in_memory_mesh(pipeline.config.coil_mesh_file).vertices, before)


def test_owned_inputs_follow_config_copies_and_are_released():
    mesh = Mesh(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
    mesh.metadata = {"nested": {"values": [1]}}
    config = CoilGenConfig(coil_mesh_file=registry.own_in_memory_mesh(mesh))
    key = str(config.coil_mesh_file)
    with pytest.raises(KeyError):
        registry.get_in_memory_target_field(key)
    owner = weakref.ref(registry._OWNED_INPUTS[key])
    copied = deepcopy(config)
    values = asdict(config)
    del config
    gc.collect()
    assert owner() is not None
    mesh.metadata["nested"]["values"].append(2)
    returned = registry.get_in_memory_mesh(key)
    returned.vertices[:] = 5
    returned.metadata["nested"]["values"].append(3)
    assert registry.get_in_memory_mesh(key).metadata == {"nested": {"values": [1]}}
    assert registry.get_in_memory_mesh(key).vertices[0, 0] == 0
    with pytest.raises(ValueError, match="owned"):
        registry.register_in_memory_mesh(mesh, key=key)
    del copied, values
    gc.collect()
    assert owner() is None
    with pytest.raises(KeyError):
        registry.get_in_memory_mesh(key)


def test_target_snapshot_is_detached_and_pickle_restores_owned_reference():
    target = TargetField(
        coords=np.zeros((1, 3)),
        b=np.ones((1, 3)),
        weights=np.ones(1),
        suppressed=np.zeros(1, dtype=bool),
        target_field_group_inds=np.ones(1, dtype=int),
        target_diagnostics={"nested": {"values": [1]}},
    )
    key = registry.own_in_memory_target_field(target)
    with pytest.raises(KeyError):
        registry.get_in_memory_mesh(key)
    restored = pickle.loads(pickle.dumps(key))
    target.b[:] = 9
    target.target_diagnostics["nested"]["values"].append(2)
    result = registry.get_in_memory_target_field(restored)
    np.testing.assert_array_equal(result.b, np.ones((1, 3)))
    assert result.target_diagnostics == {"nested": {"values": [1]}}
    result.target_diagnostics["nested"]["values"].append(3)
    assert registry.get_in_memory_target_field(restored).target_diagnostics == {"nested": {"values": [1]}}
    del restored
    gc.collect()
    np.testing.assert_array_equal(registry.get_in_memory_target_field(key).b, np.ones((1, 3)))


def test_repeated_problem_conversion_releases_mesh_and_target_snapshots():
    from coilgen import CoilDesignProblem, CoilSupportSpec, SupportMesh, TargetFieldSpec

    mesh = Mesh(np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]), np.array([[0, 1, 2]]))
    problem = CoilDesignProblem(
        support=CoilSupportSpec("single_mesh", (SupportMesh(mesh, role="primary"),)),
        target_field=TargetFieldSpec.from_arrays(np.array([[0.0, 0, 0.02]]), np.array([1e-5])),
    )
    owners = []
    for _ in range(20):
        config = problem.to_config()
        for key in (config.coil_mesh_file, config.target_field_definition_file):
            owners.append(weakref.ref(registry._OWNED_INPUTS[str(key)]))
    del config, key
    gc.collect()
    assert all(owner() is None for owner in owners)
