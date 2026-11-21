"""Large winding provenance remains compact, exact and detached after reopening."""

from copy import deepcopy
import json

import numpy as np
import pytest

from coilgen.models import WirePathGeometry, WireSegmentOwner
from coilgen.pipeline_stages import _decode, _encode


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("owners", [False, True])
def test_large_winding_provenance_roundtrips_without_manifest_expansion(container, owners):
    values = (
        [WireSegmentOwner("contour", loop_index=i // 1000) for i in range(70_000)]
        if owners
        else [WirePathGeometry.SURFACE] * 35_000 + [WirePathGeometry.FREE_3D] * 35_000
    )
    arrays = {}
    encoded = _encode(container(values), arrays)
    assert len(json.dumps(encoded)) < 32_000
    actual = _decode(encoded, arrays)
    assert type(actual) is container
    assert actual == container(values)
    if owners:
        actual[0].loop_index = 999
        assert actual[1].loop_index == values[0].loop_index == 0


@pytest.mark.parametrize(
    "indices",
    [np.array([-1], dtype=np.int32), np.array([1], dtype=np.uint32), np.zeros((1, 1), dtype=np.uint32)],
)
def test_provenance_index_corruption_is_rejected(indices):
    arrays = {}
    encoded = _encode([WireSegmentOwner("contour")] * 32, arrays)
    encoded["indices"] = _encode(indices, arrays)
    with pytest.raises(ValueError, match="provenance indices"):
        _decode(encoded, arrays)


def test_provenance_table_cannot_expand_nested_objects():
    arrays = {}
    encoded = _encode([WireSegmentOwner("contour")] * 32, arrays)
    changed = deepcopy(encoded)
    changed["values"] = [{"$list": ["unexpected"]}]
    with pytest.raises(ValueError, match="segment owners or geometry"):
        _decode(changed, arrays)


def test_historical_expanded_provenance_still_decodes():
    owners = [WireSegmentOwner("contour", loop_index=2), WireSegmentOwner("connector", connector_index=3)]
    arrays = {}
    encoded = {"$tuple": [_encode(owner, arrays) for owner in owners]}
    assert _decode(encoded, arrays) == tuple(owners)
