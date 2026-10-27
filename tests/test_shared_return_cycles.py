"""Real original biplanar regression; these seeds are never physical certificates."""

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.shared_return_cycles import _clip_return


def _config():
    return CoilGenConfig(inter_group_connection_path_method="direct", progress_flag=False)


def test_return_clipping_keeps_directed_corners_and_does_not_move_source():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 2.0, 0.0]])
    before = source.copy()
    np.testing.assert_array_equal(
        _clip_return(source, 0.1, 0.8), [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]]
    )
    np.testing.assert_array_equal(source, before)
    with pytest.raises(ValueError):
        _clip_return(source, 0.8, 0.1)
