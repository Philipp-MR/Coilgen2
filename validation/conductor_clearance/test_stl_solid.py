import numpy as np

from stl_solid import inspect


def test_real_stl_coordinate_connectivity_orientation_and_missing_face():
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tetra = p[[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]]
    good = inspect(tetra)
    assert good["closed_oriented_solid"] and np.isclose(good["signed_volume_m3"], 1 / 6)
    assert inspect(tetra[:-1])["open_edges"] == 3
    reversed_face = tetra.copy()
    reversed_face[0] = reversed_face[0, ::-1]
    assert inspect(reversed_face)["inconsistent_winding_edges"] == 3
    assert not inspect(tetra[:, ::-1])["closed_oriented_solid"]
