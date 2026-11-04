"""Check closed, consistently oriented copper solids from the actual STL coordinate triples."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def inspect(triangles):
    t = np.asarray(triangles, dtype=float)
    if t.ndim != 3 or t.shape[1:] != (3, 3) or not len(t) or not np.isfinite(t).all():
        raise ValueError("Expected a nonempty finite STL triangle array")
    points, ids = np.unique(t.reshape(-1, 3), axis=0, return_inverse=True)
    f = ids.reshape(-1, 3)
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    pairs, edge_ids, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    directions = np.bincount(edge_ids, weights=np.sign(edges[:, 1] - edges[:, 0]))
    adjacency = coo_matrix((np.ones(len(pairs), dtype=bool), pairs.T), shape=(len(points), len(points)))
    components = connected_components(adjacency, directed=False, return_labels=False)
    areas = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2
    relative = t - np.mean(points, axis=0)
    volume = np.einsum("ij,ij->", relative[:, 0], np.cross(relative[:, 1], relative[:, 2])) / 6
    r = dict(
        vertices=len(points),
        facets=len(t),
        connected_components=int(components),
        open_edges=int(np.count_nonzero(counts == 1)),
        nonmanifold_edges=int(np.count_nonzero(counts > 2)),
        inconsistent_winding_edges=int(np.count_nonzero((counts == 2) & (directions != 0))),
        degenerate_facets=int(np.count_nonzero(areas == 0)),
        minimum_triangle_area_m2=float(areas.min()),
        signed_volume_m3=float(volume),
    )
    r["closed_oriented_solid"] = bool(
        components == 1
        and volume > 0
        and not any(
            r[k] for k in ("open_edges", "nonmanifold_edges", "inconsistent_winding_edges", "degenerate_facets")
        )
    )
    return r
