"""Content signatures for mutable result data used by plot diagnostics."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np


def diagnostic_fingerprint(*values: Any) -> str:
    """Hash numeric content and parameters without retaining result snapshots.

    The linear scan also detects in-place edits of equal-sized arrays. It is
    deliberately cheaper than rebuilding spatial/interpolation operators, and
    avoids requiring callers to remember a manual revision counter.
    """

    digest = hashlib.blake2b(digest_size=20)

    def update(value: Any) -> None:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(repr((array.shape, array.dtype.str)).encode())
            if array.dtype.hasobject:
                update(array.tolist())
            elif array.size:
                digest.update(memoryview(array).cast("B"))
        elif isinstance(value, Mapping):
            digest.update(b"mapping")
            for key in sorted(value, key=str):
                update(key)
                update(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(f"sequence:{len(value)}".encode())
            for item in value:
                update(item)
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")

    for value in values:
        update(value)
    return digest.hexdigest()
