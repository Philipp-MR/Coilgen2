"""Optional native workbench; importing CoilGen never imports Qt or VTK.

Run ``python -m coilgen.gui`` or call :func:`launch` from a scientific script.
"""

from __future__ import annotations


def launch(project=None, *, jobs_directory=None, background=False) -> int:
    """Open the native workbench on an optional ProjectDocument.

    Numerical script APIs continue to operate independently of this module.
    """
    from .__main__ import launch as _launch

    return _launch(project, jobs_directory=jobs_directory, background=background)


__all__ = ["launch"]
