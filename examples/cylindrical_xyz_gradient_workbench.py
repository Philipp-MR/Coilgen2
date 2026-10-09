"""Create or compute a complete three-channel cylindrical XYZ gradient insert.

The default writes one portable editable JSON; --gui opens it in the workbench.
--compute performs the independently configured complete Gx/Gy/Gz studies.
See cylindrical_xyz_gradient_README.md for the physical inputs and scope.
"""

from __future__ import annotations

import multiprocessing

from biplanar_gradient_shim_workbench import main as _system_main


def main(argv=None) -> int:
    return _system_main(argv, kind="cylindrical_xyz_gradient")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
