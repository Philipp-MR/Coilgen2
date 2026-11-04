"""Create, open or solve three foil gradient channels with active shields."""

import multiprocessing

from biplanar_gradient_shim_workbench import main as _system_main


def main(argv=None) -> int:
    return _system_main(argv, kind="shielded_xyz_gradient")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
