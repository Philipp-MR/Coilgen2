"""Standalone execution helpers; importing examples does not start work."""

from contextlib import ExitStack, contextmanager
import argparse
from importlib.util import find_spec
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def example_options(default_output, argv=None, *, study=False):
    parser = argparse.ArgumentParser(description="Run this CoilGen example and save its numerical evidence.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / default_output)
    if study:
        parser.add_argument("--no-resume", dest="resume", action="store_false", help="Recompute all study candidates.")
    options = parser.parse_args(argv)
    options.output_dir = options.output_dir.resolve()
    return options


@contextmanager
def example_runtime(*, extra=None, required=(), fresh_numba_cache=False):
    """Use this checkout and one native thread only while explicitly running.

    Environment limits cover freshly imported numerical libraries and spawned
    workers; threadpoolctl also covers libraries already loaded by a caller.
    """
    variables = (*_THREAD_VARIABLES, "NUMBA_CACHE_DIR")
    environment = {name: os.environ.get(name) for name in variables}
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT / "src"))
        for name in _THREAD_VARIABLES:
            os.environ[name] = "1"
        missing = [name for name in required if find_spec(name) is None]
        if missing:
            package = str(ROOT) + (f"[{extra}]" if extra else "")
            raise SystemExit(
                f"Missing example dependencies: {', '.join(missing)}. Install with:\n"
                f'"{sys.executable}" -m pip install -e "{package}"'
            )
        with ExitStack() as stack:
            if fresh_numba_cache:
                if "numba" in sys.modules:
                    raise RuntimeError("Run the geometry benchmark in a fresh Python process to measure compilation.")
                os.environ["NUMBA_CACHE_DIR"] = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="coilgen_numba_benchmark_")
                )
            if find_spec("threadpoolctl") is not None:
                from threadpoolctl import threadpool_limits

                stack.enter_context(threadpool_limits(limits=1))
            yield ROOT
    finally:
        sys.path[:] = previous_path
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
