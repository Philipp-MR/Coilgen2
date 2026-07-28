"""Matplotlib backend selection shared by all CoilGen plotting front ends."""

from __future__ import annotations


def activate_matplotlib_interactive_backend(preferred: str = "qt") -> str:
    """Request an interactive backend suitable for Spyder and IPython."""

    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("matplotlib", preferred)
            import matplotlib

            _disable_pyplot_auto_show()
            return str(matplotlib.get_backend())
    except Exception:
        pass

    import matplotlib

    for backend in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(backend, force=True)
            _disable_pyplot_auto_show()
            return str(matplotlib.get_backend())
        except Exception:
            continue
    _disable_pyplot_auto_show()
    return str(matplotlib.get_backend())


def _disable_pyplot_auto_show() -> None:
    """Keep figures hidden until CoilGen explicitly shows a window or book."""

    try:
        import matplotlib.pyplot as plt

        plt.ioff()
    except Exception:
        pass
