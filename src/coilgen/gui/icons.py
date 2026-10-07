"""Repository-owned scientific pictograms rendered against the current Qt palette.

Icons describe model objects and operations. A status badge describes recorded
evidence; the conductor and export pictograms alone make no readiness claim.
All geometry is kept here as small SVG paths, without font or package-data
dependencies. Labels and accessible names remain the responsibility of controls.
"""

from types import MappingProxyType

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QIconEngine, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from .colors import tone_color


# One 24-unit drawing grid, rounded 1.6-unit strokes and no font glyphs.
# Surface boundaries are visible: the cylinder/frustum icons are open surfaces.
_DRAWINGS = {
    "target-gradient": '<path d="M4 19V5m-2 2 2-2 2 2M4 19h17m-2-2 2 2-2 2M8 16v-2m4 2V11m4 5V8m4 8V5"/>',
    "target-uniform": '<rect x="3" y="4" width="18" height="16" rx="3" stroke-dasharray="2 2"/>'
    '<path d="M7 16V8m-2 2 2-2 2 2m3 6V8m-2 2 2-2 2 2m3 6V8m-2 2 2-2 2 2"/>',
    "target-harmonic": '<circle cx="12" cy="12" r="9" stroke-dasharray="2 2"/>'
    '<ellipse cx="8" cy="8" rx="3" ry="4" transform="rotate(-45 8 8)"/>'
    '<ellipse cx="16" cy="16" rx="3" ry="4" transform="rotate(-45 16 16)"/>'
    '<path d="M6 8h4m-2-2v4m6 6h4M14 6l4 4M6 14l4 4"/>',
    "target-map": '<rect x="3" y="3" width="18" height="18" rx="2"/>'
    '<path d="M3 9h18M3 15h18M9 3v18M15 3v18M5 18l4-4 3 1 5-7 2-1"/>',
    "roi": '<circle cx="12" cy="12" r="9" stroke-dasharray="2 2"/>'
    '<ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>'
    '<circle cx="8" cy="8" r=".6"/><circle cx="16" cy="15" r=".6"/>',
    "field-component": '<path d="M8 16V3m-2 2 2-2 2 2M8 16h13m-2-2 2 2-2 2M8 16l-5 5m0-3v3h3"/>'
    '<path d="M12 12V6m-2 2 2-2 2 2" stroke-width="2.5"/>',
    "support-cylinder": '<ellipse cx="12" cy="5" rx="8" ry="3"/>'
    '<path d="M4 5v14m16-14v14M4 19c0 4 16 4 16 0"/>'
    '<path d="M4 19c0-4 16-4 16 0" stroke-dasharray="2 2"/>',
    "support-frustum": '<ellipse cx="12" cy="5" rx="5" ry="2.5"/>'
    '<path d="M7 5 3 19m14-14 4 14M3 19c0 4 18 4 18 0"/>'
    '<path d="M3 19c0-4 18-4 18 0" stroke-dasharray="2 2"/>',
    "support-biplanar": '<path d="m3 5 13-2 5 5-13 2zM3 14l13-2 5 5-13 2z"/><path d="M5 9v4m-2-2 2 2 2-2"/>',
    "support-annulus": '<ellipse cx="12" cy="12" rx="10" ry="7"/><ellipse cx="12" cy="12" rx="4" ry="2.8"/>',
    "support-cap": '<path d="M3 17a9 13 0 0 1 18 0M6 18C5 5 19 5 18 18"/><ellipse cx="12" cy="17" rx="9" ry="3"/>',
    "support-elliptical-cylinder": '<ellipse cx="12" cy="5" rx="9" ry="2"/>'
    '<path d="M3 5v14m18-14v14M3 19c0 3 18 3 18 0"/>'
    '<path d="M3 19c0-3 18-3 18 0" stroke-dasharray="2 2"/>',
    "support-toroid": '<ellipse cx="12" cy="12" rx="10" ry="7"/>'
    '<ellipse cx="12" cy="12" rx="4" ry="2.5"/>'
    '<path d="M3 9c3 7 15 7 18 0" stroke-dasharray="2 2"/>',
    "support-import": '<path d="m3 19 3-13 8-3 7 11-9 7zM6 6l6 15m2-18-2 18M3 19l18-5M6 6l15 8"/>'
    '<path d="M18 3v6m-2-2 2 2 2-2"/>',
    "placement": '<path d="m4 14 8-4 7 4-8 4zM11 18v-9m-2 2 2-2 2 2M11 18h10m-2-2 2 2-2 2"/>',
    "surface-normal": '<path d="m2 16 10-5 10 5-10 5zM12 16V3m-3 3 3-3 3 3"/>',
    "mesh": '<path d="m3 19 3-13 8-3 7 11-9 7zM6 6l6 15m2-18-2 18M3 19l18-5M6 6l15 8"/>',
    "boundary": '<path d="m3 19 3-13 8-3 7 11-9 7z" stroke-width="2.5"/>'
    '<path d="M6 6l6 15m2-18-2 18M3 19l18-5" stroke-width="1"/>',
    "keep-out": '<rect x="3" y="3" width="18" height="18" rx="2"/>'
    '<path d="M3 7h8m5 0h5M3 12h5m8 0h5M3 17h5m8 0h5"/>'
    '<rect x="9" y="7" width="6" height="11" rx="1"/>'
    '<path d="m9 11 4-4m-4 8 6-6m-4 9 4-4" stroke-width="1"/>',
    "conductor-profile": '<rect x="6" y="7" width="13" height="12" rx="1"/>'
    '<path d="M6 3h13m-11-1-2 1 2 1m9-2 2 1-2 1M2 7v12m-1-10 1-2 1 2m-2 8 1 2 1-2"/>',
    "insulation": '<rect x="3" y="3" width="18" height="18" rx="3"/>'
    '<rect x="7" y="7" width="10" height="10" rx="1"/><path d="M5 5l2 2m12 12 2 2"/>',
    "stream-function": '<path d="M3 19c7-2 11-6 18-16M3 15c6-1 9-4 14-12M3 11c4-1 7-4 10-8M3 7c2 0 4-1 6-4"/>'
    '<path d="M3 21h18M3 21V3" stroke-width="1"/>',
    "contours": '<path d="m3 5 14-3 4 17-14 3z" stroke-width="1"/>'
    '<path d="M6 8c2-5 9-4 11 1s1 10-4 10-7-6-7-11z'
    'M9 8c2-2 5-1 6 2s1 6-2 6-5-4-4-8z"/>',
    "connected-winding": '<path d="M19 4C13 1 3 2 3 5c0 4 17 4 17 0v7c0 4-17 4-17 0s17-4 17 0v7'
    'c0 4-17 4-17 0s10-4 16-1M19 4h3M19 18h3"/>',
    "separate-circuits": '<path d="M9 4C4 2 2 3 2 6s8 3 8 0v11c0 3-8 3-8 0s3-3 7-1M9 4h3M9 16h3'
    'M21 4c-5-2-7-1-7 2s8 3 8 0v11c0 3-8 3-8 0s3-3 7-1M21 4h2M21 16h2"/>',
    "independent-turns": '<ellipse cx="12" cy="5" rx="9" ry="2.5"/>'
    '<ellipse cx="12" cy="12" rx="9" ry="2.5"/><ellipse cx="12" cy="19" rx="9" ry="2.5"/>'
    '<path d="m17 3 3 2-3 2m0 3 3 2-3 2m0 3 3 2-3 2"/>',
    "routing": '<path d="M10 4H5a3 3 0 0 0-3 3v2a3 3 0 0 0 3 3h5m4 0h5a3 3 0 0 1 3 3v2'
    'a3 3 0 0 1-3 3h-5M10 4c8 0-4 16 4 16"/><circle cx="10" cy="12" r="1"/><circle cx="14" cy="12" r="1"/>',
    "cut-location": '<path d="M15 5a8 8 0 1 0 4 4"/><path d="m14 3 7 7M20 3l-6 7" stroke-dasharray="2 2"/>',
    "terminals": '<path d="M8 4H6a3 3 0 0 0-3 3v10a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3V7a3 3 0 0 0-3-3h-2"/>'
    '<circle cx="9" cy="4" r="1.5"/><circle cx="15" cy="4" r="1.5"/>',
    "conductor-sweep": '<path d="M2 18c4-7 7-7 10-5"/>'
    '<path d="M11 10c4-3 7-2 10 0v5c-3-2-6-3-10 0zM11 10v5m10-5-3 1v5"/>',
    "wire-spacing": '<path d="M3 4h18M3 7h18M3 17h18M3 20h18M12 8v8m-2-6 2-2 2 2m-4 4 2 2 2-2"/>',
    "bend-radius": '<path d="M3 21V11a8 8 0 0 1 8-8h10M7 21V11a4 4 0 0 1 4-4h10M11 11l-5-5'
    'm0 3V6h3"/><circle cx="11" cy="11" r="1"/>',
    "collision": '<path d="M2 8h20M2 12h20M8 2v20M12 2v20"/>'
    '<circle cx="10" cy="10" r="5"/><path d="m7 7 6 6m0-6-6 6"/>',
    "current-direction": '<ellipse cx="12" cy="12" rx="9" ry="7"/><path d="m16 5 5 4-5 1M3 12l3-2v5"/>',
    "field-error": '<path d="M3 3v18h18M5 17 19 5M5 15l4-2 4-2 6-4"/><path d="M13 9v3m-1-2 1-1 1 1m-2 1 1 1 1-1"/>',
    "independent-validation": '<circle cx="12" cy="12" r="9" stroke-dasharray="2 2"/>'
    '<path d="M7 6v2m-1-1h2m3 3v2m-1-1h2m3-5v2m-1-1h2m-8 8v2m-1-1h2m8-4v2m-1-1h2"/>'
    '<circle cx="6" cy="12" r=".7"/><circle cx="12" cy="18" r=".7"/><circle cx="17" cy="17" r=".7"/>',
    "current-density": '<rect x="3" y="3" width="18" height="18" rx="2"/>'
    '<path d="M7 17V7m-2 2 2-2 2 2m3 10V7m-2 2 2-2 2 2m3 10V7m-2 2 2-2 2 2"/>',
    "resistance": '<path d="M2 12h3l2-5 3 10 3-10 3 10 2-5h4"/>',
    "inductance": '<path d="M2 15h3v-5a2 2 0 0 1 4 0v5-5a2 2 0 0 1 4 0v5-5a2 2 0 0 1 4 0v5h5"/>',
    "power-loss": '<path d="M3 17h18M3 21h18M6 13c-3-3 3-4 0-8m6 8c-3-3 3-4 0-8m6 8c-3-3 3-4 0-8"/>',
    "constraint": '<path d="M4 5v14m16-14v14M5 12h14m-11-2-3 2 3 2m8-4 3 2-3 2"/>',
    "regularisation": '<path d="M3 8 6 3l3 8 3-7 3 6 3-5 3 3M3 19c6-8 12 5 18-3"/>',
    "compute-study": '<path d="M3 4v16l10-8zM16 4h5v5h-5zM16 15h5v5h-5zM18.5 9v6"/>',
    "compare-designs": '<path d="M10 3v18M14 3v18M2 6c6-4 6 4 0 0v12c6-4 6 4 0 0M16 6c6-4 6 4 0 0v12c6-4 6 4 0 0"/>',
    "design-variant": '<path d="M4 3v8c0 3 4 3 4 6v4M4 11c0 3 15 0 15 6v4"/>'
    '<circle cx="4" cy="3" r="1.5"/><circle cx="8" cy="21" r="1.5"/><circle cx="19" cy="21" r="1.5"/>',
    "run-history": '<path d="M3 8a9 9 0 1 1 0 8M3 3v5h5M12 7v5l4 2"/>',
    "export-conductor": '<path d="M3 8c0-4 11-4 11 0s-11 4-11 0v11c0 4 11 4 11 0v-5"/><path d="M13 9h9m-4-4 4 4-4 4"/>',
    "project-json": '<path d="M5 2h10l4 4v16H5zM15 2v5h4M10 10c-3 0 0 3-3 4 3 1 0 4 3 4m4-8c3 0 0 3 3 4-3 1 0 4-3 4"/>',
    "load-template": '<path d="M5 2h10l4 4v16H5zM15 2v5h4"/>'
    '<ellipse cx="12" cy="11" rx="4" ry="1.5"/>'
    '<path d="M8 11v6c0 2 8 2 8 0v-6M8 14c0 2 8 2 8 0"/>',
    "shielded-gradient": '<ellipse cx="12" cy="4" rx="10" ry="3"/>'
    '<path d="M2 4v16c0 4 20 4 20 0V4"/>'
    '<ellipse cx="12" cy="8" rx="6" ry="2"/>'
    '<path d="M6 8v8c0 3 12 3 12 0V8M6 12c0 3 12 3 12 0"/>',
    "open": '<path d="M2 19V5h8l2 3h10l-4 11zM2 19l4-8h16"/>',
    "save": '<path d="M3 3h15l3 3v15H3zM7 3v6h10V3M7 21v-8h10v8"/>',
    "undo": '<path d="M3 7h10a7 7 0 0 1 0 14M3 7l5-5M3 7l5 5"/>',
    "redo": '<path d="M21 7H11a7 7 0 0 0 0 14m10-14-5-5m5 5-5 5"/>',
    "fit-view": '<path d="M3 9V3h6m6 0h6v6m0 6v6h-6m-6 0H3v-6M8 8h8v8H8z"/>',
    "visibility": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>',
    "pin": '<path d="m8 3 10 0-2 7 3 4H5l3-4zM12 14v8"/>',
    "unpin": '<path d="M8 3h10l-2 7 3 4H9M12 14v8M3 3l18 18"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v7m-1-7h1m-1 7h2"/><circle cx="12" cy="7" r=".7"/>',
    "locate": '<circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="1.5"/>'
    '<path d="M12 2v5m0 10v5M2 12h5m10 0h5"/>',
    "cancel": '<path d="m5 5 14 14M19 5 5 19"/>',
    "expand": '<path d="m5 9 7 7 7-7"/>',
    "collapse": '<path d="m5 15 7-7 7 7"/>',
    "edit": '<path d="m3 21 1-6L16 3l5 5L9 20zM14 5l5 5M4 15l5 5"/>',
    "settings": '<path d="M3 6h18M3 12h18M3 18h18"/>'
    '<circle cx="8" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="9" cy="18" r="2"/>',
    "warning": '<path d="M12 3 2 21h20zM12 9v5"/><circle cx="12" cy="17.5" r=".7"/>',
    "check": '<path d="m4 12 5 5L20 6"/>',
    "refresh": '<path d="M20 9A8 8 0 0 0 5 5L2 8m0-5v5h5m-3 7a8 8 0 0 0 15 4l3-3m0 5v-5h-5"/>',
    "delete": '<path d="M3 6h18M8 6V3h8v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
    "copy": '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M15 5V3H3v12h2"/>',
    "view-axes": '<path d="M9 15V3m-2 2 2-2 2 2M9 15h12m-2-2 2 2-2 2M9 15l-6 6m0-3v3h3"/>',
    "slice": '<path d="m3 7 10-4 8 4-10 4zM3 7v10l8 4 10-4V7M11 11v10"/>'
    '<path d="m3 12 8 4 10-4" stroke-dasharray="2 2"/>',
    "chart": '<path d="M3 3v18h18M6 16l4-5 4 2 6-8"/>',
    "table": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/>',
    "filter": '<path d="M3 4h18l-7 8v7l-4 2v-9z"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9 8a3 3 0 0 1 6 0c0 3-3 2-3 6"/><circle cx="12" cy="18" r=".7"/>',
    "back": '<path d="M21 12H3m6-6-6 6 6 6"/>',
    "next": '<path d="M3 12h18m-6-6 6 6-6 6"/>',
    "export": '<path d="M3 13v8h18v-8M12 16V3m-5 5 5-5 5 5"/>',
    "clear": '<path d="m3 15 11-12 7 7-10 11H8zM8 10l7 7M11 21h10"/>',
}

_DRAWINGS.update({"remove": _DRAWINGS["delete"], "fit": _DRAWINGS["fit-view"], "information": _DRAWINGS["info"]})


def _signed_lobe(x, y, rx, ry, *, positive, rotation=0):
    """A signed scalar lobe, deliberately independent of any vector-field arrow."""
    ellipse = f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" transform="rotate({rotation} {x} {y})"/>'
    sign = f'<path d="M{x - 1.2} {y}h2.4'
    if positive:
        sign += f"M{x} {y - 1.2}v2.4"
    return f'<g stroke-width="1.2">{ellipse}{sign}"/></g>'


_SCALAR_ROI = '<circle cx="12" cy="12" r="10" stroke-width="1" stroke-dasharray="2 2"/>'
_DRAWINGS.update(
    {
        "channels": '<path d="M3 4v16M3 6h7m-7 6h7m-7 6h7"/>'
        '<rect x="10" y="3" width="11" height="6" rx="1"/>'
        '<rect x="10" y="10" width="11" height="4" rx="1"/>'
        '<rect x="10" y="15" width="11" height="6" rx="1"/>',
        "assembly": '<path d="m3 6 9-4 9 4-9 4zM3 12l9 4 9-4M3 18l9 4 9-4"/>'
        '<path d="M3 6v4m18-4v4M3 12v4m18-4v4M12 10v4m0 2v4" stroke-dasharray="1.5 2"/>',
        "biplanar-system": '<path d="m2 5 13-3 7 5-13 3zM2 15l13-3 7 5-13 3z"/>'
        '<path d="m5 5 9-1 4 2-9 2zM5 15l9-1 4 2-9 2z" stroke-width="1"/>'
        '<path d="M3 10v3m18-3v3" stroke-dasharray="1 2"/>',
        "workspace": '<rect x="2" y="3" width="20" height="18" rx="2"/><path d="M2 7h20M7 7v14M17 7v14M7 17h10"/>',
        "model-tree": '<path d="M4 3v14h4M4 7h4M12 7v14h4M12 12h4"/>'
        '<rect x="8" y="4" width="6" height="6" rx="1"/>'
        '<rect x="16" y="10" width="5" height="4" rx="1"/>'
        '<rect x="16" y="19" width="5" height="4" rx="1"/>',
        "inspector": '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<path d="M12 3v18M5 8h5m-5 8h5m5-9h4m-4 4h4m-4 4h4m-4 4h4"/>',
        "evidence-panel": '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<path d="m6 8 1.5 1.5L10 6M13 8h5m-12 6 1.5 1.5L10 12m3 2h5M6 19h12"/>',
        "jobs-panel": '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<path d="m6 6 4 3-4 3zM13 7h5m-5 4h5M6 16h12m-12 3h8"/>',
        "results-panel": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M6 7h12M7 18v-5m5 5v-7m5 7v-9"/>',
        "gradient-x": _SCALAR_ROI
        + _signed_lobe(6, 12, 3, 6, positive=False)
        + _signed_lobe(18, 12, 3, 6, positive=True),
        "gradient-y": _SCALAR_ROI
        + _signed_lobe(7, 17, 3, 5.5, positive=False, rotation=45)
        + _signed_lobe(17, 7, 3, 5.5, positive=True, rotation=45),
        "gradient-z": _SCALAR_ROI
        + _signed_lobe(12, 6, 6, 3, positive=True)
        + _signed_lobe(12, 18, 6, 3, positive=False),
        "shim-z0": _SCALAR_ROI + '<path d="M7 6v3m-1.5-1.5h3M17 6v3m-1.5-1.5h3M12 11v3m-1.5-1.5h3'
        'M7 16v3m-1.5-1.5h3M17 16v3m-1.5-1.5h3" stroke-width="1.2"/>',
        # Z2: positive polar lobes and a negative equatorial belt, for positive
        # 2*z**2 - x**2 - y**2 coefficient. These are mode glyphs, not field maps.
        "shim-z2": _signed_lobe(12, 5, 3.5, 3.5, positive=True)
        + _signed_lobe(12, 19, 3.5, 3.5, positive=True)
        + '<ellipse cx="12" cy="12" rx="9" ry="3" stroke-width="1.2"/>'
        '<path d="M5 12h3m8 0h3" stroke-width="1.2"/>',
        # Representative slice planes distinguish xz, yz and xy; signs describe
        # scalar Bz at a positive coefficient, never a magnetic-vector direction.
        "shim-xz": '<path d="M3 2h18v20H3z" stroke-width=".8" stroke-dasharray="2 2"/>'
        + _signed_lobe(8, 6, 3.2, 4, positive=False, rotation=-30)
        + _signed_lobe(16, 6, 3.2, 4, positive=True, rotation=30)
        + _signed_lobe(8, 18, 3.2, 4, positive=True, rotation=30)
        + _signed_lobe(16, 18, 3.2, 4, positive=False, rotation=-30),
        "shim-yz": '<path d="m5 1 17 5-3 17-17-5z" stroke-width=".8" stroke-dasharray="2 2"/>'
        + _signed_lobe(8, 5, 2.8, 3.3, positive=False, rotation=-25)
        + _signed_lobe(18, 8, 2.8, 3.3, positive=True, rotation=35)
        + _signed_lobe(6, 16, 2.8, 3.3, positive=True, rotation=35)
        + _signed_lobe(16, 19, 2.8, 3.3, positive=False, rotation=-25),
        "shim-x2-y2": _signed_lobe(5, 12, 3.5, 2.8, positive=True)
        + _signed_lobe(19, 12, 3.5, 2.8, positive=True)
        + _signed_lobe(12, 5, 3.5, 2.8, positive=False, rotation=90)
        + _signed_lobe(12, 19, 3.5, 2.8, positive=False, rotation=90),
        "shim-xy": _SCALAR_ROI
        + _signed_lobe(7, 7, 4, 2.8, positive=False, rotation=45)
        + _signed_lobe(17, 7, 4, 2.8, positive=True, rotation=-45)
        + _signed_lobe(7, 17, 4, 2.8, positive=True, rotation=-45)
        + _signed_lobe(17, 17, 4, 2.8, positive=False, rotation=45),
    }
)
_DRAWINGS["history-panel"] = _DRAWINGS["run-history"]
_DRAWINGS.update(
    {
        "support-plane": '<path d="m2 8 12-5 8 10-12 5zM6 9l9-4m-7 10 9-4"/>',
        "support-gallery": '<ellipse cx="6.5" cy="4" rx="4" ry="1.5"/>'
        '<path d="M2.5 4v7c0 2 8 2 8 0V4m3-1 8 2-3 7-5-2z"/>'
        '<ellipse cx="12" cy="16" rx="3" ry="1.3"/>'
        '<path d="m9 16-4 5c0 2 14 2 14 0l-4-5"/>',
        "support-disc": '<ellipse cx="12" cy="12" rx="10" ry="6"/><path d="M5 12h14m-7-3v6" stroke-width="1"/>',
        "support-partial-cylinder": '<path d="M4 6c0-5 16-5 16 0M4 6v13m16-13v13M4 19c0-5 16-5 16 0"/>',
        "support-superellipse": '<rect x="3" y="2" width="18" height="6" rx="2"/>'
        '<path d="M3 5v14c0 4 18 4 18 0V5"/>'
        '<path d="M3 19c0-4 18-4 18 0" stroke-dasharray="2 2"/>',
        "support-circular-biplanar": '<ellipse cx="12" cy="6" rx="10" ry="3.5"/>'
        '<ellipse cx="12" cy="18" rx="10" ry="3.5"/>'
        '<path d="M4 10v4m16-4v4" stroke-dasharray="1 2"/>',
        "support-ellipsoidal-cap": '<path d="M2 17C2 5 22 5 22 17M7 19c-2-12 12-12 10 0"/>'
        '<ellipse cx="12" cy="17" rx="10" ry="3"/>',
        "roi-ellipsoid": '<ellipse cx="12" cy="12" rx="10" ry="7" stroke-dasharray="2 2"/>'
        '<ellipse cx="12" cy="12" rx="4" ry="7"/><path d="M2 12h20"/>',
        "roi-box": '<path d="m3 7 9-4 9 4-9 4zM3 7v10l9 4 9-4V7M12 11v10" stroke-dasharray="2 2"/>'
        '<circle cx="8" cy="12" r=".7"/><circle cx="17" cy="13" r=".7"/>'
        '<circle cx="8" cy="17" r=".7"/><circle cx="16" cy="17" r=".7"/>',
        "roi-cylinder": '<ellipse cx="12" cy="5" rx="8" ry="3" stroke-dasharray="2 2"/>'
        '<ellipse cx="12" cy="19" rx="8" ry="3" stroke-dasharray="2 2"/>'
        '<path d="M4 5v14m16-14v14" stroke-dasharray="2 2"/>'
        '<circle cx="8" cy="10" r=".7"/><circle cx="16" cy="10" r=".7"/>'
        '<circle cx="8" cy="15" r=".7"/><circle cx="16" cy="15" r=".7"/>',
        "roi-enclosing-mesh": '<path d="m2 12 4-9 11-1 5 9-5 10-11 1zM6 3l11 18M17 2 6 22M2 12l20-1"/>'
        '<circle cx="9" cy="12" r="1"/><circle cx="15" cy="12" r="1"/>',
        "roi-surface": '<circle cx="12" cy="12" r="8" stroke-dasharray="2 2"/>'
        '<circle cx="12" cy="4" r="1.4"/><circle cx="4" cy="12" r="1.4"/>'
        '<circle cx="20" cy="12" r="1.4"/><circle cx="12" cy="20" r="1.4"/>',
        "roi-volume": '<path d="m3 7 9-4 9 4-9 4zM3 7v10l9 4 9-4V7M12 11v10" stroke-width="1.2"/>'
        '<circle cx="7" cy="11" r=".7"/><circle cx="7" cy="16" r=".7"/>'
        '<circle cx="17" cy="11" r=".7"/><circle cx="17" cy="16" r=".7"/>'
        '<circle cx="12" cy="6" r=".7"/>',
    }
)
_DRAWINGS["roi-sphere"] = _DRAWINGS["roi"]

ICON_NAMES = tuple(_DRAWINGS)
WORKFLOW_ICONS = MappingProxyType(
    {
        "target": "target-gradient",
        "support": "support-cylinder",
        "conductor": "conductor-profile",
        "solve": "stream-function",
        "winding": "connected-winding",
        "check": "independent-validation",
        "review": "field-error",
        "export": "export-conductor",
    }
)
INSPECTION_ICONS = MappingProxyType(
    {"target": "roi", "field": "field-error", "wire": "conductor-sweep", "connections": "routing"}
)
WINDING_ICONS = MappingProxyType(
    {
        "single": "connected-winding",
        "per_part": "separate-circuits",
        "independent": "independent-turns",
        "contours": "contours",
    }
)
SUPPORT_ICONS = MappingProxyType(
    {
        "cylinder": "support-cylinder",
        "biplanar": "support-biplanar",
        "frustum": "support-frustum",
        "import": "support-import",
        "annulus": "support-annulus",
        "cap": "support-cap",
        "elliptical-cylinder": "support-elliptical-cylinder",
        "toroid": "support-toroid",
        "rectangle": "support-plane",
        "disc": "support-disc",
        "partial-cylinder": "support-partial-cylinder",
        "superellipse": "support-superellipse",
        "circular-biplanar": "support-circular-biplanar",
        "ellipsoidal-cap": "support-ellipsoidal-cap",
        "spherical-cap": "support-cap",
        "gallery": "support-gallery",
    }
)
REGION_ICONS = MappingProxyType(
    {
        "sphere": "roi-sphere",
        "ellipsoid": "roi-ellipsoid",
        "box": "roi-box",
        "cylinder": "roi-cylinder",
        "enclosing_mesh": "roi-enclosing-mesh",
        "imported_map": "target-map",
        "legacy_mesh": "roi-enclosing-mesh",
        "volume": "roi-volume",
        "surface": "roi-surface",
    }
)
TARGET_MODE_ICONS = MappingProxyType(
    {
        "Gx": "gradient-x",
        "Gy": "gradient-y",
        "Gz": "gradient-z",
        "Z0": "shim-z0",
        "Z2": "shim-z2",
        "XZ": "shim-xz",
        "YZ": "shim-yz",
        "X2-Y2": "shim-x2-y2",
        "XY": "shim-xy",
    }
)
PANEL_ICONS = MappingProxyType(
    {
        "model": "model-tree",
        "inspector": "inspector",
        "evidence": "evidence-panel",
        "jobs": "jobs-panel",
        "results": "results-panel",
        "history": "history-panel",
    }
)

# The badge is a separate assertion, never an intrinsic property of a shape.
_BADGES = {
    "passed": ("success", '<path d="m16 18.5 1.6 1.6 3.4-3.6"/>'),
    "failed": ("error", '<path d="m16.5 16.5 4 4m0-4-4 4"/>'),
    "unverified": ("warning", '<path d="M17 17a1.5 1.5 0 0 1 3 0c0 1.3-1.5 1-1.5 2"/><path d="M18.5 20.7v.1"/>'),
    "pending": ("info", '<path d="M18.5 15.7v2.8l1.8 1"/>'),
    "stale": ("warning", '<path d="M16.1 18.5a2.5 2.5 0 1 1 1.1 2M16 16.5v2h2"/>'),
    "baseline": ("info", '<path d="M17 15.7h3l-.5 2 1 1h-4l1-1zM18.5 18.7v2.6"/>'),
    "evaluated": ("info", '<path d="M16 18.5h5M18.5 16v5"/>'),
    "skipped": ("neutral", '<path d="M16 18.5h5"/>'),
}
STATUS_BADGES = tuple(_BADGES)


def svg_data(name: str, *, badge: str | None = None, mode=QIcon.Mode.Normal, foreground="default") -> bytes:
    """Return the current theme's SVG, useful for previews and render validation."""
    if name not in _DRAWINGS:
        raise ValueError(f"Unknown CoilGen icon: {name!r}")
    if badge is not None and badge not in _BADGES:
        raise ValueError(f"Unknown CoilGen icon badge: {badge!r}")
    if foreground not in ("default", "primary"):
        raise ValueError(f"Unknown CoilGen icon foreground: {foreground!r}")
    palette = QApplication.palette()
    disabled = mode == QIcon.Mode.Disabled
    selected = mode == QIcon.Mode.Selected
    group = QPalette.ColorGroup.Disabled if disabled else QPalette.ColorGroup.Active
    role = QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.ButtonText
    # The workbench primary-action stylesheet has a fixed #075aaa background.
    # Its explicit light foreground is independent of the OS highlight palette.
    primary = foreground == "primary" and not disabled and not selected
    color = "#ffffff" if primary else palette.color(group, role).name()
    drawing = _DRAWINGS[name]
    if badge is not None:
        tone, glyph = _BADGES[badge]
        badge_color = color if disabled or selected or primary else tone_color(tone, palette).name()
        backing_role = QPalette.ColorRole.Highlight if selected else QPalette.ColorRole.Window
        backing = "#075aaa" if primary else palette.color(group, backing_role).name()
        drawing += (
            f'<g stroke="{backing}" stroke-width="2.8" fill="{backing}">'
            '<circle cx="18.5" cy="18.5" r="4.2"/></g>'
            f'<g stroke="{badge_color}" stroke-width="1.2" fill="none">'
            f'<circle cx="18.5" cy="18.5" r="4.2"/>{glyph}</g>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
        f'<g fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round">{drawing}</g></svg>'
    ).encode("ascii")


class _ScientificIconEngine(QIconEngine):
    """Render vectors on demand so existing QIcons follow palette changes."""

    def __init__(self, name: str, badge: str | None = None, foreground="default"):
        super().__init__()
        self.name = name
        self.badge = badge
        self.foreground = foreground
        self._svg = None
        self._renderer = None

    def clone(self):
        return _ScientificIconEngine(self.name, self.badge, self.foreground)

    def key(self):
        return "CoilGenScientificIcon"

    def paint(self, painter, rect, mode, state):
        data = svg_data(self.name, badge=self.badge, mode=mode, foreground=self.foreground)
        if data != self._svg:
            self._svg = data
            self._renderer = QSvgRenderer(QByteArray(data))
        # Keep the drawing square even when the requested button rectangle is not.
        bounds = QRectF(rect)
        side = min(bounds.width(), bounds.height())
        square = QRectF(0, 0, side, side)
        square.moveCenter(bounds.center())
        self._renderer.render(painter, square)

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size, mode, state, scale):
        if size.isEmpty() or scale <= 0:
            return QPixmap()
        physical = QSize(round(size.width() * scale), round(size.height() * scale))
        result = QPixmap(physical)
        result.fill(Qt.GlobalColor.transparent)
        result.setDevicePixelRatio(scale)
        painter = QPainter(result)
        self.paint(painter, QRectF(0, 0, size.width(), size.height()), mode, state)
        painter.end()
        return result


def icon(name: str, *, badge: str | None = None, foreground="default") -> QIcon:
    """Create a theme-aware vector icon; unknown names/states fail explicitly."""
    if name not in _DRAWINGS:
        raise ValueError(f"Unknown CoilGen icon: {name!r}")
    if badge is not None and badge not in _BADGES:
        raise ValueError(f"Unknown CoilGen icon badge: {badge!r}")
    if foreground not in ("default", "primary"):
        raise ValueError(f"Unknown CoilGen icon foreground: {foreground!r}")
    return QIcon(_ScientificIconEngine(name, badge, foreground))
