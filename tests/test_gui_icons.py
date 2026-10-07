"""Scientific icons stay valid, legible and theme-aware without desktop use."""

import os
import xml.etree.ElementTree as ET

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QByteArray, QRect, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPalette
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from coilgen.gui.icons import (
    ICON_NAMES,
    INSPECTION_ICONS,
    PANEL_ICONS,
    REGION_ICONS,
    STATUS_BADGES,
    SUPPORT_ICONS,
    TARGET_MODE_ICONS,
    WINDING_ICONS,
    WORKFLOW_ICONS,
    icon,
    svg_data,
)


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["scientific-icons-test"])


@pytest.fixture
def restore_palette(application):
    original = QPalette(application.palette())
    yield
    application.setPalette(original)


def _visible_pixels(image):
    return [(x, y) for y in range(image.height()) for x in range(image.width()) if image.pixelColor(x, y).alpha() > 0]


def _pixels(image):
    return image.constBits().tobytes()


@pytest.mark.parametrize("name", ICON_NAMES)
def test_every_pictogram_is_valid_vector_and_renders_transparently(application, name):
    source = svg_data(name)
    root = ET.fromstring(source)
    assert root.attrib["viewBox"] == "0 0 24 24"
    assert QSvgRenderer(QByteArray(source)).isValid()
    assert b"<text" not in source and b"<image" not in source  # No font, emoji or bitmap dependencies.
    image = icon(name).pixmap(QSize(24, 24)).toImage()
    assert len(_visible_pixels(image)) >= 25
    assert image.pixelColor(0, 0).alpha() == 0
    assert not icon(name).pixmap(QSize(16, 16)).isNull()


def test_semantic_maps_resolve_and_topology_glyphs_remain_distinct(application):
    assert len(ICON_NAMES) == len(set(ICON_NAMES))
    for mapping in (
        WORKFLOW_ICONS,
        INSPECTION_ICONS,
        SUPPORT_ICONS,
        WINDING_ICONS,
        TARGET_MODE_ICONS,
        PANEL_ICONS,
        REGION_ICONS,
    ):
        assert all(name in ICON_NAMES for name in mapping.values())
    drawings = {svg_data(name) for name in WINDING_ICONS.values()}
    assert len(drawings) == 4


def test_region_shape_and_sampling_pictograms_are_distinct(application):
    shapes = ("sphere", "ellipsoid", "box", "cylinder", "enclosing_mesh")
    assert len({svg_data(REGION_ICONS[shape]) for shape in shapes}) == len(shapes)
    assert svg_data(REGION_ICONS["surface"]) != svg_data(REGION_ICONS["volume"])
    assert svg_data("support-cylinder") != svg_data("support-partial-cylinder")
    assert svg_data("support-biplanar") != svg_data("support-circular-biplanar")
    assert svg_data("support-annulus") != svg_data("support-disc")
    assert SUPPORT_ICONS["gallery"] == "support-gallery"
    assert svg_data("support-gallery") != svg_data("support-import")


def test_scalar_mode_artwork_is_distinct_and_has_no_vector_arrows(application):
    sources = [svg_data(name) for name in TARGET_MODE_ICONS.values()]
    assert len(set(sources)) == 9
    for source in sources:
        assert b"<text" not in source and b"marker" not in source
        assert b"polyline" not in source  # Signed scalar lobes rather than field-vector arrows.
    for name in ("channels", "assembly", "biplanar-system", "workspace"):
        assert name in ICON_NAMES


@pytest.mark.parametrize(
    "mode,centres,coordinates",
    [
        ("Gx", ((6, 12), (18, 12)), ((-1, 0, 0), (1, 0, 0))),
        ("Gy", ((7, 17), (17, 7)), ((0, -1, 0), (0, 1, 0))),
        ("Gz", ((12, 6), (12, 18)), ((0, 0, 1), (0, 0, -1))),
        ("Z2", ((12, 5), (12, 19)), ((0, 0, 1), (0, 0, -1))),
        ("XZ", ((8, 6), (16, 6), (8, 18), (16, 18)), ((-1, 0, 1), (1, 0, 1), (-1, 0, -1), (1, 0, -1))),
        ("YZ", ((8, 5), (18, 8), (6, 16), (16, 19)), ((0, -1, 1), (0, 1, 1), (0, -1, -1), (0, 1, -1))),
        ("X2-Y2", ((5, 12), (19, 12), (12, 5), (12, 19)), ((-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0))),
        ("XY", ((7, 7), (17, 7), (7, 17), (17, 17)), ((-1, 1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0))),
    ],
)
def test_lobe_polarity_matches_positive_scalar_basis(application, mode, centres, coordinates):
    root = ET.fromstring(svg_data(TARGET_MODE_ICONS[mode]))
    paths = [element.attrib["d"] for element in root.iter("{http://www.w3.org/2000/svg}path")]
    for (cx, cy), (x, y, z) in zip(centres, coordinates):
        expected = {
            "Gx": x,
            "Gy": y,
            "Gz": z,
            "Z2": 2 * z * z - x * x - y * y,
            "XZ": x * z,
            "YZ": y * z,
            "X2-Y2": x * x - y * y,
            "XY": x * y,
        }[mode]
        glyph = next(path for path in paths if path.startswith(f"M{cx - 1.2} {cy}h2.4"))
        assert ("v2.4" in glyph) == (expected > 0)
    if mode == "Z2":
        assert "M5 12h3m8 0h3" in paths  # Both equatorial lobes carry minus signs.


@pytest.mark.parametrize("badge", STATUS_BADGES)
def test_badges_add_distinct_valid_evidence_marks(application, badge):
    source = svg_data("conductor-sweep", badge=badge)
    assert QSvgRenderer(QByteArray(source)).isValid()
    plain = icon("conductor-sweep").pixmap(QSize(32, 32)).toImage()
    marked = icon("conductor-sweep", badge=badge).pixmap(QSize(32, 32)).toImage()
    assert _pixels(plain) != _pixels(marked)
    assert marked.pixelColor(0, 0).alpha() == 0


def test_badge_states_are_distinct_without_colour(application):
    palette = application.palette()
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#222222"))
    # Disabled mode deliberately removes semantic hue but retains each glyph.
    sources = {svg_data("mesh", badge=state, mode=QIcon.Mode.Disabled) for state in STATUS_BADGES}
    assert len(sources) == len(STATUS_BADGES)


def test_existing_icon_tracks_palette_changes_and_mode_roles(application, restore_palette):
    existing = icon("mesh")
    copied = QIcon(existing)
    palette = QPalette(application.palette())
    palette.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText, QColor("#ed1122"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#2244ee"))
    palette.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.HighlightedText, QColor("#22cc55"))
    application.setPalette(palette)
    assert b'stroke="#ed1122"' in svg_data("mesh")
    assert b'stroke="#2244ee"' in svg_data("mesh", mode=QIcon.Mode.Disabled)
    assert b'stroke="#22cc55"' in svg_data("mesh", mode=QIcon.Mode.Selected)
    first = existing.pixmap(QSize(24, 24)).toImage()
    disabled = existing.pixmap(QSize(24, 24), QIcon.Mode.Disabled).toImage()
    selected = existing.pixmap(QSize(24, 24), QIcon.Mode.Selected).toImage()
    assert len({_pixels(first), _pixels(disabled), _pixels(selected)}) == 3
    palette.setColor(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText, QColor("#dddddd"))
    application.setPalette(palette)
    changed = existing.pixmap(QSize(24, 24)).toImage()
    assert _pixels(first) != _pixels(changed)
    assert _pixels(copied.pixmap(QSize(24, 24)).toImage()) == _pixels(changed)


def test_primary_action_foreground_is_light_on_fixed_blue(application, restore_palette):
    palette = application.palette()
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#151515"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#151515"))
    application.setPalette(palette)
    assert b'stroke="#ffffff"' in svg_data("compute-study", foreground="primary")
    normal = icon("compute-study").pixmap(QSize(24, 24)).toImage()
    primary = icon("compute-study", foreground="primary").pixmap(QSize(24, 24)).toImage()
    assert _pixels(normal) != _pixels(primary)


def test_hidpi_renders_at_physical_resolution(application):
    source = icon("support-cylinder", badge="unverified")
    ordinary = source.pixmap(QSize(24, 24))
    high_dpi = source.pixmap(QSize(24, 24), 2.0)
    assert ordinary.size() == QSize(24, 24)
    assert high_dpi.size() == QSize(48, 48)
    assert high_dpi.devicePixelRatio() == 2.0
    assert len(_visible_pixels(high_dpi.toImage())) > len(_visible_pixels(ordinary.toImage())) * 2


def test_non_square_paint_preserves_geometry_aspect_ratio(application):
    image = QImage(48, 24, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    icon("roi").paint(painter, QRect(0, 0, 48, 24))
    painter.end()
    visible = _visible_pixels(image)
    assert min(x for x, _ in visible) >= 12
    assert max(x for x, _ in visible) < 36


@pytest.mark.parametrize("factory", [icon, svg_data])
def test_unknown_identifiers_fail_explicitly(application, factory):
    with pytest.raises(ValueError, match="Unknown CoilGen icon:"):
        factory("unimplemented-physical-operation")
    with pytest.raises(ValueError, match="badge"):
        factory("mesh", badge="ready-for-construction")
    with pytest.raises(ValueError, match="foreground"):
        factory("mesh", foreground="unreadable")
