import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)


class PlotBookLazyRegistrationTest(unittest.TestCase):
    def test_add_spec_defers_factory_until_materialized(self) -> None:
        calls = {"count": 0}

        class FakePyplot:
            def isinteractive(self):
                return False

        def factory():
            calls["count"] += 1
            return SimpleNamespace(get_label=lambda: "Deferred Figure")

        from coilgen.plot_book import MatplotlibPlotBookBuilder

        builder = MatplotlibPlotBookBuilder(title="Lazy Book", plt=FakePyplot())
        builder.add_spec(
            SimpleNamespace(key="lazy_page", title="Lazy Page", category="test", channel="Gx", factory=factory),
            short_name_factory=lambda _spec, title, _index, _total: title,
        )

        self.assertEqual(calls["count"], 0)
        self.assertEqual(builder.entries[0].channel, "Gx")
        entries = builder.entries[0].figure.materialize()
        self.assertEqual(calls["count"], 1)
        self.assertEqual(entries[0].title, "Deferred Figure")
        self.assertEqual(entries[0].channel, "Gx")

    def test_plot_book_table_column_resize_does_not_cap_long_titles(self) -> None:
        from coilgen.plot_book import _resize_plot_book_table_columns

        class FakeTable:
            def __init__(self):
                self.widths = {0: 620, 1: 42, 2: 32}
                self.resized = False

            def resizeColumnsToContents(self):
                self.resized = True

            def columnWidth(self, section):
                return self.widths[section]

            def setColumnWidth(self, section, width):
                self.widths[section] = width

        table = FakeTable()

        _resize_plot_book_table_columns(table)

        self.assertTrue(table.resized)
        self.assertEqual(table.widths[0], 620)
        self.assertEqual(table.widths[1], 70)
        self.assertEqual(table.widths[2], 70)

    def test_plot_book_preload_mode_normalization_accepts_user_friendly_values(self) -> None:
        from coilgen.plot_book import _normalize_plot_book_preload_mode

        self.assertEqual(_normalize_plot_book_preload_mode(True), "idle")
        self.assertEqual(_normalize_plot_book_preload_mode(False), "none")
        self.assertEqual(_normalize_plot_book_preload_mode("background"), "idle")
        self.assertEqual(_normalize_plot_book_preload_mode("off"), "none")
        self.assertEqual(_normalize_plot_book_preload_mode("eager"), "all")

    def test_plot_book_preload_order_prioritizes_current_neighbors(self) -> None:
        from coilgen.plot_book import _plot_book_preload_order

        self.assertEqual(_plot_book_preload_order(2, 6), (2, 3, 1, 4, 0, 5))
        self.assertEqual(_plot_book_preload_order(0, 4), (0, 1, 2, 3))
        self.assertEqual(_plot_book_preload_order(10, 3), (2, 1, 0))
        self.assertEqual(_plot_book_preload_order(0, 0), ())

    def test_page_embedding_failure_becomes_an_unavailable_page(self) -> None:
        from coilgen.plot_book import PlotBookEntry, QtMatplotlibPlotBook

        entry = PlotBookEntry(
            key="broken_page",
            title="Broken Page",
            short_name="broken page",
            category="test",
            figure=object(),
        )
        unavailable_widget = object()
        replacements = []
        state = SimpleNamespace(status="ready", error=None)

        def fail_to_embed(_entry):
            raise RuntimeError("canvas embedding failed")

        fake_book = SimpleNamespace(
            entries=[entry],
            page_widgets=[None],
            page_count=1,
            _page_widget=fail_to_embed,
            _unavailable_page_widget=lambda _entry, _exc: unavailable_widget,
            _replace_page_widget=lambda index, widget: replacements.append((index, widget)),
            _replace_entry=lambda _index, _entry: None,
            add_entry=lambda _entry: None,
            _export_item=lambda _index: (entry, state),
            _update_row=lambda _index: None,
            _touch=lambda _page_id: None,
            _trim_canvas_cache=lambda: None,
        )

        QtMatplotlibPlotBook._ensure_page_widget(fake_book, 0)

        self.assertEqual(replacements, [(0, unavailable_widget)])
        self.assertEqual(state.status, "failed")
        self.assertIsInstance(state.error, RuntimeError)

    def test_plot_book_3d_rotation_buttons_cover_world_axes(self) -> None:
        from coilgen.plot_book import THREE_D_VIEW_ROTATIONS

        labels = [entry[0] for entry in THREE_D_VIEW_ROTATIONS]

        self.assertEqual(labels, ["X+90", "X-90", "Y+90", "Y-90", "Z+90", "Z-90"])
        self.assertEqual({entry[2] for entry in THREE_D_VIEW_ROTATIONS}, {"x", "y", "z"})

    def test_rotate_3d_view_angles_uses_world_axis_quarter_turns(self) -> None:
        from coilgen.plot_book import _rotate_3d_view_angles

        elev, azim, roll = _rotate_3d_view_angles(elev=0.0, azim=0.0, roll=0.0, axis="z", quarter_turns=1)
        self.assertAlmostEqual(elev, 0.0, places=12)
        self.assertAlmostEqual(azim, 90.0, places=12)
        self.assertAlmostEqual(roll, 0.0, places=12)

        elev, azim, roll = _rotate_3d_view_angles(elev=90.0, azim=-90.0, roll=0.0, axis="x", quarter_turns=1)
        self.assertAlmostEqual(elev, 0.0, places=12)
        self.assertAlmostEqual(azim, -90.0, places=12)
        self.assertAlmostEqual(roll, 0.0, places=12)

    def test_rotate_3d_view_angles_round_trips(self) -> None:
        from coilgen.plot_book import _rotate_3d_view_angles, _view_direction_and_up

        original = _view_direction_and_up(elev=22.0, azim=135.0, roll=17.0)
        rotated = _rotate_3d_view_angles(elev=22.0, azim=135.0, roll=17.0, axis="y", quarter_turns=1)
        round_tripped = _rotate_3d_view_angles(
            elev=rotated[0],
            azim=rotated[1],
            roll=rotated[2],
            axis="y",
            quarter_turns=-1,
        )
        recovered = _view_direction_and_up(elev=round_tripped[0], azim=round_tripped[1], roll=round_tripped[2])

        for actual, expected in zip(recovered, original):
            self.assertTrue(((actual - expected) ** 2).sum() < 1.0e-24)


if __name__ == "__main__":
    unittest.main()
