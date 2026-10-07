from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.models import TargetField
from coilgen.project import (
    _channel_color,
    _channel_prefixed_plot_spec,
    _project_channel_sh_fits,
    _project_plot_book_short_name,
    plot_project_run_book,
)


def _channel(name, radius, unit):
    axis = np.linspace(-radius, radius, 5)
    coords = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    field = np.zeros_like(coords)
    field[:, 2] = 0.01 * coords[:, 2] + 2e-6
    target = TargetField(
        coords=coords,
        b=field.copy(),
        weights=np.ones(len(coords)),
        suppressed=np.zeros(len(coords), dtype=bool),
        target_field_group_inds=np.ones(len(coords), dtype=int),
    )
    result = SimpleNamespace(
        input_data=CoilGenConfig(field_display_units=unit),
        target_field=target,
        field_by_layout=field.copy(),
        b_field_opt_sf=field.copy(),
        diagnostic_cache={},
        coil_parts=[],
    )
    return SimpleNamespace(
        channel=SimpleNamespace(name=name, display_label=name, safe_name=name, color=None),
        result=result,
    )


def test_project_sh_comparison_uses_common_physical_basis_and_units():
    channels = (_channel("Gz small", 0.04, "T"), _channel("Gz large", 0.08, "Hz"))
    before = [channel.result.field_by_layout.copy() for channel in channels]
    fits, unavailable = _project_channel_sh_fits(SimpleNamespace(channel_runs=channels), max_order=2)
    assert not unavailable
    first, second = (pair[1] for pair in fits)
    assert first.unit == second.unit == "T"
    np.testing.assert_array_equal(first.center, second.center)
    assert first.radius == second.radius
    np.testing.assert_allclose(first.coefficients, second.coefficients, atol=1e-17, rtol=1e-12)
    assert second.field_rms > first.field_rms  # Each channel retains its own DSV samples.
    for original, channel in zip(before, channels):
        np.testing.assert_array_equal(channel.result.field_by_layout, original)


def test_project_sh_comparison_reports_unavailable_channel():
    bad = _channel("missing field", 0.04, "T")
    bad.result.field_by_layout[:] = np.nan
    fits, unavailable = _project_channel_sh_fits(
        SimpleNamespace(channel_runs=(bad, _channel("valid field", 0.04, "T"))), max_order=2
    )
    assert len(fits) == 1
    assert unavailable and unavailable[0].startswith("missing field:")


def test_project_channels_have_distinct_default_colors_and_preserve_custom_colors():
    channels = tuple(_channel(name, 0.04, "T") for name in ("Gx", "Gy", "Gz"))
    run = SimpleNamespace(channel_runs=channels)
    assert len({_channel_color(run, channel.channel.name) for channel in channels}) == 3
    channels[1].channel.color = "#123456"
    assert _channel_color(run, "Gy") == "#123456"


def test_project_multipart_labels_preserve_distinguishing_leaf_names():
    spec = SimpleNamespace(title="Gx: Coil geometry", key="geometry", channel="Gx")
    first = _project_plot_book_short_name(spec, "Gx: Coil geometry long diagnostic name first part", 1, 2)
    second = _project_plot_book_short_name(spec, "Gx: Coil geometry long diagnostic name second part", 2, 2)
    assert "first part" in first and "second part" in second
    assert first.endswith("1/2") and second.endswith("2/2")
    assert not first.startswith("Gx:")


def test_channel_factory_handles_tuple_and_empty_figures():
    import matplotlib.pyplot as plt
    from coilgen.matplotlib_visualization import MatplotlibPlotSpec

    figure = plt.figure()
    figure.set_label("First part")
    try:
        original = MatplotlibPlotSpec("geometry", "Geometry", "layout", lambda: (figure, None))
        wrapped = _channel_prefixed_plot_spec(MatplotlibPlotSpec, _channel("Gx", 0.04, "T"), original)
        assert wrapped.factory() == (figure, None)
        assert figure.get_label() == "Gx: First part"
    finally:
        plt.close(figure)


def test_headless_project_book_respects_project_include_and_backend():
    channel = _channel("Gx", 0.04, "T")
    run = SimpleNamespace(
        project=SimpleNamespace(name="headless", plot_include=("wire",)),
        channel_runs=(channel,),
        summary_rows=[],
        results=(channel.result,),
    )
    with (
        patch("coilgen.matplotlib_visualization.activate_matplotlib_interactive_backend") as activate,
        patch("coilgen.matplotlib_visualization.build_result_plot_specs", return_value=[]) as specs,
        patch("coilgen.matplotlib_visualization.MatplotlibPlotBookBuilder") as builder_type,
    ):
        builder_type.return_value.entries = [object()]
        book = plot_project_run_book(run, close_previous=False, show=False)
    activate.assert_not_called()
    assert specs.call_args.kwargs["include"] == ("wire",)
    assert book is builder_type.return_value.build.return_value
    book.show.assert_not_called()
