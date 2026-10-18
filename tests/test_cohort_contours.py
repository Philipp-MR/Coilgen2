from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.cohort_contours import (
    ContourChannelOptions,
    ContourDictionary,
    ContourFamily,
    PhysicalContour,
    contour_dictionary_from_coilgen_result,
    design_contour_dictionary_channels,
    evaluate_contour_family_fields,
    load_contour_dictionary,
    save_contour_channel_design,
    save_contour_dictionary,
)
from coilgen.cohort_shim import CohortFitOptions, PopulationShimDataset, RigidTransform, ShimScenario
from coilgen.models import ContourLine, ContourSparsificationResult
from coilgen.cohort_contours import _canonical_surjections, _stirling_partition_count


def _closed_triangle(x: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            [x, -0.01, -0.01],
            [x, 0.01, -0.01],
            [x, 0.0, 0.01],
            [x, -0.01, -0.01],
        ]
    )


def _dictionary(family_count: int = 6, frame: str = "scanner") -> ContourDictionary:
    return ContourDictionary(
        tuple(
            ContourFamily(
                f"f{index}",
                (PhysicalContour(_closed_triangle(0.001 * index), 0, index),),
            )
            for index in range(family_count)
        ),
        frame,
    )


def _identity_transform(source: str = "mask_cad") -> RigidTransform:
    return RigidTransform(source, "scanner", np.eye(4))


def _dataset(disturbances: np.ndarray) -> PopulationShimDataset:
    coords = np.column_stack((np.arange(disturbances.shape[1]), np.zeros((disturbances.shape[1], 2))))
    scenarios = tuple(
        ShimScenario(
            scenario_id=f"s{index}",
            subject_id=f"subject_{index}",
            coords_scanner_m=coords,
            disturbance_T=disturbance,
            scanner_from_mask=_identity_transform(),
        )
        for index, disturbance in enumerate(disturbances)
    )
    return PopulationShimDataset(scenarios)


def test_dictionary_roundtrip_uses_ragged_numeric_arrays_without_pickle(tmp_path) -> None:
    dictionary = _dictionary()

    path = save_contour_dictionary(tmp_path / "dictionary.npz", dictionary)
    restored = load_contour_dictionary(path)

    assert restored.coordinate_frame == "scanner"
    assert restored.family_count == 6
    assert restored.contour_count == 6
    np.testing.assert_allclose(restored.families[3].contours[0].points_m, _closed_triangle(0.003))


def test_factory_uses_physical_sources_not_weighted_representatives() -> None:
    source = [ContourLine(_closed_triangle(0.001 * index), float(index)) for index in range(4)]
    weighted_representatives = [
        ContourLine(_closed_triangle(0.1), 0.5, turn_multiplicity=2),
        ContourLine(_closed_triangle(0.2), 2.5, turn_multiplicity=2),
    ]
    part = SimpleNamespace(
        contour_lines=weighted_representatives,
        contour_sparsification=ContourSparsificationResult(
            source_contours=source,
            final_source_contour_indices=[(0, 1), (2, 3)],
        ),
    )
    result = SimpleNamespace(
        coil_parts=[part],
        combined_mesh=SimpleNamespace(metadata={"coordinate_units": "m"}),
    )

    dictionary = contour_dictionary_from_coilgen_result(result, coordinate_frame="mask_cad")

    assert dictionary.family_count == 2
    assert dictionary.contour_count == 4
    assert [len(family.contours) for family in dictionary.families] == [2, 2]
    assert float(np.max(dictionary.families[0].contours[0].points_m[:, 0])) < 0.01
    assert dictionary.metadata["weighted_contour_representatives_used"] is False


def test_mask_frame_field_evaluation_applies_registered_rigid_transform() -> None:
    dictionary_mask = _dictionary(1, "mask_cad")
    transform_matrix = np.eye(4)
    transform_matrix[0, 3] = 0.05
    transform = RigidTransform("mask_cad", "scanner", transform_matrix)
    scanner_family = ContourFamily(
        "f0",
        (PhysicalContour(transform.apply_points(dictionary_mask.families[0].contours[0].points_m)),),
    )
    dictionary_scanner = ContourDictionary((scanner_family,), "scanner")
    coords = np.asarray([[0.07, 0.0, 0.0], [0.07, 0.0, 0.02]])
    scenario = ShimScenario("s0", "p0", coords, np.zeros(2), transform)
    dataset = PopulationShimDataset((scenario,))

    mask_fields = evaluate_contour_family_fields(
        dictionary_mask,
        dataset,
        scanner_from_source=transform,
    )
    scanner_fields = evaluate_contour_family_fields(dictionary_scanner, dataset)

    np.testing.assert_allclose(mask_fields["s0"], scanner_fields["s0"], rtol=1.0e-12, atol=1.0e-18)


def test_mask_frame_field_evaluation_rejects_mixed_scenario_registrations() -> None:
    dictionary = _dictionary(1, "mask_cad")
    transform = _identity_transform()
    shifted_matrix = np.eye(4)
    shifted_matrix[0, 3] = 0.001
    shifted = RigidTransform("mask_cad", "scanner", shifted_matrix)
    coords = np.asarray([[0.03, 0.0, 0.0], [0.03, 0.0, 0.02]])
    dataset = PopulationShimDataset(
        (
            ShimScenario("s0", "p0", coords, np.zeros(2), transform),
            ShimScenario("s1", "p1", coords, np.zeros(2), shifted),
        )
    )

    with pytest.raises(ValueError, match="all scenarios"):
        evaluate_contour_family_fields(
            dictionary,
            dataset,
            scanner_from_source=transform,
        )


def test_exact_six_family_four_channel_search_is_complete_and_disjoint(monkeypatch, tmp_path) -> None:
    basis = np.vstack((np.eye(4), -np.ones((1, 4))))
    family_fields = np.column_stack(
        (
            basis[:, 0],
            basis[:, 1],
            basis[:, 2],
            basis[:, 3],
            0.2 * basis[:, 0] + 0.1 * basis[:, 2],
            0.1 * basis[:, 1] - 0.2 * basis[:, 3],
        )
    )
    disturbances = np.vstack((basis.T, 0.5 * basis.T))
    dataset = _dataset(disturbances)

    def fake_fields(dictionary, dataset, **kwargs):
        return {scenario.scenario_id: family_fields for scenario in dataset.scenarios}

    monkeypatch.setattr("coilgen.cohort_contours.evaluate_contour_family_fields", fake_fields)
    result = design_contour_dictionary_channels(
        dataset,
        _dictionary(),
        ContourChannelOptions(
            channel_count=4,
            current_limit_A=2.0,
            fit_options=CohortFitOptions(center_frequency=False),
            optimize_family_polarity=True,
        ),
    )

    # S(6, 4) = 65 canonical partitions and 2^(6-4) relative-polarity states.
    assert result.partition_count == 65
    assert result.candidate_count == 260
    assert result.feasible_candidate_count > 0
    assert sorted(np.unique(result.family_to_channel).tolist()) == [0, 1, 2, 3]
    assert len(result.family_to_channel) == 6
    assert result.diagnostics.channel_numerical_rank == 4
    assert result.routing_validated is False
    assert result.summary()["single_layer_realization_claimed"] is False

    outputs = save_contour_channel_design(tmp_path, result)
    assert len(outputs) == 6
    exported_families: list[int] = []
    for path in outputs[:4]:
        with np.load(path, allow_pickle=False) as payload:
            exported_families.extend(payload["assigned_family_indices"].tolist())
            assert not bool(payload["routing_validated"])
    assert sorted(exported_families) == list(range(6))


@pytest.mark.parametrize("family_count", range(1, 9))
def test_exact_partition_count_matches_enumeration(family_count) -> None:
    for channels in range(1, family_count + 1):
        assert _stirling_partition_count(family_count, channels) == sum(
            1 for _ in _canonical_surjections(family_count, channels)
        )


def test_large_dictionary_guard_runs_before_fields_or_partition_enumeration(monkeypatch) -> None:
    def unexpected(*_args, **_kwargs):
        pytest.fail("An over-budget search must stop before field evaluation or enumeration")

    monkeypatch.setattr("coilgen.cohort_contours.evaluate_contour_family_fields", unexpected)
    monkeypatch.setattr("coilgen.cohort_contours._canonical_surjections", unexpected)
    exact_count = 907413148520234062433363985673077391360
    with pytest.raises(ValueError, match=str(exact_count)):
        design_contour_dictionary_channels(_dataset(np.zeros((1, 5))), _dictionary(46))


def test_guard_includes_asymmetric_power_channel_labelings(monkeypatch) -> None:
    monkeypatch.setattr(
        "coilgen.cohort_contours.evaluate_contour_family_fields",
        lambda *_args, **_kwargs: pytest.fail("Power channel labelings must be counted before fields"),
    )
    # S(4, 3) * 2^(4-3) * 3! = 72; symmetry cannot remove unequal-resistance labelings.
    with pytest.raises(ValueError, match="72 candidates"):
        design_contour_dictionary_channels(
            _dataset(np.zeros((1, 5))),
            _dictionary(4),
            ContourChannelOptions(
                channel_count=3,
                channel_resistance_ohm=(1.0, 2.0, 3.0),
                total_power_limit_W=1.0,
                maximum_candidate_count=71,
            ),
        )
