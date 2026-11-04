"""Choose crossing repairs using the mounted copper, in the export coordinate frame."""

from dataclasses import replace
from hashlib import sha256

import numpy as np

from .crossing_lifts import apply_crossing_lifts, apply_lift_proposal
from .group_route_validation import _isolated_part
from .layout import cleanup_wire_paths
from .manufacturing import create_swept_coil_meshes
from .mounted_crossing_lifts import mounted_gap_proposal, mounted_lift_proposal
from .crossing_conditioning import conditioned_lift
from .paired_crossing_lifts import paired_witness_lift_proposal, smooth_lift_ceiling


def crossing_repair_enabled(config):
    return (
        not (config.skip_sweep or config.contour_only_sweep_mode or config.skip_postprocessing)
        and not config.skip_normal_shift
        and bool(config.normal_shift_length)
        and str(config.sweep_self_intersection_check).lower() != "off"
    )


def _clear(part):
    diagnostics = part.manufacturing_diagnostics
    return (
        diagnostics.get("exact_collision_completed", False)
        and diagnostics.get("mesh_solid_ok", False)
        and not diagnostics.get("copper_clearance_failure", True)
    )


def _export_clear(part, config):
    if not _clear(part):
        return False
    shift = part.normal_shift_diagnostics
    heights = np.asarray(part.normal_shift_provenance.get("lift_distance_m", ())) * np.sign(config.normal_shift_length)
    if len(heights) and (
        not np.isfinite(heights).all()
        or heights.min() < -1e-9
        or heights.max() > abs(config.normal_shift_length) + 1e-9
    ):
        return False
    if shift.get("normal_shift_method") == "arclength_crossing_layers" and (
        shift.get("normal_shift_centerline_intersection_count", 0)
        or shift.get("normal_shift_introduced_backtrack_count", 0)
        or part.route_diagnostics.get("wire_clearance_failure_count", 0)
        or part.route_diagnostics.get("inter_group_connector_clearance_violation_count", 0)
    ):
        return False
    if config.sweep_surface_realization == "mesh":
        return True
    # Fabrication may remount the same source path on the triangle support.
    export = _isolated_part(part)
    create_swept_coil_meshes([export], replace(config, sweep_surface_realization="mesh"))
    return _clear(export)


def _score(trial):
    d = trial.manufacturing_diagnostics
    return (
        not d.get("exact_collision_completed", False),
        d.get("exact_intersection_count", 0),
        d.get("copper_gap_violating_facet_pairs", 0),
    )


def _realize(trial, config):
    cleanup_wire_paths([trial], config=config)
    create_swept_coil_meshes([trial], config)


def _modified(trial, config, proposal):
    if proposal is None:
        return None
    modified = _isolated_part(trial)
    if not apply_lift_proposal(modified, config, proposal):
        return None
    _realize(modified, config)
    return modified


def _paired_visits(trial, config):
    for reverse in (False, True):
        proposal = paired_witness_lift_proposal(trial, config, reverse=reverse)
        candidate = _modified(trial, config, proposal)
        if candidate is not None:
            yield candidate


def _refine(trial, config):
    from .crossing_relaxation import improve_crossing_candidate

    for _ in range(3):
        if _clear(trial):
            break
        relaxed = improve_crossing_candidate(trial, config, max_iterations=4)
        if relaxed is None or _score(relaxed) >= _score(trial):
            break
        trial = relaxed if _clear(relaxed) else _condition(relaxed, config)
    return trial


def _accepted(trial, incumbent):
    trial.normal_shift_diagnostics["normal_shift_physical_clearance_status"] = "verified_swept_copper"
    trial.joint_route_diagnostics = {
        **incumbent.joint_route_diagnostics,
        "postselection_copper_repair": True,
        "postselection_evidence": "joint field comparisons refer to the pre-repair route; final fields use the repaired conductor",
    }
    return trial


def _condition(trial, config):
    def rank(part):
        # Once the copper clears, prefer a valid loft and successively reduce
        # short edges without relaxing the original solid/quality tolerances.
        d = part.manufacturing_diagnostics
        return (*_score(part), not d.get("mesh_solid_ok", False), d.get("short_edge_count", 0))

    for _ in range(5):
        for scale, include_tight in ((1.0, False), (0.7, False), (1.5, False), (0.7, True)):
            proposal = conditioned_lift(trial, config, scale=scale, include_tight_bends=include_tight)
            if proposal is None:
                continue
            rounded = _isolated_part(trial)
            if not apply_lift_proposal(rounded, config, proposal):
                continue
            _realize(rounded, config)
            if rank(rounded) < rank(trial):
                trial = rounded
                break
        else:
            break
        if _clear(trial):
            break
    return trial


def repair_swept_crossings(parts, config):
    """Keep an already clear winding. Adopt a new route only after actual-body checks."""
    if not crossing_repair_enabled(config):
        return parts
    trial_config = replace(config, fail_on_route_quality_issues=False)
    result = []
    for part in parts:
        result.append(part)
        if part.wire_path is None or not any(
            o.kind in {"inter_group_connector", "within_group_return"} or o.connector_role == "joint_spiral_join"
            for o in part.wire_path.segment_owners
        ):
            continue
        if part.swept_body is None or "copper_clearance_failure" not in part.manufacturing_diagnostics:
            create_swept_coil_meshes([part], trial_config)
        if _export_clear(part, trial_config):
            continue
        failed = []
        raw_seeds = []
        seen = set()
        for trial in _proposals(part, trial_config):
            digest = sha256(trial.wire_path.v.tobytes())
            for owner, geometry in zip(trial.wire_path.segment_owners, trial.wire_path.segment_geometry):
                digest.update(repr((owner, geometry.value)).encode())
            signature = digest.digest()
            if signature in seen:
                continue
            seen.add(signature)
            paired_cut = "paired_cut_levels" in trial.normal_shift_provenance
            if trial.swept_body is None or "copper_clearance_failure" not in trial.manufacturing_diagnostics:
                _realize(trial, trial_config)
            if not _clear(trial):
                raw_seeds.append(trial)
                trial = _condition(trial, trial_config)
            witnesses = trial.manufacturing_diagnostics.get("copper_clearance_witness_intervals", ())
            contacts = trial.manufacturing_diagnostics.get("exact_intersection_count", 0)
            if witnesses and ((len(witnesses) <= 2 and not contacts) or (paired_cut and contacts <= 100)):
                trial = _refine(trial, trial_config)
            if _export_clear(trial, trial_config):
                result[-1] = _accepted(trial, part)
                break
            if trial.manufacturing_diagnostics.get("copper_clearance_witness_intervals"):
                failed.append(trial)
        else:
            # When a first repair leaves two visits on the same layer, their
            # measured copper witnesses identify which branches must separate.
            for seed in sorted(raw_seeds, key=_score):
                broad = _modified(seed, trial_config, conditioned_lift(seed, trial_config, scale=1.5))
                if broad is not None:
                    if _export_clear(broad, trial_config):
                        result[-1] = _accepted(broad, part)
                        break
                    failed.append(broad)
            if result[-1] is not part:
                continue
            for seed in sorted(failed, key=_score):
                for paired in _paired_visits(seed, trial_config):
                    if _export_clear(paired, trial_config):
                        result[-1] = _accepted(paired, part)
                        break
                if result[-1] is not part:
                    break
            if result[-1] is not part:
                continue

            for trial in sorted(failed, key=_score):
                trial = _refine(trial, trial_config)
                if _export_clear(trial, trial_config):
                    result[-1] = _accepted(trial, part)
                    break
    return result


def _limited_two_layer_trials(part, config):
    """Solve the upper/lower ordering when a stacked winding has less than two conductor heights."""
    cap = abs(config.normal_shift_length)
    two_layers = 2 * config.conductor_cross_section_height
    if not 0.75 * two_layers <= cap < two_layers:
        return
    for explore in (False, True):
        # The wider exploratory seed determines the order of the two visits.
        # Only a newly mounted route within the requested signed limit may be returned.
        working = (
            replace(config, normal_shift_length=np.copysign(two_layers, config.normal_shift_length))
            if explore
            else config
        )
        seed = _isolated_part(part)
        if not apply_crossing_lifts(
            seed,
            working,
            ramp_scale=1.4 if explore else 1.0,
            clearance_factor=1.5 if explore else 1.12,
            separate_joins=True,
            straighten_returns=True,
            round_returns=True,
            multiple_layers=True,
        ):
            continue
        _realize(seed, working)
        if not explore and _export_clear(seed, config):
            yield seed
            return
        broad = _modified(seed, working, conditioned_lift(seed, working, scale=1.5))
        if broad is None:
            continue
        if not explore:
            if _export_clear(broad, config):
                yield broad
                return
            continue
        for reverse in (True, False):
            paired = paired_witness_lift_proposal(broad, working, reverse=reverse)
            if paired is not None:
                bounded = _modified(broad, config, smooth_lift_ceiling(paired, config))
                if bounded is not None and _export_clear(bounded, config):
                    yield bounded
                    return


def _proposals(part, config):
    from .layout import _normal_shift_physical_clearance

    yield from _limited_two_layer_trials(part, config)
    if not part.manufacturing_diagnostics.get("exact_intersection_count", 0):
        proposal = mounted_gap_proposal(part, config)
        if proposal is not None:
            trial = _isolated_part(part)
            if apply_lift_proposal(trial, config, proposal):
                yield trial

    def mounted(reverse):
        proposal = mounted_lift_proposal(part, config, reverse=reverse)
        if proposal is not None:
            trial = _isolated_part(part)
            if apply_lift_proposal(trial, config, proposal):
                yield trial

    # Prefer preserving existing lifts if most remote witnesses lie on the
    # two sides of one or more connection cuts.
    f = part.fabrication_path
    same_cut = remote = 0
    envelope = _normal_shift_physical_clearance(config)
    if f is not None:
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(f.wire_points_m, axis=0), axis=1))]
        for i, j, _ in part.manufacturing_diagnostics.get("copper_clearance_witness_intervals", ()):
            distance = abs(arc[j] - arc[i])
            if f.closed:
                distance = min(distance, arc[-1] - distance)
            if distance <= 4 * envelope:
                continue
            remote += 1
            a, b = f.segment_owners[i], f.segment_owners[j]
            same_cut += (
                a.kind == b.kind == "inter_group_connector"
                and a.connector_index == b.connector_index
                and a.connector_role != b.connector_role
            )
    prefer_mounted = bool(remote and 2 * same_cut >= remote)
    if prefer_mounted:
        yield from mounted(False)
    if abs(config.normal_shift_length) >= 2 * envelope:
        for round_returns in (True, False):
            trial = _isolated_part(part)
            if apply_crossing_lifts(
                trial,
                config,
                multiple_layers=True,
                separate_joins=True,
                straighten_returns=True,
                round_returns=round_returns,
            ):
                yield trial
    from .paired_cut_lifts import paired_cut_proposals

    yield from paired_cut_proposals(part, config)
    for ramp, factor, separate, shorten, protect in (
        (1.0, 1.12, True, False, False),
        (1.0, 1.12, False, False, False),
        (0.70, 1.5, False, False, False),
        (0.85, 1.45, True, True, False),
        (1.4, 1.5, True, True, False),
        (1.0, 1.12, True, True, True),
        (1.0, 1.12, True, True, False),
    ):
        trial = _isolated_part(part)
        if apply_crossing_lifts(
            trial,
            config,
            ramp_scale=ramp,
            clearance_factor=factor,
            separate_joins=separate,
            straighten_returns=shorten,
            round_returns=shorten,
            protect_contours=protect,
        ):
            yield trial
    if not prefer_mounted:
        yield from mounted(False)
    yield from mounted(True)
