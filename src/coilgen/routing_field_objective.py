"""Same-current longitudinal distortion relative to directed source contours."""

from __future__ import annotations

import numpy as np


def longitudinal_field_distortion(reference_per_amp, candidate_per_amp, target_field, b0_direction, current_a):
    """Return weighted DSV RMS and absolute peak of ``I * (candidate-reference) dot b0``.

    Only positive, finite-weight, nonsuppressed samples participate. Weights and
    field amplitudes are scaled before squaring, and B0 is normalized without
    overflowing its norm. No current is fitted and no mean offset is removed.
    Missing/invalid context and nonrepresentable results raise ``ValueError``;
    an unchecked domain must never look like zero distortion.
    """
    try:
        current = float(current_a)
        if np.ndim(current_a) != 0 or not np.isfinite(current) or current == 0:
            raise ValueError("A finite nonzero fixed current is required.")
        direction = np.asarray(b0_direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("B0 direction must contain three finite values.")
        direction_scale = float(np.max(np.abs(direction)))
        if direction_scale == 0:
            raise ValueError("B0 direction must be nonzero.")
        direction = direction / direction_scale
        direction /= np.linalg.norm(direction)

        reference = np.asarray(reference_per_amp, dtype=float)
        candidate = np.asarray(candidate_per_amp, dtype=float)
        coords = np.asarray(target_field.coords, dtype=float)
        weights = np.asarray(target_field.weights, dtype=float)
        suppressed = np.asarray(target_field.suppressed, dtype=bool)
        if coords.ndim != 2 or coords.shape[1:] != (3,):
            raise ValueError("Target coordinates must have shape (N, 3).")
        if reference.shape != coords.shape or candidate.shape != coords.shape:
            raise ValueError("Reference and candidate fields must match the target grid.")
        if weights.shape != (len(coords),) or suppressed.shape != weights.shape:
            raise ValueError("Weights and suppression membership must match target points.")
        active = np.isfinite(weights) & (weights > 0) & ~suppressed
        if not np.any(active):
            raise ValueError("No positive finite-weight DSV samples.")
        if not all(np.all(np.isfinite(values[active])) for values in (reference, candidate, coords)):
            raise ValueError("Active DSV coordinates and fields must be finite.")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            projected = ((candidate[active] - reference[active]) @ direction) * current
            peak = float(np.max(np.abs(projected)))
            sample_weights = weights[active] / np.max(weights[active])
            rms = (
                peak * float(np.sqrt(np.dot(sample_weights, (projected / peak) ** 2) / sample_weights.sum()))
                if peak > 0
                else 0.0
            )
        if not np.isfinite(rms) or not np.isfinite(peak):
            raise ValueError("Longitudinal distortion is not representable as a finite value.")
        return {
            "rms_T": rms,
            "peak_T": peak,
            "active_sample_count": int(np.count_nonzero(active)),
            "current_a": current,
            "current_refitted": False,
        }
    except (AttributeError, TypeError, FloatingPointError, OverflowError) as error:
        raise ValueError(f"Invalid longitudinal field context: {error}") from error
