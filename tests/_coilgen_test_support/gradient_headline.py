"""Shared fixtures for gradient headline."""

import numpy as np
from coilgen.application.service import RunRecord


BODIES = ("part001_body0001", "part002_body0001")


FIELD_ARRAYS = ("target_points_m", "target_field_T", "fabricated_points_m", "fabricated_field_T")


def _metadata():
    """A shielded gradient run: primary and shield in series, the shield over its fringe limit."""
    current = 19.026
    return {
        "name": "Shielded Gy",
        "project_revision": 1,
        "config": {"target_field_component": "Bz", "design_context": "mr_gradient"},
        "summary": {
            "accepted": False,
            "sheet_accepted": True,
            "scope": "fabrication_review",
            "pipeline_stage": "evaluated",
            "design_context": "mr_gradient",
            "current_a": None,
            "resistance_ohm": None,
            "inductance_h": None,
            "target_gradient_t_per_m": 0.02,
            "layout_gradient_efficiency_mt_per_m_per_a": 1.02072,
            "layout_current_for_target_a": 19.594,
            "layout_series_resistance_ohm": 0.08452,
            "layout_series_inductance_h": 2.80248e-05,
        },
        "fabrication": {
            "ready": False,
            "electrical_metrics": {
                "bodies": [
                    {"body_id": BODIES[0], "current_A": current, "resistance_ohm": 0.0602, "length_m": 13.4},
                    {"body_id": BODIES[1], "current_A": current, "resistance_ohm": 0.0243, "length_m": 5.4},
                ]
            },
            "paths": [
                {
                    "body_id": body,
                    "diagnostics": {
                        "manufacturing_failure_reasons": "minimum_bend_radius_violation;curvature_envelope_violation"
                    },
                }
                for body in BODIES
            ],
        },
        "pipeline_sweeps": {
            "checks_passed": False,
            "bodies": [
                {
                    "body_id": "pipeline_" + body,
                    "diagnostics": {
                        "manufacturing_failure_reasons": "minimum_bend_radius_violation;support_surface_penetration"
                    },
                }
                for body in BODIES
            ],
        },
        "fabricated_physics": {
            "current_source": "original_contour_current_no_refit",
            "currents_A": {body: current for body in BODIES},
            "checks": [
                {
                    "name": "Fabricated path field error (original current)",
                    "verified": True,
                    "rms_error_T": 1e-6,
                    "sample_count": 257,
                    "component": "Bz",
                    "best_fit_current_scale": 1.1,
                    "best_fit_relative_rms_error": 0.0357,
                }
            ],
            "sampled_suppression": {
                "sample_count": 314,
                "rms_limit_T": 2e-05,
                "peak_limit_T": None,
                "sheet": {"available": True, "rms_T": 1.9999e-05, "rms_passed": True},
                "fabricated": {"available": True, "rms_T": 2.954e-05, "rms_passed": False},
            },
        },
    }


def _record(tmp_path, metadata=None, names=FIELD_ARRAYS):
    return RunRecord(tmp_path, metadata or _metadata(), {name: np.zeros((3, 3)) for name in names})
