"""Explicit DC homogenization of parallel wire tracks for a sheet objective."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class WireSheetMaterial:
    """Match sheet Joule loss to a locally uniform layer of parallel wires.

    For wire area A, track pitch d and bulk resistivity rho, K=I/d gives
    rho/t_eff * K**2 = rho*I**2/(A*d), hence t_eff=A/d. This is a directional
    DC homogenization along the wires, not an isotropic conductor equivalence.
    Variable spacing, leads, interconnects, skin/proximity effects, insulation
    and temperature changes require separate modeling. Final wire resistance
    remains rho*length/A and must be recomputed on the realized circuit.
    """

    wire_area_m2: float
    pitch_m: float
    resistivity_ohm_m: float = 1.8e-8

    def __post_init__(self):
        for name in ("wire_area_m2", "pitch_m", "resistivity_ohm_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive quantity in SI units.")

    @classmethod
    def round_wire(cls, diameter_m, pitch_m, *, resistivity_ohm_m=1.8e-8):
        if not np.isfinite(diameter_m) or diameter_m <= 0.0:
            raise ValueError("diameter_m must be a finite positive length.")
        if pitch_m < diameter_m:
            raise ValueError("Single-layer round-wire pitch cannot be smaller than the bare wire diameter.")
        return cls(np.pi * diameter_m**2 / 4.0, pitch_m, resistivity_ohm_m)

    @property
    def effective_thickness_m(self):
        return self.wire_area_m2 / self.pitch_m

    @property
    def sheet_resistance_ohm(self):
        return self.resistivity_ohm_m / self.effective_thickness_m

    def apply_to(self, config):
        """Return an explicit sheet-objective config; wire dimensions stay separate.

        ``specific_conductivity_conductor`` is the historical config name for
        bulk resistivity in ohm metres, despite the word 'conductivity'.
        """
        return replace(
            config,
            conductor_thickness=self.effective_thickness_m,
            specific_conductivity_conductor=self.resistivity_ohm_m,
        )

    def summary(self):
        return {
            "model": "single_layer_locally_parallel_uniform_pitch_dc",
            "wire_area_m2": self.wire_area_m2,
            "pitch_m": self.pitch_m,
            "resistivity_ohm_m": self.resistivity_ohm_m,
            "effective_thickness_m": self.effective_thickness_m,
            "sheet_resistance_ohm": self.sheet_resistance_ohm,
            "spacing_and_temperature_assumed_uniform": True,
            "leads_and_interconnects_included": False,
        }
