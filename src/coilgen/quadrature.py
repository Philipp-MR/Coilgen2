from __future__ import annotations

import numpy as np


def gauss_legendre_integration_points_triangle(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta, weights = np.polynomial.legendre.leggauss(int(order))
    u_values: list[float] = []
    v_values: list[float] = []
    c_values: list[float] = []
    for eta_i, weight_i in zip(eta, weights):
        for eta_j, weight_j in zip(eta, weights):
            u_values.append((1.0 + eta_i) / 2.0)
            v_values.append((1.0 - eta_i) * (1.0 + eta_j) / 4.0)
            c_values.append(((1.0 - eta_i) / 8.0) * weight_i * weight_j)
    return np.asarray(u_values), np.asarray(v_values), np.asarray(c_values)
