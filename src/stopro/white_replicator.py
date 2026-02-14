from typing import Any, Literal

import numpy as np

from ._utils import _as_vector
from .gillespie_replicator import gillespie_replicator


def white_replicator(
    T: float,
    dt: float | None = None,
    *,
    steps: int | None = None,
    N: int = 2,
    mu: float | np.ndarray = 1.0,
    sigma: float | np.ndarray = 1.0,
    initial_condition: np.ndarray | None = None,
    gap: int = 1,
    samples: int = 1,
    covariance: np.ndarray | None = None,
    mixing_matrix: np.ndarray | None = None,
    order: Literal["STD", "SDT"] = "STD",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    White replicator model.

    This is the Gillespie replicator with the drift adjusted so that the
    underlying log-process uses mu (i.e. shift Gillespie's mu by +0.5*sigma^2).

    order : {"STD","SDT"}, default="STD"
        Output array layout for X:
        - "STD": (samples, time, dim)  [default, plot-friendly]
        - "SDT": (samples, dim, time)  [legacy]
    """
    N = int(N)
    mu = _as_vector(mu, N, "mu")
    sigma = _as_vector(sigma, N, "sigma")

    mu_shifted = mu + 0.5 * sigma**2

    return gillespie_replicator(
        T,
        dt,
        steps=steps,
        N=N,
        mu=mu_shifted,
        sigma=sigma,
        initial_condition=initial_condition,
        gap=gap,
        samples=samples,
        covariance=covariance,
        mixing_matrix=mixing_matrix,
        order=order,
        seed=seed,
    )