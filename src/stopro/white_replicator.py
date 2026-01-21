import numpy as np

from ._utils import _as_vector
from .gillespie_replicator import gillespie_replicator


def white_replicator(
    T,
    dt=None,
    *,
    steps=None,
    N=2,
    mu=1.0,
    sigma=1.0,
    initial_condition=None,
    gap=1,
    samples=1,
    covariance=None,
    mixing_matrix=None,
):
    """
    White replicator model.

    This is the Gillespie replicator with the drift adjusted so that the
    underlying log-process uses mu (i.e. shift Gillespie's mu by +0.5*sigma^2).
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
    )