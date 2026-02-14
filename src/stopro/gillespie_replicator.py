from typing import Any, Literal

import numpy as np
from scipy.special import logsumexp

from ._utils import _as_vector, _simplex_initial_condition
from .wiener import wiener


def gillespie_replicator(
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
    order: Literal["STD", "SDT"] = "STD",  # "STD" (samples, time, dim) or "SDT" (samples, dim, time),
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Gillespie replicator model (softmax normalization of correlated GBMs).

    Construct GBMs
        X_i(t) = X_i(0) * exp((mu_i - 0.5*sigma_i^2) t + sigma_i W_i(t))
    and normalize on the simplex:
        Y_i(t) = X_i(t) / sum_j X_j(t)

    Uses a numerically-stable logsumexp (softmax) normalization.

    order : {"STD","SDT"}, default="STD"
        Output array layout for X:
        - "STD": (samples, time, dim)  [default, plot-friendly]
        - "SDT": (samples, dim, time)  [legacy]
    """
    N = int(N)
    if N < 2:
        raise ValueError(f"gillespie_replicator requires N>=2, got N={N}.")

    gap = int(gap)
    if gap <= 0:
        raise ValueError("gap must be a positive integer.")

    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be a positive integer.")

    # Initial condition on simplex (length N, sums to 1)
    x0 = _simplex_initial_condition(initial_condition, N=N)

    # Force mu/sigma to vectors of length N
    mu = _as_vector(mu, N, "mu")
    sigma = _as_vector(sigma, N, "sigma")

    # Correlated Wiener driver on the *returned* grid (gap handled here)
    # Force wiener to return a known internal layout for this function.
    res_w = wiener(
        T,
        dt,
        steps=steps,
        gap=gap,
        N=N,
        samples=samples,
        covariance=covariance,
        mixing_matrix=mixing_matrix,
        order="SDT",
        seed=seed,  # (samples, dim, time) for convenient math below
    )

    W = res_w["X"]  # (samples, N, K)
    t = res_w["t"]  # (K,)

    # log(x0); allow zeros -> -inf
    with np.errstate(divide="ignore"):
        logx0 = np.where(x0 > 0, np.log(x0), -np.inf)  # (N,)

    drift = (mu - 0.5 * sigma**2)[None, :, None] * t[None, None, :]  # (1, N, K)
    noise = sigma[None, :, None] * W                                 # (samples, N, K)
    logX = logx0[None, :, None] + drift + noise                      # (samples, N, K)

    log_denom = logsumexp(logX, axis=1, keepdims=True)                # (samples, 1, K)
    Y_sdt = np.exp(logX - log_denom)                                  # (samples, N, K)

    # Convert once at the boundary
    if order == "STD":
        Y_out = np.moveaxis(Y_sdt, 1, 2)  # (samples, dim, time) -> (samples, time, dim)
    elif order == "SDT":
        Y_out = Y_sdt
    else:
        raise ValueError("order must be 'STD' or 'SDT'")

    # Build result dict in the same style as your other processes
    res = dict(res_w)
    res["X"] = Y_out
    res["mu"] = mu
    res["sigma"] = sigma
    res["initial_condition"] = x0
    res["order"] = order

    # Standardize covariance naming (wiener returns 'covariance' in this codebase)
    res["noise_covariance"] = res.get("covariance", None)
    if "covariance" in res:
        del res["covariance"]

    return res