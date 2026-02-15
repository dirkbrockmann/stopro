from typing import Any, Literal

import numpy as np
from scipy.special import logsumexp

from ._utils import _as_vector, _simplex_initial_condition
from .integrated_ornstein_uhlenbeck import integrated_ornstein_uhlenbeck


def colored_replicator(
    T: float,
    dt: float | None = None,
    *,
    steps: int | None = None,
    N: int = 2,
    mu: float | np.ndarray = 1.0,
    sigma: float | np.ndarray = 1.0,
    tau: float | np.ndarray = 1.0,
    initial_condition: np.ndarray | None = None,
    gap: int = 1,
    samples: int = 1,
    covariance: np.ndarray | None = None,
    mixing_matrix: np.ndarray | None = None,
    order: Literal["STD", "SDT"] = "STD",
    seed: int | None = None,
) -> dict[str, Any]:
    r"""
    Simulate the colored stochastic replicator (softmax-normalized colored log-process).

    Model:
        dX_i(t) = mu_i X_i(t) dt + sigma_i X_i(t) Z_i(t) dt
        tau_i dZ_i(t) = -Z_i(t) dt + dW_i(t)

    With I_i(t) = ∫_0^t Z_i(s) ds:
        X_i(t) = X_i(0) * exp(mu_i t + sigma_i I_i(t))
        Y_i(t) = X_i(t) / sum_j X_j(t)

    Notes
    -----
    The underlying OU Z is started at zero (not stationary) to make comparisons to
    Wiener-driven models consistent as tau -> 0.

    order : {"STD","SDT"}, default="STD"
        Output array layout for X:
        - "STD": (samples, time, dim)  [default, plot-friendly]
        - "SDT": (samples, dim, time)  [legacy]

    Returns
    -------
    dict with keys including:
      - "X": shape depends on `order` (replicator trajectory on the saved grid)
      - "t": (K,) time grid (subsampled by gap)
      - plus metadata from integrated_ornstein_uhlenbeck, and "mu","sigma","tau","initial_condition","order"
    """
    N = int(N)
    if N < 2:
        raise ValueError(f"colored_replicator requires N>=2, got N={N}.")

    gap = int(gap)
    if gap <= 0:
        raise ValueError("gap must be a positive integer.")

    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be a positive integer.")

    mu = _as_vector(mu, N, "mu")
    sigma = _as_vector(sigma, N, "sigma")
    tau = _as_vector(tau, N, "tau")
    if np.any(tau <= 0):
        raise ValueError("tau must be > 0 (component-wise).")

    # Replicator starts on simplex
    x0 = _simplex_initial_condition(initial_condition, N=N)

    # Choose OU stdev so that tau dZ = -Z dt + dW  <=>  stationary stdev(Z) = 1/sqrt(2*tau)
    # (even though we start Z(0)=0, this keeps the intended tau-scaling of the colored noise)
    stdev_z = 1.0 / np.sqrt(2.0 * tau)

    # I(t) = ∫ Z ds (computed on fine grid internally; returned on gap-grid)
    # We intentionally start OU at zero for comparability (initial_condition=None).
    # Force SDT internally for the component-centric math; convert once at the boundary.
    res_I = integrated_ornstein_uhlenbeck(
        T,
        dt,
        steps=steps,
        gap=gap,
        N=N,
        samples=samples,
        stdev=stdev_z,
        timescale=tau,
        initial_condition="stationary",
        covariance=covariance,
        mixing_matrix=mixing_matrix,
        order="SDT",
        seed=seed,  # (samples, dim, time) for convenient math below
    )

    I = res_I["X"]  # (samples, N, K)
    t = res_I["t"]  # (K,)

    # log X_i(t) = log x0_i + mu_i t + sigma_i I_i(t)
    # Normalize with logsumexp for numerical stability.
    with np.errstate(divide="ignore"):
        logx0 = np.where(x0 > 0, np.log(x0), -np.inf)  # (N,)

    logX = (
        logx0[None, :, None]
        + mu[None, :, None] * t[None, None, :]
        + sigma[None, :, None] * I
    )  # (samples, N, K)

    log_denom = logsumexp(logX, axis=1, keepdims=True)  # (samples, 1, K)
    Y_sdt = np.exp(logX - log_denom)                    # (samples, N, K)

    # Convert once at the boundary
    if order == "STD":
        Y_out = np.moveaxis(Y_sdt, 1, 2)  # (samples, dim, time) -> (samples, time, dim)
    elif order == "SDT":
        Y_out = Y_sdt
    else:
        raise ValueError("order must be 'STD' or 'SDT'")

    res = dict(res_I)
    res["X"] = Y_out
    res["mu"] = mu
    res["sigma"] = sigma
    res["tau"] = tau
    res["initial_condition"] = x0
    res["order"] = order
    return res