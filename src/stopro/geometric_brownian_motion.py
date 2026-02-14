from typing import Any, Literal

import numpy as np

from ._utils import _as_vector
from .wiener import wiener


def geometric_brownian_motion(
    T: float,
    dt: float | None = None,
    *,
    steps: int | None = None,
    gap: int = 1,
    N: int = 1,
    samples: int = 1,
    mu: float | np.ndarray = 1.0,
    sigma: float | np.ndarray = 1.0,
    initial_condition: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    mixing_matrix: np.ndarray | None = None,
    order: Literal["STD", "SDT"] = "STD",  # "STD" (samples, time, dim) or "SDT" (samples, dim, time),
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Simulate (possibly multivariate) geometric Brownian motion (GBM) on [0, T].

        dX_i(t) = mu_i X_i(t) dt + sigma_i X_i(t) dW_i(t),

    where W is an N-dimensional Wiener process (optionally correlated via `covariance`
    or `mixing_matrix`, mutually exclusive).

    Provide exactly one of `dt` or `steps`. Use `gap>1` to subsample returned points.

    order : {"STD","SDT"}, default="STD"
        Output array layout for X:
        - "STD": (samples, time, dim)  [default, plot-friendly]
        - "SDT": (samples, dim, time)  [legacy]

    Returns
    -------
    dict
        Keys: 'X' (shape depends on `order`), 't' (savedsteps+1,), 'dt', 'steps',
        'savedsteps', 'gap', 'N', 'mu', 'sigma', 'initial_condition', 'noise_covariance', 'order'.
    """
    # Generate Wiener paths on the saved grid.
    # Use SDT internally here because the GBM formula is component-centric.
    res_w = wiener(
        T,
        dt=dt,
        steps=steps,
        gap=gap,
        N=N,
        samples=samples,
        covariance=covariance,
        mixing_matrix=mixing_matrix,
        order="SDT",  # (samples, dim, time)
        seed=seed,
    )

    W = res_w["X"]  # (samples, N, K)
    t = res_w["t"]  # (K,)
    N_res = res_w["N"]

    mu = _as_vector(mu, N_res, "mu")
    sigma = _as_vector(sigma, N_res, "sigma")

    if initial_condition is None:
        x0 = np.ones(N_res, dtype=float)
    else:
        x0 = _as_vector(initial_condition, N_res, "initial_condition")

    # Closed-form transform:
    # X_i(t) = x0_i * exp((mu_i - 0.5 sigma_i^2) t + sigma_i W_i(t))
    drift = (mu - 0.5 * sigma**2)[None, :, None]  # (1, N, 1)
    diff = sigma[None, :, None]                   # (1, N, 1)
    tt = t[None, None, :]                         # (1, 1, K)

    X_sdt = x0[None, :, None] * np.exp(drift * tt + diff * W)  # (samples, N, K)

    # Convert once at the boundary
    if order == "STD":
        X_out = np.moveaxis(X_sdt, 1, 2)  # (samples, dim, time) -> (samples, time, dim)
    elif order == "SDT":
        X_out = X_sdt
    else:
        raise ValueError("order must be 'STD' or 'SDT'")

    return {
        "X": X_out,
        "t": t,
        "dt": res_w["dt"],
        "steps": res_w["steps"],
        "savedsteps": res_w["savedsteps"],
        "gap": res_w["gap"],
        "N": N_res,
        "mu": mu,
        "sigma": sigma,
        "initial_condition": x0,
        "noise_covariance": res_w.get("covariance", None),
        "order": order,
        "seed": seed,
    }