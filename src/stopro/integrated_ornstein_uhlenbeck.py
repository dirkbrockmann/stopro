from typing import Any, Literal

import numpy as np

from .ornstein_uhlenbeck import ornstein_uhlenbeck


def integrated_ornstein_uhlenbeck(
    T: float,
    dt: float | None = None,
    *,
    steps: int | None = None,
    stdev: float | np.ndarray = 1,
    timescale: float | np.ndarray = 1,
    N: int = 1,
    gap: int = 1,
    samples: int = 1,
    initial_condition: None | Literal["stationary"] | np.ndarray = None,
    covariance: np.ndarray | None = None,
    mixing_matrix: np.ndarray | None = None,
    theta: float | np.ndarray | None = None,
    sigma: float | np.ndarray | None = None,
    order: Literal["STD", "SDT"] = "STD",  # "STD" (samples, time, dim) or "SDT" (samples, dim, time)
) -> dict[str, Any]:
    """
    Simulate the time-integral of an Ornstein–Uhlenbeck process on [0, T].

        Z(t): Ornstein–Uhlenbeck process
        X(t) = ∫_0^t Z(s) ds  (approximated by a Riemann sum)

    Notes
    -----
    The OU process is simulated on the *full* time grid (gap=1) for integration accuracy,
    then the integrated trajectory is subsampled by `gap` for the returned result.

    order : {"STD","SDT"}, default="STD"
        Output array layout for X:
        - "STD": (samples, time, dim)  [default, plot-friendly]
        - "SDT": (samples, dim, time)  [legacy]
    """

    gap = int(gap)
    if gap <= 0:
        raise ValueError("gap must be a positive integer.")

    # --- simulate OU on the full grid, always in STD ---
    res = ornstein_uhlenbeck(
        T,
        dt,
        steps=steps,
        stdev=stdev,
        timescale=timescale,
        N=N,
        gap=1,  # full grid for integration
        samples=samples,
        initial_condition=initial_condition,
        covariance=covariance,
        mixing_matrix=mixing_matrix,
        theta=theta,
        sigma=sigma,
        order="STD",  # force known layout
    )

    t_full = res["t"]        # (steps+1,)
    z_full = res["X"]        # (samples, time, dim)
    dt_used = res["dt"]
    steps_used = res["steps"]

    # --- integrate over time axis ---
    x_full = dt_used * np.cumsum(z_full, axis=1)  # axis=1 is time

    # --- subsample for return ---
    idx = np.arange(0, steps_used + 1, gap)
    t_sub = t_full[idx]
    X_std = x_full[:, idx, :]  # (samples, savedsteps+1, dim)

    # --- convert layout at the boundary if requested ---
    if order == "STD":
        X_out = X_std
    elif order == "SDT":
        X_out = np.moveaxis(X_std, 1, 2)  # (samples, time, dim) -> (samples, dim, time)
    else:
        raise ValueError("order must be 'STD' or 'SDT'")

    res["t"] = t_sub
    res["X"] = X_out
    res["gap"] = gap
    res["savedsteps"] = len(idx) - 1
    res["order"] = order

    return res