from typing import Literal,Any

import numpy as np

from ._utils import _time_grid, _mixing


def wiener(
    T: float,
    dt: float | None = None,
    *,
    steps: int | None = None,
    gap: int = 1,
    N: int = 1,
    samples: int = 1,
    covariance: np.ndarray | None = None,
    mixing_matrix: np.ndarray | None = None,
    order: Literal["STD", "SDT"] = "STD",  # "STD" (samples, time, dim) or "SDT" (samples, dim, time)
) -> dict[str, Any]:
    """
    Simulate an N-dimensional Wiener process on [0, T].

    Notes
    -----
    - Uses a uniform grid. If `steps` is provided, it overrides `dt` via `dt = T/steps`.
    - Set `gap>1` to return every `gap`-th time point.

    Parameters
    ----------
    T : float
        End time of the simulation interval [0, T].
    dt : float
        Time step size (overridden if `steps` is provided).
    steps : int, optional
        Number of time steps (overrides `dt`).
    gap : int, default=1
        Subsampling factor for returned points.
    N : int, default=1
        Process dimension (overridden if `covariance` or `mixing_matrix` is provided).
    samples : int, default=1
        Number of independent realizations.
    covariance : (N,N) array, optional
        Covariance of increments. Must be positive semidefinite.
    mixing_matrix : (N,M) array, optional
        Mixing matrix S such that dW = S dV and covariance = S S^T.
    order : {"STD","SDT"}, default="STD"
        Output array layout for X:
        - "STD": (samples, time, dim)  [default, plot-friendly]
        - "SDT": (samples, dim, time)  [legacy]

    Returns
    -------
    dict
        Keys:
        - 'X': array, shape depends on `order`
        - 't': array, shape (savedsteps+1,)
        - 'dt', 'steps', 'savedsteps', 'N', 'gap', 'covariance', 'order'
    """

    dt, steps, t_full = _time_grid(T, dt=dt, steps=steps)

    gap = int(gap)
    if gap <= 0:
        raise ValueError("gap must be a positive integer.")

    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be a positive integer.")

    # Resolve mixing / covariance
    S, covariance, N, M = _mixing(
        N=N,
        covariance=covariance,
        mixing_matrix=mixing_matrix,
    )

    # Subsampling index used everywhere
    idx = np.arange(0, steps + 1, gap)
    t = t_full[idx]
    savedsteps = len(t) - 1

    # Internal layout: (samples, dim, time)
    X = np.zeros((samples, N, savedsteps + 1), dtype=float)

    for i in range(samples):
        dw = S @ np.random.randn(M, steps + 1)
        dw[:, 0] = 0.0
        W_full = np.sqrt(dt) * np.cumsum(dw, axis=1)  # (N, steps+1)
        X[i] = W_full[:, idx]                         # (N, savedsteps+1)

    # Convert once at the boundary
    if order == "STD":
        X_out = np.moveaxis(X, 1, 2)  # (samples, dim, time) -> (samples, time, dim)
    elif order == "SDT":
        X_out = X
    else:
        raise ValueError("order must be 'STD' or 'SDT'")

    return {
        "X": X_out,
        "t": t,
        "dt": dt,
        "steps": steps,
        "savedsteps": savedsteps,
        "N": N,
        "gap": gap,
        "covariance": covariance,
        "order": order,
    }