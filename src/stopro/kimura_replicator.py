from random import seed
from typing import Literal,Any

import numpy as np

from ._utils import _time_grid, _mixing, _as_vector, _simplex_initial_condition  


def kimura_replicator(
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
    Simulate the (stochastic) Kimura replicator dynamics on the simplex.

    The process evolves frequencies X(t) with components X_i >= 0 and sum_i X_i = 1.
    Noise can be correlated via `covariance` (PSD) or via a `mixing_matrix` S such that
    covariance = S S^T (mutually exclusive).

    Provide exactly one of `dt` or `steps`. Use `gap>1` to subsample returned time points.

    Parameters
    ----------
    T : float
        End time of the simulation interval [0, T].
    dt : float, optional
        Time step size (use `steps` instead to specify a fixed number of steps).
    steps : int, optional
        Number of time steps (alternative to `dt`).
    N : int, default=2
        Number of species (must be >= 2; may be inferred from covariance/mixing_matrix).
    mu, sigma : float or array-like, default=1
        Drift and noise strength parameters; scalars are broadcast to length N.
    initial_condition : None or array-like shape (N,), optional
        If None, uses uniform (1/N,...,1/N). Otherwise normalizes the provided vector onto
        the simplex (nonnegative with positive sum).
    gap : int, default=1
        Subsampling factor for returned points.
    samples : int, default=1
        Number of independent realizations.
    covariance : array-like (N,N), optional
        Covariance of Wiener increments (positive semidefinite).
    mixing_matrix : array-like (N,M), optional
        Mixing matrix S that induces covariance = S S^T.
    order : {"STD","SDT"}, default="STD"
        Output array layout for X:
        - "STD": (samples, time, dim)  [default, plot-friendly]
        - "SDT": (samples, dim, time)  [legacy]
    seed : int, optional
        Seed for reproducible randomness (seeds NumPy global RNG).

    Returns
    -------
    dict
        Keys: 'X' (shape depends on `order`), 't' (savedsteps+1,), 'dt', 'steps',
        'savedsteps', 'gap', 'N', 'noise_covariance', 'mu', 'sigma', 'initial_condition', 'order'.
    """

    if seed is not None:
        np.random.seed(seed)

    dt, steps, t_full = _time_grid(T, dt=dt, steps=steps)
    sqdt = np.sqrt(dt)

    gap = int(gap)
    if gap <= 0:
        raise ValueError("gap must be a positive integer.")

    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be a positive integer.")

    S, covariance, N, M = _mixing(N=N, covariance=covariance, mixing_matrix=mixing_matrix)

    if int(N) < 2:
        raise ValueError(f"kimura_replicator requires N>=2, got N={N}.")

    x0 = _simplex_initial_condition(initial_condition, N=N)

    # force mu/sigma to vectors of length N
    mu = _as_vector(mu, N, "mu")
    sigma = _as_vector(sigma, N, "sigma")

    # subsampling
    idx = np.arange(0, steps + 1, gap)
    t = t_full[idx]
    savedsteps = len(t) - 1

    # Internal layout: (samples, dim, time)
    X = np.zeros((samples, N, savedsteps + 1), dtype=float)

    for i in range(samples):
        x = np.zeros((N, steps + 1), dtype=float)
        dw = S @ np.random.randn(M, steps + 1)

        x[:, 0] = x0

        for j in range(steps):
            r = mu * dt + sigma * dw[:, j] * sqdt
            phi = np.sum(r * x[:, j])
            dx = (r - phi) * x[:, j]
            x[:, j + 1] = x[:, j] + dx
            x[:, j + 1] = np.where(x[:, j + 1] < 0, 0, x[:, j + 1])

        X[i] = x[:, idx]

    # Convert once at the boundary
    if order == "STD":
        X_out = np.moveaxis(X, 1, 2)  # (samples, dim, time) -> (samples, time, dim)
    elif order == "SDT":
        X_out = X
    else:
        raise ValueError("order must be 'STD' or 'SDT'")

    return {
        "initial_condition": x0,
        "mu": mu,
        "sigma": sigma,
        "noise_covariance": covariance,
        "steps": steps,
        "dt": dt,
        "t": t,
        "X": X_out,
        "gap": gap,
        "N": N,
        "savedsteps": savedsteps,
        "order": order,
        "seed": seed,
    }