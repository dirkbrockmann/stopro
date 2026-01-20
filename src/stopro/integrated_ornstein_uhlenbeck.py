import numpy as np

from .ornstein_uhlenbeck import ornstein_uhlenbeck

def integrated_ornstein_uhlenbeck(
    T,
    dt=None,
    *,
    steps=None,
    stdev=1,
    timescale=1,
    N=1,
    gap=1,
    samples=1,
    initial_condition=None,
    covariance=None,
    mixing_matrix=None,
    theta=None,
    sigma=None,
):
    """
    Simulate the time-integral of an Ornstein–Uhlenbeck process on [0, T].

        Z(t): Ornstein–Uhlenbeck process
        X(t) = ∫_0^t Z(s) ds  (approximated by a Riemann sum)

    Provide exactly one of `dt` or `steps`. Use `gap>1` to subsample returned points.

    Notes
    -----
    The OU process is simulated on the *full* time grid (gap=1) for integration accuracy,
    then the integrated trajectory is subsampled by `gap` for the returned result.

    Returns
    -------
    dict
        Same structure as `ornstein_uhlenbeck`, but with 'X' replaced by the integrated
        process and with 'gap'/'savedsteps' reflecting the returned (subsampled) grid.
    """
    gap = int(gap)
    if gap <= 0:
        raise ValueError("gap must be a positive integer.")

    # Simulate OU on the full grid for accurate integration
    res = ornstein_uhlenbeck(
        T,
        dt,
        steps=steps,
        stdev=stdev,
        timescale=timescale,
        N=N,
        gap=1,
        samples=samples,
        initial_condition=initial_condition,
        covariance=covariance,
        mixing_matrix=mixing_matrix,
        theta=theta,
        sigma=sigma,
    )

    t_full = res["t"]          # (steps+1,)
    z_full = res["X"]          # (samples, N, steps+1)
    dt_used = res["dt"]
    steps_used = res["steps"]

    # X(t_k) ≈ dt * sum_{j=0..k} Z(t_j)
    x_full = dt_used * np.cumsum(z_full, axis=2)

    # Subsample for return
    idx = np.arange(0, steps_used + 1, gap)
    res["t"] = t_full[idx]
    res["X"] = x_full[:, :, idx]
    res["gap"] = gap
    res["savedsteps"] = len(idx) - 1

    return res