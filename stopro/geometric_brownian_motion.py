import numpy as np

from ._utils import _as_vector
from .wiener import wiener


def geometric_brownian_motion(
    T,
    dt=None,
    *,
    steps=None,
    gap=1,
    N=1,
    samples=1,
    mu=1.0,
    sigma=1.0,
    initial_condition=None,
    covariance=None,
    mixing_matrix=None,
):
    """
    Simulate (possibly multivariate) geometric Brownian motion (GBM) on [0, T].

        dX_i(t) = mu_i X_i(t) dt + sigma_i X_i(t) dW_i(t),

    where W is an N-dimensional Wiener process (optionally correlated via `covariance`
    or `mixing_matrix`, mutually exclusive).

    Provide exactly one of `dt` or `steps`. Use `gap>1` to subsample returned points.

    Returns
    -------
    dict
        Keys: 'X' (samples, N, savedsteps+1), 't' (savedsteps+1,), 'dt', 'steps',
        'savedsteps', 'gap', 'N', 'mu', 'sigma', 'initial_condition', 'noise_covariance'.
    """
    # Generate Wiener paths already on the saved grid (no double gap-handling here)
    res_w = wiener(
        T,
        dt=dt,
        steps=steps,
        gap=gap,
        N=N,
        samples=samples,
        covariance=covariance,
        mixing_matrix=mixing_matrix,
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

    X = x0[None, :, None] * np.exp(drift * tt + diff * W)

    return {
        "X": X,
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
    }