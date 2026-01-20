import numpy as np

from ._utils import _as_vector
from .integrated_ornstein_uhlenbeck import integrated_ornstein_uhlenbeck


def colored_geometric_brownian_motion(
    T,
    dt=None,
    *,
    steps=None,
    gap=1,
    N=1,
    samples=1,
    mu=1.0,
    sigma=1.0,
    tau=1.0,
    initial_condition=None,
    covariance=None,
    mixing_matrix=None,
):
    """
    Simulate multivariate colored geometric Brownian motion (cGBM) on [0, T].

        dX_i(t) = mu_i X_i(t) dt + sigma_i X_i(t) Z_i(t) dt
        tau * dZ_i(t) = -Z_i(t) dt + dW_i(t)

    Hence,
        X_i(t) = X_i(0) * exp(mu_i t + sigma_i * ∫_0^t Z_i(s) ds).

    Provide exactly one of `dt` or `steps`. Noise correlation can be specified via
    `covariance` or `mixing_matrix` (mutually exclusive). Use `gap>1` to subsample.
    """
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be > 0.")

    N = int(N)
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    # Choose OU stdev so that: tau*dZ = -Z dt + dW  <=>  dZ = -(1/tau)Z dt + (1/tau)dW
    # which implies stationary stdev(Z) = 1/sqrt(2*tau).
    stdev_z = 1.0 / np.sqrt(2.0 * tau)

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
    )

    I = res_I["X"]  # (samples, N, K) = ∫_0^t Z(s) ds (approx.)
    t = res_I["t"]  # (K,)
    N_res = res_I["N"]

    mu = _as_vector(mu, N_res, "mu")
    sigma = _as_vector(sigma, N_res, "sigma")

    if initial_condition is None:
        x0 = np.ones(N_res, dtype=float)
    else:
        x0 = _as_vector(initial_condition, N_res, "initial_condition")

    # X_i(t) = x0_i * exp(mu_i t + sigma_i I_i(t))
    tt = t[None, None, :]  # (1, 1, K)
    X = x0[None, :, None] * np.exp(mu[None, :, None] * tt + sigma[None, :, None] * I)

    res_I["X"] = X
    res_I["mu"] = mu
    res_I["sigma"] = sigma
    res_I["tau"] = tau
    res_I["initial_condition"] = x0
    return res_I