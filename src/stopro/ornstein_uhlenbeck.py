from ._utils import _time_grid, _mixing, _as_vector, _parse_initial_condition, _psd_factor
import numpy as np

def _taustdevthetasigma(*, N, stdev=1.0, timescale=1.0, theta=None, sigma=None):
    N = int(N)
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    using_theta_sigma = (theta is not None) or (sigma is not None)

    if using_theta_sigma:
        
        theta = 1.0 if theta is None else theta
        sigma = 1.0 if sigma is None else sigma

        theta = _as_vector(theta, N, "theta")
        sigma = _as_vector(sigma, N, "sigma")

        if np.any(theta < 0):
            raise ValueError("theta must be >= 0 (negative theta makes the OU unstable).")

        with np.errstate(divide="ignore", invalid="ignore"):
            timescale = np.where(theta > 0, 1.0 / theta, np.inf)
            stdev = np.where(theta > 0, sigma / np.sqrt(2.0 * theta), np.inf)

        return theta, sigma, stdev, timescale

    # using stdev/timescale
    timescale = _as_vector(timescale, N, "timescale")
    stdev = _as_vector(stdev, N, "stdev")

    if np.any(timescale <= 0):
        raise ValueError("timescale must be > 0 (use np.inf for the theta=0 limit).")

    with np.errstate(divide="ignore", invalid="ignore"):
        theta = 1.0 / timescale
        sigma = stdev * np.sqrt(2.0 / timescale)

    return theta, sigma, stdev, timescale



def ornstein_uhlenbeck(T,
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
    Simulate an (optionally multivariate) Ornstein–Uhlenbeck process on [0, T].

    The OU process solves (component-wise)

        dX = -theta * X dt + sigma dW,

    where W is a (possibly correlated) Wiener process. Provide either (`stdev`, `timescale`)
    or (`theta`, `sigma`). Noise correlation can be specified via `covariance` *or*
    `mixing_matrix` (mutually exclusive). Use `gap>1` to subsample the returned trajectory.

    Parameters
    ----------
    T : float
        End time of the simulation interval [0, T].
    dt : float
        Time step size (overridden if `steps` is provided).
    steps : int, optional
        Number of time steps (overrides `dt` via dt = T/steps).
    N : int, default=1
        Process dimension (overridden if `covariance` or `mixing_matrix` is provided).
    samples : int, default=1
        Number of independent realizations.
    gap : int, default=1
        Subsampling factor for returned points.

    stdev, timescale : float or array-like, default=1
        Stationary standard deviation and autocorrelation time. Used if `theta`/`sigma`
        are not provided.
    theta, sigma : float or array-like, optional
        OU parameters. If given, override `stdev`/`timescale`.

    initial_condition : None | 'stationary' | array-like, optional
        If None, start at 0. If 'stationary', draw X(0) from the stationary distribution.
        Otherwise use the provided vector as X(0).

    covariance : array-like (N,N), optional
        Covariance matrix of Wiener increments (must be positive semidefinite).
    mixing_matrix : array-like (N,M), optional
        Mixing matrix S such that dW = S dV, with independent dV. Implies covariance = S S^T.

    Returns
    -------
    dict
        Keys include: 'X', 't', 'dt', 'steps', 'savedsteps', 'gap', 'N',
        'noise_covariance', 'theta', 'sigma', 'stdev', 'timescale', 'initial_condition'.
    """


    dt, steps, t_full = _time_grid(T, dt=dt, steps=steps)

    sqdt = np.sqrt(dt)

    gap = int(gap)
    if gap <= 0:
        raise ValueError("gap must be a positive integer.")

    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be a positive integer.")

    S, covariance, N, M = _mixing(N=N, covariance=covariance, mixing_matrix=mixing_matrix)

    theta, sigma, stdev, timescale = _taustdevthetasigma(N=N, stdev=stdev, timescale=timescale, theta=theta, sigma=sigma)
    
    idx = np.arange(0, steps + 1, gap)
    t = t_full[idx]
    savedsteps = len(t) - 1
    X = np.zeros((samples, N, savedsteps+1), dtype=float)


    stationary, x0 = _parse_initial_condition(initial_condition, N=N)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    C = np.asarray(covariance, dtype=float)

    if stationary and np.any(theta <= 0):
        raise ValueError("stationary initial_condition requires theta > 0 in every dimension.")

    if stationary:
        denom = theta[:, None] + theta[None, :]
        P = (sigma[:, None] * sigma[None, :]) * C / denom
        P = 0.5 * (P + P.T)
        Pfac = _psd_factor(P)

    # --- exact discrete-time OU update precomputations ---
    a = np.exp(-theta * dt)  # (N,)

    denom = theta[:, None] + theta[None, :]  # (N, N)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        F = (1.0 - np.exp(-denom * dt)) / denom
    F = np.where(denom == 0.0, dt, F)  # continuous limit when denom -> 0

    Q = (sigma[:, None] * sigma[None, :]) * C * F
    Q = 0.5 * (Q + Q.T)
    Qfac = _psd_factor(Q)

    for i in range(samples):
        x = np.zeros((N, steps + 1), dtype=float)

        if stationary:
            x[:, 0] = Pfac @ np.random.randn(N)
        else:
            x[:, 0] = x0

        for j in range(steps):
            x[:, j + 1] = a * x[:, j] + (Qfac @ np.random.randn(N))

        X[i] = x[:, idx]
        
    return {
            'initial_condition': initial_condition,
            'stdev': stdev,
            'timescale': timescale,
            'sigma': sigma,
            'theta': theta,
            'noise_covariance': covariance,
            'steps': steps,
            'dt': dt,
            't': t,
            'X': X,
            'gap': gap,
            'savedsteps': savedsteps,
            'N': N
            }
