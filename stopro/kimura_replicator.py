from ._utils import _simplex_initial_condition, _time_grid, _mixing, _as_vector
import numpy as np


def kimura_replicator(
                T,
                dt=None,
                *,
                steps=None,
                N=2,
                mu=1.0,
                sigma=1.0,
                initial_condition=None,
                gap=1,
                samples=1,
                covariance=None,
                mixing_matrix=None):
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

    Returns
    -------
    dict
        Keys: 'X' (samples, N, savedsteps+1), 't' (savedsteps+1,), 'dt', 'steps',
        'savedsteps', 'gap', 'N', 'noise_covariance', 'mu', 'sigma', 'initial_condition'.
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
    
    X = np.zeros((samples, N, savedsteps+1), dtype=float)

    for i in range(samples):
        x = np.zeros((N,steps+1))
        dw = S @ np.random.randn(M, steps+1)

        x[:,0] = x0

        for j in range(steps):
            r = mu * dt + sigma * dw[:,j] * sqdt
            phi = np.sum(r * x[:,j])
            dx = (r-phi)*x[:,j]
            x[:,j+1] = x[:,j] + dx
            x[:,j+1] = np.where(x[:,j+1]<0, 0, x[:,j+1])

        X[i] = x[:, idx]

    return {
            'initial_condition': x0,
            'mu': mu,
            'sigma': sigma,
            'noise_covariance': covariance,
            'steps': steps,
            'dt': dt,
            't': t,
            'X': X,
            'gap': gap,
            'N' : N,
            'savedsteps': savedsteps,
            }
