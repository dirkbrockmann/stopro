import numpy as np

from ._utils import _time_grid, _mixing

def wiener(T,
           dt=None,
           *,
           steps=None,
           gap=1,
           N=1,
           samples=1,
           covariance=None,
           mixing_matrix=None):

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

    Returns
    -------
    dict
        Keys:
        - 'X': array, shape (samples, N, savedsteps+1)
        - 't': array, shape (savedsteps+1,)
        - 'dt', 'steps', 'savedsteps', 'N', 'gap', 'covariance'
    """

    dt, steps, t_full = _time_grid(T, dt=dt, steps=steps)

    gap = int(gap)
    if gap <= 0:
        raise ValueError("gap must be a positive integer.")

    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be a positive integer.")

    S, covariance, N, M = _mixing(N=N, covariance=covariance, mixing_matrix=mixing_matrix)
    
   # Single subsampling index used everywhere
    idx = np.arange(0, steps + 1, gap)
    t = t_full[idx]
    savedsteps = len(t) - 1

    X = np.zeros((samples, N, savedsteps+1), dtype=float)
    
    for i in range(samples):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dw = S @ np.random.randn(M, steps + 1)
        dw[:, 0] = 0.0
        W_full = np.sqrt(dt) * np.cumsum(dw, axis=1)   # (N, steps+1)
        X[i] = W_full[:, idx]
    


    return {
        'X': X,
        't': t,
        'dt': dt,
        'steps': steps,
        'savedsteps': savedsteps,
        'N': N,
        'gap': gap,
        'covariance': covariance,
    }