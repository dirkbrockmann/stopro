import numpy as np

from ._utils import _as_vector
from .ornstein_uhlenbeck import ornstein_uhlenbeck


def exponential_ornstein_uhlenbeck(
    T,
    dt=None,
    *,
    steps=None,
    gap=1,
    N=1,
    samples=1,
    mean=1.0,
    coeff_var=1.0,
    timescale=1.0,
    initial_condition=None,
    covariance=None,
    mixing_matrix=None,
):
    """
    Simulate an exponential Ornstein–Uhlenbeck (lognormal OU) process on [0, T].

    Constructs X from an OU process Z via

        X_i(t) = A_i * exp(B_i * Z_i(t)),

    where Z has stationary mean 0 and stationary stdev 1 (OU timescale set by
    `timescale`). A and B are chosen so that X has the specified `mean` and
    coefficient of variation `coeff_var` (component-wise).

    Provide exactly one of `dt` or `steps`. Noise correlation can be specified via
    `covariance` or `mixing_matrix` (mutually exclusive). Use `gap>1` to subsample.

    Returns
    -------
    dict
        The return dict from `ornstein_uhlenbeck`, with 'X' replaced by the
        exponential transform and added keys: 'mean', 'coeff_var', 'A', 'B'.
    """
    N = int(N)
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    mean = _as_vector(mean, N, "mean")
    coeff_var = _as_vector(coeff_var, N, "coeff_var")

    if np.any(mean <= 0):
        raise ValueError("mean must be positive.")
    if np.any(coeff_var < 0):
        raise ValueError("coeff_var must be >= 0.")
    if np.any(np.asarray(timescale, dtype=float) <= 0):
        raise ValueError("timescale must be > 0.")

    # For Z ~ N(0,1):  CV^2 = exp(B^2) - 1  =>  B = sqrt(log(1 + CV^2))
    #                 E[X]  = A exp(B^2/2)  =>  A = mean / exp(B^2/2) = mean / sqrt(1 + CV^2)
    A = mean / np.sqrt(1.0 + coeff_var**2)
    B = np.sqrt(np.log(1.0 + coeff_var**2))

    # Underlying OU: keep stationary stdev=1 so A/B calibration is correct.
    res = ornstein_uhlenbeck(
        T,
        dt,
        steps=steps,
        gap=gap,
        N=N,
        samples=samples,
        initial_condition=initial_condition,
        stdev=1.0,
        timescale=timescale,
        covariance=covariance,
        mixing_matrix=mixing_matrix,
    )

    Z = res["X"]  # (samples, N, K)
    res["X"] = A[None, :, None] * np.exp(B[None, :, None] * Z)

    res["mean"] = mean
    res["coeff_var"] = coeff_var
    res["A"] = A
    res["B"] = B
    return res