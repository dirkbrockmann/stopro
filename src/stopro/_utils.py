import numpy as np

# Build a uniform time grid on [0, T] from either dt or steps (exactly one must be given).
# Returns (dt, steps, t) with t including both endpoints 0 and T.
def _time_grid(T, dt=None, steps=None):
    """
    Build a uniform grid on [0, T].

    Provide exactly one of dt or steps.
    - If steps is provided: dt = T/steps.
    - If dt is provided: choose integer steps close to T/dt, then use dt = T/steps
      so the grid ends exactly at T.
    """
    if (dt is None) == (steps is None):
        raise ValueError("Provide exactly one of dt or steps (not both, not neither).")

    T = float(T)
    if T <= 0:
        raise ValueError("T must be > 0.")

    if steps is not None:
        steps = int(steps)
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")
        dt = T / steps
        t = np.linspace(0.0, T, steps + 1)
        return dt, steps, t

    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be > 0.")

    steps = int(np.round(T / dt))
    steps = max(1, steps)

    dt = T / steps
    t = np.linspace(0.0, T, steps + 1)
    return dt, steps, t


# Robustly factor a symmetric PSD matrix C into L such that C ≈ L @ L.T.
# Uses Cholesky when possible, otherwise falls back to eigen-decomposition (clipping small negatives).
def _psd_factor(C, *, jitter=0.0, tol=1e-12):

    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + C.T)

    if jitter and jitter > 0:
        Cj = C + jitter * np.eye(C.shape[0])
    else:
        Cj = C

    try:
        return np.linalg.cholesky(Cj)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(C)
        w = np.where(w > tol, w, 0.0)
        return V @ np.diag(np.sqrt(w))


# Normalize "noise specification" into a mixing matrix S with covariance C = S @ S.T.
# Accepts either covariance (must be symmetric PSD) or a mixing_matrix; otherwise defaults to identity.
def _mixing(*, N=None, covariance=None, mixing_matrix=None, jitter=1e-12, psd_tol=1e-12, sym_tol=1e-12):
    if covariance is not None and mixing_matrix is not None:
        raise ValueError("Provide only one of covariance or mixing_matrix (not both).")

    if mixing_matrix is not None:
        S = np.asarray(mixing_matrix, dtype=float)
        if S.ndim != 2:
            raise ValueError("mixing_matrix must be a 2D array.")
        N_res, M = S.shape
        if N is not None and int(N) != N_res:
            raise ValueError(f"N={N} is inconsistent with mixing_matrix.shape[0]={N_res}.")
        C = S @ S.T
        C = 0.5 * (C + C.T)
        return S, C, N_res, M

    if covariance is not None:
        C = np.asarray(covariance, dtype=float)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("covariance must be a square (N,N) array.")

        # Require symmetry (within tolerance) for user-provided covariance
        max_asym = float(np.max(np.abs(C - C.T)))
        if max_asym > sym_tol:
            raise ValueError(
                f"covariance must be symmetric; max |C - C.T| = {max_asym:g} exceeds sym_tol={sym_tol:g}."
            )

        C = 0.5 * (C + C.T)

        N_res = C.shape[0]
        if N is not None and int(N) != N_res:
            raise ValueError(f"N={N} is inconsistent with covariance.shape[0]={N_res}.")

        lam_min = float(np.min(np.linalg.eigvalsh(C)))
        if lam_min < -psd_tol:
            raise ValueError(
                f"covariance must be positive semidefinite; min eigenvalue = {lam_min:g} < -psd_tol={psd_tol:g}."
            )

        S = _psd_factor(C, jitter=jitter, tol=psd_tol)
        M = N_res
        return S, C, N_res, M
    # Neither covariance nor mixing_matrix: default to independent Wiener components
    if N is None:
        raise ValueError("N must be provided when neither covariance nor mixing_matrix is given.")
    N_res = int(N)
    if N_res <= 0:
        raise ValueError("N must be a positive integer.")
    S = np.eye(N_res, dtype=float)
    C = np.eye(N_res, dtype=float)
    M = N_res
    return S, C, N_res, M

# Coerce a parameter into a length-N float vector (broadcast scalar; validate 1D shape).
def _as_vector(x, N, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(int(N), float(arr))
    if arr.shape == (int(N),):
        return arr
    raise ValueError(f"{name} must be a scalar or an array of shape ({N},), got shape {arr.shape}.")


# Parse initial_condition into either ("stationary") or an explicit x0 vector of length N.
# Used by OU/Wiener-like processes to support a convenient stationary start.
def _parse_initial_condition(initial_condition, *, N):
    if isinstance(initial_condition, str):
        if initial_condition.lower() == "stationary":
            return True, None
        raise ValueError(f"unknown initial_condition '{initial_condition}'")

    if initial_condition is None:
        return False, np.zeros(int(N), dtype=float)

    # scalar or vector
    x0 = _as_vector(initial_condition, N, "initial_condition")
    return False, x0


# Create an initial condition on the probability simplex (nonnegative entries summing to 1).
# Accepts None (uniform), scalar (broadcast then normalize), or a length-N vector (validate+normalize).
def _simplex_initial_condition(initial_condition, *, N, name="initial_condition"):

    N = int(N)
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    if initial_condition is None:
        return np.full(N, 1.0 / N, dtype=float)

    x0 = np.asarray(initial_condition, dtype=float)

    if x0.ndim == 0:
        c = float(x0)
        if c <= 0:
            raise ValueError(f"{name} scalar must be > 0 to normalize, got {c}.")
        x0 = np.full(N, c, dtype=float)

    if x0.shape != (N,):
        raise ValueError(f"{name} must be None, a scalar, or an array of shape ({N},), got shape {x0.shape}.")

    if np.any(x0 < 0):
        raise ValueError(f"{name} must be nonnegative.")

    s = float(np.sum(x0))
    if s <= 0:
        raise ValueError(f"{name} must have positive sum to normalize.")

    return x0 / s