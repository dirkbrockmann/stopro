# -*- coding: utf-8 -*-
"""
Contains functions simulating the stochastic processes.
"""

import numpy as np

# Wiener Process

def wiener(T,dt,dimension=1,samples=1,covariance=None,mixing_matrix=None):
    """
    Generates realizations of a multivariate Wiener process.

    Returns realizations (samples) on the time interval [0,T] at increments of size dt.
    You can specify the number of realizations, as well as the covariance or
    mixing matrix in case the process is multidimensional.

    Parameters
    ----------
    T : float
        The upper bound of the stochastic integral.
    dt : float
        size of the time step
    dimension : int, default = 1
        The dimension of the stochastic process.
        If no mixing matrix is provided, ``dimension``
        will be equal to the ``target_dimension`` of the resulting Wiener process.
    samples : int, default = 1
        The number of samples generated
    covariance : numpy.ndarray of shape (``dimension``, ``dimension``), default = None
        In case of a multivariate process the covariance matrix,
        which must be positive semidefinite. If specified, overrides the ``dimension`` parameter.
        The resulting realizations will have ``target_dimension = dimension``.
    mixing_matrix : numpy.ndarray of shape (``target_dimension``, ``dimension``), default = None
        This matrix, let's call it S with elements :math:`S_{ij}` is used to generate an
        N-dimensional covariant Wiener processes W (with components :math`W_i, i=1,...,N`) by superposition
        of independent components :math:`V_j` of an M-dimensional Wiener process V

        .. math::

            W_i = \sum_j S_{ij} \cdot V_j.

        The covariance of W is given by :math:`S \cdot S^T`.
        Specifying the mixing matrix overrides the covariance parameter and the dimension parameter.


    Returns
    -------
    result : dict

    Result dictionary containing the following entries

        .. code:: python

            {
                'X': 'numpy.ndarray of shape (samples, target_dimension, steps+1) such that  X[i,j,k] is time point k of component j of realization i',
                't': 'array of time points',
                'dt': 'time increment',
                'steps': 'numper of time steps',
                'covariance': 'covariance matrix, numpy.ndarray of shape (target_dimension, target_dimension)'
            }
    """

    steps = int (T / dt)

    target_dimension = dimension

    if covariance is not None:
        covariance = np.array(covariance)
        (n,m) = np.shape(covariance)
        assert n==m, "covariance matrix must be square"
        assert np.all(np.linalg.eigvals(covariance) >= 0), "covariance matrix is not positive definite"
        dimension = n
        S = np.linalg.cholesky(covariance)
    else:
        covariance = np.identity(dimension)
        S = covariance

    if mixing_matrix is not None:
        S = np.array(mixing_matrix)
        (n,m) = S.shape
        covariance = S @ S.T
        dimension = m

    target_dimension, _ = S.shape

    t = np.linspace(0,T,steps+1)
    X = np.zeros((samples,target_dimension,steps+1))

    for i in range(samples):
        dw = S @ np.random.randn(dimension,steps+1)
        dw[:,0] = 0
        W = np.sqrt(dt)*np.cumsum(dw,axis=1)
        X[i] = W

    return {
        'X': X,
        't': t,
        'dt': dt,
        'steps': steps,
        'covariance': covariance,
    }

def ornsteinuhlenbeck(T,
                      dt,
                      variability=1,
                      timescale=1,
                      dimension=1,
                      samples=1,
                      stationary=False,
                      covariance_matrix=None,
                      mixing_matrix=None,
                      steps=None,
                      theta=None,
                      sigma=None,
                      ):

    r"""
    Generates realizations of the multivariate Ornstein-Uhlenbeck Process X(t)

    The OUP is the solution to the SDE

    .. math::

        dX = -\theta X dt + \sigma dW,

    where W is the Wiener Process.

    Returns realizations on the time interval [0,T] at increments dt.

    Parameters
    ----------
    T : float
        The upper bound of the stochastic integral.
    dt : float
        size of the time step. Will be overridden if ``steps`` is provided instead.
    variability : float, default = 1
        Standard deviation, defined as ``variability = sigma sqrt(2/timescale)``.
        Is overridden if any of the parameters ``sigma`` or ``theta`` is provided.
    timescale : float, default = 1
        Autocorrelation time of the process, defined as ``timescale = 1 / theta``.
        Is overridden if parameter ``theta`` is provided.
    dimension : int, default = 1
        The dimension of the stochastic process.
        If no mixing matrix is provided, ``dimension``
        will be equal to the ``target_dimension`` of the resulting Wiener process.
    samples : int, default = 1
        The number of samples generated
    initial_conditions : str or numpy.ndarray, default = None
        If ``None``, process will be initiated at X = 0.
        If ``'stationary'``, initial conditions will be drawn from stationary distribution.
        Else, process will be initiated as ``X[:,0] = initial_conditions``.
        If ``True`` the processes are initialized so they are stationary,
    theta : float, default = None
        Defines the timescale of the process as ``timescale = 1/theta``.
        Overrides ``timescale`` and ``variability`` if defined.
    sigma: float, default = None
        Defines the variability of the process as ``variability = sigma sqrt(2/timescale)``.
        Overrides ``variability`` if defined.
    covariance : numpy.ndarray of shape (``dimension``, ``dimension``), default = None
        In case of a multivariate process the covariance matrix,
        which must be positive semidefinite. If specified, overrides the ``dimension`` parameter.
        The resulting realizations will have ``target_dimension = dimension``.
    mixing_matrix : numpy.ndarray of shape (``target_dimension``, ``dimension``), default = None
        This matrix, let's call it S with elements :math:`S_{ij}` is used to generate an
        N-dimensional covariant Wiener processes W (with components :math`W_i, i=1,...,N`) by superposition
        of independent components :math:`V_j` of an M-dimensional Wiener process V

        .. math::

            W_i = \sum_j S_{ij} \cdot V_j.

        The covariance of W is given by :math:`S \cdot S^T`.
        Specifying the mixing matrix overrides the covariance parameter and the dimension parameter.
        if ``False`` (default) all realizations start at the origin.
    steps : int, default = None
        if provided, defines a number of time steps. Overrides `dt`.

    Returns
    -------
    result : dict

        Result in the following structure:

        .. code:: python

            {
                'X': 'numpy.ndarray of shape (samples, target_dimension, steps+1) such that  X[i,j,k] is time point k of component j of realization i',
                't': 'array of time points',
                'dt': 'time increment',
                'steps': 'numper of time steps',
                'noise_covariance': 'covariance matrix, numpy.ndarray of shape (target_dimension, target_dimension)',
                'variability': 'as defined above',
                'timescale': 'as defined above',
            }

    Notes
    -----

    You can either provide a covariance matrix OR a mixing_matrix, but not both.
    """

    if isinstance(initial_condition, str) and initial_condition == 'stationary':
        pass # make stationary
    elif initital_condition is None:
        pass # make zero


    if theta is not None or sigma is not None:
        if theta is None:
            theta = 1
        if sigma is None:
            sigma = 1
        timescale = 1 / theta
        variability = sigma / np.sqrt (2 * theta)
    else:
        theta = 1 / timescale
        sigma = variability * np.sqrt (2 / timescale)

    steps = int( T / dt )
    covariance = np.identity(dimension)
    S = covariance
    target_dimension = dimension

    if steps is not None:
        dt = T/steps

    assert not ( (covariance is not None) and (mixing_matrix is not None)), "you cannot specify both, covariance AND mixing_matrix"

    if covariance is not None:
        covariance = covariance
        (n,m) = np.shape(covariance)
        assert n==m, "covariance must square"
        assert np.all(np.linalg.eigvals(covariance) >= 0), "covariance is not positive definite"
        dimension = n
        target_dimension = n
        S = np.linalg.cholesky(covariance)

    elif mixing_matrix is not None:
        S = mixing_matrix
        (n,m) = np.shape(S)
        covariance = S @ S.T
        dimension = m
        target_dimension = n

    sqdt = np.sqrt(dt)
    t = np.linspace(0,T,steps+1)
    X = np.zeros((samples,target_dimension,steps+1))

    X[0,:,0] = initial_condition


    for i in range(samples):
        x = np.zeros((target_dimension,steps+1))
        dw = S @ np.random.randn(dimension, steps+1)
        x[:,0] = 0

        if stationary:
            x[:,0] = S @ np.random.randn(dimension)*sigma/np.sqrt(2*theta)

        for j in range(steps):
            x[:,j+1] = x[:,j] + (-theta) * dt * x[:,j]+ sigma * sqdt * dw[:,j]

        X[i] = x

    return {
            'variability': variability,
            'timescale': timescale,
            'noise_covariance': covariance,
            'steps': steps,
            'dt': dt,
            't': t,
            'X': X
            }

def exponential_ornsteinuhlenbeck(T,dt,mean=1,coeff_var=1,**kwargs):
    r"""
    Generates a non-negative, multivariate stochastic process
    :math:`X(t)` that is the exponential of an Ornstein-Uhlenbeck
    Process :math:`Z(t)` like so

    .. math::

        X(t) = A exp( B Z(t) ).

    The constants `A`, `B` are chosen such that the process `X(t)` has
    the specified mean (default 1) and coefficient of
    variation (default 1). The Ornstein-Uhlenbeck process chosen
    here is the solution to :math:`dZ = - Z dt + \sqrt(2) dW`

    Returns realizations on the
    time interval [0,T] at increments dt.

    Parameters
    ----------
    T : float
        The upper bound of the stochastic integral.
    dt : float
        size of the time step. Will be overridden if ``steps`` is provided instead.
    mean : float, default = 1
        Desired mean of the resulting realizations.
    coeff_var : float, default = 1
        Desired coefficient of variation of the resulting realizations.
    **kwargs
        Additional keyword arguments that will be passed
        to :func:`stopro.stopro.ornsteinuhlenbeck`.

    Returns
    -------
    result : dict

        Result in the following structure:

        .. code:: python

            {
                'X': 'numpy.ndarray of shape ``(samples, target_dimension, steps+1)`` such that  X[i,j,k] is time point k of component j of realization i',
                't': 'array of time points',
                'dt': 'time increment',
                'steps': 'numper of time steps',
                'noise_covariance': 'covariance matrix, numpy.ndarray of shape ``(target_dimension, target_dimension)``',
                'variability': 'Standard deviation of the underlying OU process',
                'timescale': 'Autocorrelation time of the process',
                'mean': 'as defined above',
                'coeff_var': 'as defined above',
            }
    Notes
    -----
    This function calls :func:`stopro.stopro.ornsteinuhlenbeck` and rescales
    the result. Check out :func:`stopro.stopro.ornsteinuhlenbeck` for more information
    about function parameters.
    """
    A = mean / np.sqrt(1+coeff_var**2)
    B = np.sqrt(np.log(1+coeff_var**2))

    res = ornsteinuhlenbeck(T,dt,**kwargs)

    res["X"] = A*np.exp(B*res["X"])

    res["mean"] = mean
    res["coeff_var"] = coeff_var

    return res

