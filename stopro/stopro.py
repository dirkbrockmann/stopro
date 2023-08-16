# -*- coding: utf-8 -*-
"""
Contains functions simulating elementary stochastic processes.
"""
# add brownian bridge

import numpy as np
from math import inf

def wiener_differential(dt,steps=1,**kwargs):
    return wiener(T,None,steps=steps,**kwargs)


def wiener(T,dt,gap=1,N=1,samples=1,covariance=None,mixing_matrix=None,steps=None):

    """
    Generates realizations of a multivariate Wiener Wurst process.

    Returns realizations (samples) on the time interval [0,T] at increments of size dt.
    You can specify the number of realizations, as well as the covariance or
    mixing matrix in case the process is multidimensional.

    Parameters
    ----------
    T : float
        Time interval.
    dt : float
        Time step size. Will be overridden if ``steps`` is provided instead.
    gap : int, default = 1
        Gap between saved (returned) points.
    N : int, default = 1
        The dimension of the stochastic process.
    samples : int, default = 1
        The number realizations.
    covariance : numpy.ndarray of shape (N, N), default = None
        In case of a multivariate process the covariance matrix
        which must be positive semidefinite. If specified, overrides the ``N`` parameter with N.
    mixing_matrix : numpy.ndarray of shape (N, M), default = None
        This matrix :math:`S_{ij}` is used to generate an
        N-dimensional covariant Wiener process (with components :math`W_i, i=1,...,N`) by superposition
        of independent components :math:`V_j, j=1,...,M` of an M-dimensional Wiener process V.

    .. math::

        W_i = \sum_j S_{ij} \cdot V_j.

        The covariance of W is given by :math:`S \cdot S^T`.
        If specified, overrides the covariance parameter and the dimension parameter.
    steps : int, default = None
        If provided, defines a number of time steps. Overrides `dt` parameter.

    Returns
    -------
    result : dict

    Result dictionary containing the following entries

    .. code:: python

        {
            'X': 'numpy.ndarray of shape (samples, N, steps+1) such that  X[i,j,k] is time point k of component j of realization i',
            't': 'numpy.ndarray of shape (samples,steps+1) such that t[i,k] is time point k of realization i',
            'dt': 'float, time increment',
            'steps': 'int, number of time steps',
            'N': 'int, number of components of process',
            'savedsteps': 'int, number of saved points in X',
            'gap': 'int, gap between saved steps',
            'covariance': 'numpy.ndarray of shape (dimension, dimension) such that covariance[i,j] = < dW_i dW_k>'
        }



    """

    if steps is not None:
        dt = T/steps
    else:
        steps = int(T / dt)

    if covariance is not None:
        (n,m) = covariance.shape
        assert n==m, "covariance matrix must be square"
        assert np.all(np.linalg.eigvals(covariance) >= 0), "covariance matrix is not positive definite"
        N = n
        S = np.linalg.cholesky(covariance)
    else:
        S = np.identity(N)

    if mixing_matrix is not None:
        S = np.array(mixing_matrix)
        (n,m) = S.shape
        covariance = S @ S.T
        M = m

    (N, M) = S.shape


    t = np.linspace(0,T,steps+1)
    X = np.zeros((samples,N,steps+1))

    if gap > 1:
        t = t[np.arange(0,steps+1,gap)]
        X = X[:,:,np.arange(0,steps+1,gap)]
        savedsteps = int((steps+1)/gap)
    else:
        savedsteps = steps

    for i in range(samples):
        dw = S @ np.random.randn(M,steps+1)
        dw[:,0] = 0
        W = np.sqrt(dt)*np.cumsum(dw,axis=1)
        if gap > 1:
            X[i] = W[:,np.arange(0,steps+1,gap)]
        else:
            X[i] = W

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

def ornstein_uhlenbeck(T,
                     dt,
                     stdev=1,
                     timescale=1,
                     N=1,
                     gap=1,
                     samples=1,
                     initial_condition=None,
                     covariance=None,
                     mixing_matrix=None,
                     steps=None,
                     theta=None,
                     sigma=None,
                     ):

    """
    Generates realizations of the multivariate Ohrenstein-Uhlendreck Process X(t)
    The OUP is the solution to the SDE

    .. math::

        dX_i = -\theta_i X_i dt + \sigma_i dW_i,

    where :math:`W_i(t)` are components of a multivariate Wiener process.

    Returns realizations on the time interval [0,T] at increments dt.

    Parameters
    ----------
    T : float
        Time interval.
    dt : float
        Time step size. Will be overridden if ``steps`` is provided instead.
    gap : int, default = 1
        Gap between saved (returned) points.
    N : int, default = 1
        The dimension of the stochastic process.
    samples : int, default = 1
        The number realizations.
    covariance : numpy.ndarray of shape (N, N), default = None
        In case of a multivariate process the covariance matrix
        which must be positive semidefinite. If specified, overrides the ``N`` parameter with N.
    mixing_matrix : numpy.ndarray of shape (N, M), default = None
        This matrix :math:`S_{ij}` is used to generate an
        N-dimensional covariant Wiener process (with components :math`W_i, i=1,...,N`) by superposition
        of independent components :math:`V_j, j=1,...,M` of an M-dimensional Wiener process V.

    .. math::

        W_i = \sum_j S_{ij} \cdot V_j.

        The covariance of W is given by :math:`S \cdot S^T`.
        If specified, overrides the covariance parameter and the dimension parameter.
    steps : int, default = None
        If provided, defines a number of time steps. Overrides `dt` parameter.
    stdev : float or numpy.ndarray of shape (``N``,), default = 1
        Standard deviation of :math:`X(t)`.
        Is overridden if any of the parameters ``sigma`` or ``theta`` are provided.
        For multivariate process ``stdev[i]`` specifies the standard deviation of component :math:`X_i(t)`.
    timescale : float of numpy.ndarray of shape (``N``,), default = 1
        Autocorrelation time of the process.
        Is overridden if parameter ``theta`` is provided.
        For multivariate process ``timescale[i]`` specifies the autocorrelation time of component :math:`X_i(t)`.
    initial_condition : str or numpy.ndarray of shape (``N``,), default = None
        If ``initial_condition is None``, process will be initiated at X = 0.
        If ``initial_condition == 'stationary'``, initial conditions will be drawn from stationary distribution.
        Else, process will be initiated as ``X[:,0] = initial_condition``.
    theta : float or numpy.ndarray of shape (``N``,), default = None
        Overrides ``timescale`` and ``stdev`` if defined.
    sigma: float or numpy.ndarray of shape (``N``,), default = None
        Overrides ``stdev`` if defined.

    Returns
    -------
    result : dict

        Result in the following structure:

        .. code:: python

            {
                'X': 'numpy.ndarray of shape (samples, dimension, steps+1)
                      such that  X[i,j,k] is time point k of component j of realization i',
                't': 'numpy.ndarray of shape (samples,steps+1)' such that t[i,k] is time point k of realization i,
                'dt': 'float, time increment',
                'steps': 'int, numper of time steps',
                'dimension': 'int, number of components of process',
                'savedsteps': 'int, number of returned trajectory points in X'
                'gap': 'int, gap between saved steps'
                'noise_covariance': 'numpy.ndarray of shape (dimension, dimension) such that covariance[i,j] = < dW_i dW_k>'
                'stdev': 'float or numpy.ndarray of shape (dimension), standard deviation of components of process',
                'timescale': 'float or numpy.ndarray of shape (dimension), timescale of components of process'',
                'sigma': 'float or numpy.ndarray of shape (dimension), prefactors of noise terms dW_i',
                'theta': 'float or numpy.ndarray of shape (dimension), linear force terms for process components',
                'initial_condition': 'numpy.ndarray of shape (samples, dimension), initial condition for the process'
            }

    Notes
    -----

    You can either provide a covariance matrix OR a mixing_matrix, but not both.
    You should either provide the parameter pair stdev and timescale or theta and sigma.
    """


    if steps is not None:
        dt = T/steps
    else:
        steps = int(T / dt)

    if covariance is not None:
        (n,m) = covariance.shape
        assert n==m, "covariance matrix must be square"
        assert np.all(np.linalg.eigvals(covariance) >= 0), "covariance matrix is not positive definite"
        N = n
        S = np.linalg.cholesky(covariance)
    else:
        S = np.identity(N)

    if mixing_matrix is not None:
        S = np.array(mixing_matrix)
        (n,m) = S.shape
        covariance = S @ S.T
        dimension = m

    (N, M) = S.shape

    if theta is not None or sigma is not None:
        if theta is None:
            if N > 1:
                theta = np.ones(N)
            else:
                theta = 1
        if sigma is None:
            if N > 1:
                sigma = np.ones(N)
            else:
                sigma = 1
        if N > 1:
            if not hasattr(theta,'__len__'):
                theta = theta * np.ones(N)
            if not hasattr(sigma,'__len__'):
                sigma = sigma * np.ones(N)
        try:
            timescale = 1 / theta
            stdev = sigma / np.sqrt (2 * theta)
        except ZeroDivisionError:
            timescale = inf
            stdev = inf
        
    else:
        if N > 1:
            if not hasattr(timescale,'__len__'):
                timescale = timescale * np.ones(N)
            if not hasattr(stdev,'__len__'):
                stdev = stdev * np.ones(N)
            timescale = np.array(timescale)
            stdev = np.array(stdev)
            theta = 1.0 / timescale
            sigma = stdev * np.sqrt (2.0 / timescale)
        else:
            theta = 1.0 / timescale
            sigma = stdev * np.sqrt (2.0 / timescale)


    sqdt = np.sqrt(dt)
    t = np.linspace(0,T,steps+1)
    X = np.zeros((samples,N,steps+1))


    if gap > 1:
        t = t[np.arange(0,steps+1,gap)]
        X = X[:,:,np.arange(0,steps+1,gap)]
        savedsteps = int((steps+1)/gap)
    else:
        savedsteps = steps


    stationary = False
    if isinstance(initial_condition, str):
        if initial_condition.lower() == 'stationary':
            stationary = True
        else:
            raise ValueError(f"unknown initial condition '{stationary}'")
    elif initial_condition is None:
        initial_condition = np.zeros(N)

    
    if stationary == True and (0 in np.array(theta)):
        raise ValueError(f"stationary initial conditions AND theta=0 doesn't work!!")
    for i in range(samples):
        x = np.zeros((N,steps+1))
        dw = S @ np.random.randn(M, steps+1)

        if stationary:
            x[:,0] = S @ np.random.randn(M)*sigma/np.sqrt(2*theta)
        else:
            x[:,0] = initial_condition

        for j in range(steps):
            x[:,j+1] = x[:,j] + (-theta) * dt * x[:,j]+ sigma * sqdt * dw[:,j]

        if gap > 1:
            X[i] = x[:,np.arange(0,steps+1,gap)]
        else:
            X[i] = x


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

def integrated_ornstein_uhlenbeck(T,dt,**kwargs):

    """
    Generates a non-negative, multivariate stochastic process
    :math:`X(t)` that is the integral of a multivariate Ornstein-Uhlenbeck
    Process :math:`Z(t)` like so

    .. math::

        X(t) = \int Z(t) dt.

    Returns realizations on the
    time interval [0,T] at increments dt.

    Parameters
    ----------
    T : float
        The upper bound of the stochastic integral.
    dt : float
        size of the time step. Will be overridden if ``steps`` is provided instead.
    **kwargs
        Additional keyword arguments that will be passed
        to :func:`stopro.stopro.ornsteinuhlenbeck`.

    Returns
    -------
    result : dict

        Result in the following structure:

        .. code:: python

            {
                'X': 'numpy.ndarray of shape (samples, dimension, steps+1)
                      such that  X[i,j,k] is time point k of component j of realization i',
                't': 'numpy.ndarray of shape (samples,steps+1)' such that t[i,k] is time point k of realization i,
                'dt': 'float, time increment',
                'steps': 'int, numper of time steps',
                'N': 'int, number of components of process',
                'savedsteps': 'int, number of returned trajectory points in X'
                'gap': 'int, gap between saved steps'
                'noise_covariance': 'numpy.ndarray of shape (dimension, dimension) such that covariance[i,j] = < dW_i dW_k>'
                'stdev': 'float or numpy.ndarray of shape (dimension), standard deviation of components of process',
                'timescale': 'float or numpy.ndarray of shape (dimension), timescale of components of process'',
                'sigma': 'float or numpy.ndarray of shape (dimension), prefactors of noise terms dW_i',
                'theta': 'float or numpy.ndarray of shape (dimension), linear force terms for process components',
                'initial_condition': 'numpy.ndarray of shape (samples, dimension), initial condition for the process'
            }


    """

    if "gap" in kwargs:
        gap = kwargs['gap']
        del kwargs["gap"]
    else:
        gap = 1

    res = ornstein_uhlenbeck(T,dt,gap=1,**kwargs)

    t = res["t"]
    x =res["X"]
    y = dt*np.cumsum(x,axis=2)

    steps = res["steps"]

    if gap > 1:
        t = t[np.arange(0,steps+1,gap)]
        y = y[:,:,np.arange(0,steps+1,gap)]
        savedsteps = int((steps+1)/gap)
    else:
        savedsteps = steps


    res["X"]=y
    res["t"]=t
    res["savedsteps"] = savedsteps
    res["gap"] = gap
    return res

def geometric_brownian_motion(T,
                      dt,
                      mu=1,
                      sigma=1,
                      initial_condition=None,
                      **kwargs,
                      ):

    """
    Generates realizations of multivariate, geometric Brownian motion (GBM)

    GBM is the solution to the SDE

    .. math::

        dX_i = \mu_i X_i dt + \sigma_i X_i dW_i,

    where W_i are covariant Wiener processes. The solution
    of the processes are given by

    .. math::

        X_i(t)=X_i(0) * Exp [(\mu_i - 1/2*\sigma_i^2)t+\sigma_i W_i(t)].

    Returns realizations on the time interval [0,T] at increments dt.

    Parameters
    ----------
    T : float
        Time interval.
    dt : float
        Time step size. Will be overridden if ``steps`` is provided instead.
    mu : float or numpy.ndarray of shape (``N``,), default = 1
    sigma : float of numpy.ndarray of shape (``N``,), default = 1
    initial_condition : str or numpy.ndarray of shape (``N``,), default = None
        If ``initial_condition is None``, process will be initiated at X = 1.
        Else, process will be initiated as ``X[:,0] = initial_condition``.
    **kwargs
        Additional keyword arguments that will be passed
        to :func:`stopro.stopro.wiener`.

    Returns
    -------
    result : dict


        Result in the following structure:

        .. code:: python

            {
                'X': 'numpy.ndarray of shape (samples, dimension, steps+1)
                      such that  X[i,j,k] is time point k of component j of realization i',
                't': 'numpy.ndarray of shape (samples,steps+1)' such that t[i,k] is time point k of realization i,
                'dt': 'float, time increment',
                'steps': 'int, numper of time steps',
                'N': 'int, number of components of process',
                'savedsteps': 'int, number of returned trajectory points in X'
                'gap': 'int, gap between saved steps'
                'noise_covariance': 'numpy.ndarray of shape (dimension, dimension) such that covariance[i,j] = < dW_i dW_k>'
                'sigma': 'float or numpy.ndarray of shape (dimension), prefactors of noise terms dW_i',
                'mu': 'float or numpy.ndarray of shape (dimension), linear force terms for process components',
                'initial_condition': 'numpy.ndarray of shape (samples, dimension), initial condition for the process'
            }

    Notes
    -----

    You can either provide a covariance matrix OR a mixing_matrix, but not both.
    """

    if "gap" in kwargs:
        gap = kwargs['gap']
        del kwargs["gap"]
    else:
        gap = 1

    res = wiener(T,dt,gap=1,**kwargs)
    X = res["X"]
    t = res["t"]
    D = res["N"]

    if initial_condition is None:
        if D > 1:
            x0 = np.ones(D)
        else:
            x0 = 1
    else:
        if D > 1:
            if not hasattr(initial_condition,'__len__'):
                x0 = initial_condition * np.ones(D)
            else:
                x0 = initial_condition
        else:
            x0 = initial_condition

    if D > 1:
        if not hasattr(mu,'__len__'):
            mu = mu * np.ones(D)

        if not hasattr(sigma,'__len__'):
            sigma = sigma * np.ones(D)

        for i in range(D):
            X[:,i,:] = x0[i]*np.exp( ( mu[i] - 0.5*sigma[i]**2 ) * t + sigma[i]*X[:,i,:])
    else:
        X = x0 * np.exp( (mu - 0.5*sigma**2)*t + sigma*X)

    steps = res["steps"]

    if gap > 1:
        t = t[np.arange(0,steps+1,gap)]
        X = X[:,:,np.arange(0,steps+1,gap)]
        savedsteps = int((steps+1)/gap)
    else:
        savedsteps = steps

    res["X"] = X
    res["t"] = t
    res["mu"] = mu
    res["sigma"] = sigma
    res["initial_condition"] = x0
    res["noise_covariance"] = res["covariance"]
    del res["covariance"]


    return res

def colored_geometric_brownian_motion(T,
                      dt,
                      mu=1,
                      sigma=1,
                      tau=1,
                      N=1,
                      initial_condition=None,
                      **kwargs,
                      ):

    """
    Generates realizations of multivariate, colored geometric Brownian motion (cGBM)

    cGBM is the solution to the SDE

    .. math::

        dX_i = \mu_i X_i dt + \sigma_i Z_i dt,
        tau_i dZ_i = -Z_i + dW_i

    where W_i are covariant Wiener processes.

    Returns realizations on the time interval [0,T] at increments dt.

    Parameters
    ----------
    T : float
        Time interval.
    dt : float
        Time step size. Will be overridden if ``steps`` is provided instead.
    mu : float or numpy.ndarray of shape (``N``,), default = 1
    sigma : float of numpy.ndarray of shape (``N``,), default = 1
    tau : float of numpy.ndarray of shape (``N``,), default = 1
    initial_condition : str or numpy.ndarray of shape (``N``,), default = None
        If ``initial_condition is None``, process will be initiated at X = 1.
        Else, process will be initiated as ``X[:,0] = initial_condition``.
    **kwargs
        Additional keyword arguments that will be passed
        to :func:`stopro.stopro.wiener`.

    Returns
    -------
    result : dict


        Result in the following structure:

        .. code:: python

            {
                'X': 'numpy.ndarray of shape (samples, dimension, steps+1)
                      such that  X[i,j,k] is time point k of component j of realization i',
                't': 'numpy.ndarray of shape (samples,steps+1)' such that t[i,k] is time point k of realization i,
                'dt': 'float, time increment',
                'steps': 'int, numper of time steps',
                'N': 'int, number of components of process',
                'savedsteps': 'int, number of returned trajectory points in X'
                'gap': 'int, gap between saved steps'
                'noise_covariance': 'numpy.ndarray of shape (dimension, dimension) such that covariance[i,j] = < dW_i dW_k>'
                'sigma': 'float or numpy.ndarray of shape (dimension), prefactors of noise terms dW_i',
                'mu': 'float or numpy.ndarray of shape (dimension), linear force terms for process components',
                'initial_condition': 'numpy.ndarray of shape (samples, dimension), initial condition for the process'
            }

    Notes
    -----

    You can either provide a covariance matrix OR a mixing_matrix, but not both.
    """

    if "gap" in kwargs:
        gap = kwargs['gap']
        del kwargs["gap"]
    else:
        gap = 1

    res = integrated_ornstein_uhlenbeck(T,dt,theta=1.0/tau,sigma=1.0/tau,N=N,gap=1,initial_condition="stationary",**kwargs)

    X = res["X"]
    t = res["t"]
    D = res["N"]

    if initial_condition is None:
        if D > 1:
            x0 = np.ones(D)
        else:
            x0 = 1
    else:
        if D > 1:
            if not hasattr(initial_condition,'__len__'):
                x0 = initial_condition * np.ones(D)
            else:
                x0 = initial_condition
        else:
            x0 = initial_condition

    if D > 1:
        if not hasattr(mu,'__len__'):
            mu = mu * np.ones(D)

        if not hasattr(sigma,'__len__'):
            sigma = sigma * np.ones(D)

        for i in range(D):
            X[:,i,:] = x0[i]*np.exp( ( mu[i] ) * t + sigma[i]*X[:,i,:])
    else:
        X = x0 * np.exp( mu *t + sigma*X)

    steps = res["steps"]

    if gap > 1:
        t = t[np.arange(0,steps+1,gap)]
        X = X[:,:,np.arange(0,steps+1,gap)]
        savedsteps = int((steps+1)/gap)
    else:
        savedsteps = steps

    res["X"] = X
    res["t"] = t
    res["mu"] = mu
    res["sigma"] = sigma
    res["tau"] = tau
    res["N"] = N
    res["initial_condition"] = x0
    return res

def gillespie_replicator(T,
                     dt,
                     N=2,
                     initial_condition=None,
                     **kwargs
                     ):

    """
    Generates realizations :math:`Y(t)` of a multivariate, stochastic replicator model, originally analyzed by Gillespie.
    The foundation is a set of multivariate geometric Brownian motion processes with parameters :math:`(\mu_i,\sigma_i)`:

    .. math::

        dX_i = \mu_i X_i dt + \sigma_i X_i dW_i,

    where W_i are covariant Wiener processes. The Gillespie replicator model is just the normalized version of the process so

    .. math::

        Y_i(t)=X_i(t) / \sum_j X_(t)


    Parameters
    ----------
    T : float
        Time interval.
    dt : float
        Time step size. Will be overridden if ``steps`` is provided instead.
    N : int, default 2
        Number of species, must be > 1.
    initial_condition : str or numpy.ndarray of shape (``N``,), default = None
        If ``initial_condition is None``, process will be initiated at X = 1.
        Else, process will be initiated as ``X[:,0] = initial_condition``.
    **kwargs
        Additional keyword arguments that will be passed
        to :func:`stopro.stopro.geometric_brownian_motion`.

    Returns
    -------
    result : dict


    .. code:: python

        {
            'X': 'numpy.ndarray of shape (samples, dimension, steps+1)
                  such that  X[i,j,k] is time point k of component j of realization i',
                't': 'numpy.ndarray of shape (samples,steps+1)' such that t[i,k] is time point k of realization i,
                'dt': 'float, time increment',
                'steps': 'int, numper of time steps',
                'N': 'int, number of components of process',
                'savedsteps': 'int, number of returned trajectory points in X'
                'gap': 'int, gap between saved steps'
                'noise_covariance': 'numpy.ndarray of shape (dimension, dimension) such that covariance[i,j] = < dW_i dW_k>'
                'sigma': 'numpy.ndarray of shape (dimension), prefactors of noise terms dW_i',
                'mu': 'numpy.ndarray of shape (dimension), linear force terms for process components',
                'initial_condition': 'numpy.ndarray of shape (samples, dimension), initial condition for the process'
        }


    """

    assert N > 1, "The number of species n must be greater than 1"

    if "gap" in kwargs:
        gap = kwargs['gap']
        del kwargs["gap"]
    else:
        gap = 1


    if initial_condition is not None:
        initial_condition = initial_condition / np.sum(initial_condition,axis=0)
    else:
        initial_condition = np.ones(N)/N

    res = geometric_brownian_motion(T,dt,gap=1,N=N,initial_condition=initial_condition,**kwargs)

    Y = res["X"]
    t = res["t"]
    D = res["N"]

    steps = res["steps"]

    if gap > 1:
        t = t[np.arange(0,steps+1,gap)]
        Y = Y[:,:,np.arange(0,steps+1,gap)]
        savedsteps = int((steps+1)/gap)
    else:
        savedsteps = steps


    gront = np.sum (Y,axis=1)
    res["t"] = t
    res["X"] = Y / gront[:,None,:]

    return res

def kimura_replicator(T,dt,
                    N=2,
                    mu=1.0,
                    sigma=1.0,
                    initial_condition=None,
                    gap=1,
                    samples=1,
                    covariance=None,
                    mixing_matrix=None,
                    steps=None):
    """
    Generates realizations of the multivariate, stochastic replicator model introduced by Kimura:

    .. math::

        dX_i = (mu_i(t)-\phi)  X_i dt

    where

    .. math::

        \phi = sum_j mu_j X_j

    is the mean fitness at time t and the time dependent fitness functions are defined by

    .. math::

        \mu_i (t) dt = \mu_i dt + \sigma_i dW_i


    where W_i are covariant Wiener processes.


    Parameters
    ----------
    T : float
        Time interval.
    dt : float
        Time step size. Will be overridden if ``steps`` is provided instead.
    N : int, default 2
        Number of species, must be > 1.
    initial_condition : str or numpy.ndarray of shape (``N``,), default = None
        If ``initial_condition is None``, process will be initiated at X = 1.
        Else, process will be initiated as ``X[:,0] = initial_condition``.
    mu : float or numpy.ndarray of shape (``N``,), default = 1
    sigma : float of numpy.ndarray of shape (``N``,), default = 1
    gap : int, temporal sampling gap, default = 1
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
        if ``False`` (default) all realizations start at the origin.
    steps : int, default = None
        if provided, defines a number of time steps. Overrides `dt`.

    Returns
    -------
    result : dict

        Result in the following structure:

        .. code:: python

            {
                'X': 'numpy.ndarray of shape (samples, dimension, steps+1)
                      such that  X[i,j,k] is time point k of component j of realization i',
                't': 'numpy.ndarray of shape (samples,steps+1)' such that t[i,k] is time point k of realization i,
                'dt': 'float, time increment',
                'steps': 'int, numper of time steps',
                'N': 'int, number of components of process',
                'savedsteps': 'int, number of returned trajectory points in X'
                'gap': 'int, gap between saved steps'
                'noise_covariance': 'numpy.ndarray of shape (dimension, dimension) such that covariance[i,j] = < dW_i dW_k>'
                'sigma': 'numpy.ndarray of shape (dimension), prefactors of noise terms dW_i',
                'mu': 'numpy.ndarray of shape (dimension), linear force terms for process components',
                'initial_condition': 'numpy.ndarray of shape (samples, dimension), initial condition for the process'
            }

   """


    if steps is not None:
        dt = T/steps
    else:
        steps = int(T / dt)

    if covariance is not None:
        (n,m) = covariance.shape
        assert n==m, "covariance matrix must be square"
        assert np.all(np.linalg.eigvals(covariance) >= 0), "covariance matrix is not positive definite"
        N = n
        S = np.linalg.cholesky(covariance)
    else:
        S = np.identity(N)

    if mixing_matrix is not None:
        S = np.array(mixing_matrix)
        (n,m) = S.shape
        covariance = S @ S.T
        N = m

    (N, M) = S.shape


    if initial_condition is not None:
        initial_condition = initial_condition / np.sum(initial_condition,axis=0)
    else:
        initial_condition = np.ones(N)/N

    if not hasattr(mu,'__len__'):
        mu = mu * np.ones(N)

    if not hasattr(sigma,'__len__'):
        sigma = sigma * np.ones(N)



    assert N > 1, "The number of species N must be greater than 1"
    assert (len(mu) is N and len(sigma) is N), "both parameters mu and sigma must have length equal to n"

    sqdt = np.sqrt(dt)
    t = np.linspace(0,T,steps+1)
    X = np.zeros((samples,N,steps+1))


    if gap > 1:
        t = t[np.arange(0,steps+1,gap)]
        X = X[:,:,np.arange(0,steps+1,gap)]
        savedsteps = int((steps+1)/gap)
    else:
        savedsteps = steps

    for i in range(samples):
        x = np.zeros((N,steps+1))
        dw = S @ np.random.randn(M, steps+1)

        x[:,0] = initial_condition

        for j in range(steps):
            r = mu * dt + sigma * dw[:,j] * sqdt
            phi = np.sum(r * x[:,j])
            dx = (r-phi)*x[:,j]
            x[:,j+1] = x[:,j] + dx

        if gap > 1:
            X[i] = x[:,np.arange(0,steps+1,gap)]
        else:
            X[i] = x

    return {
            'initial_condition': initial_condition,
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

def white_replicator(T,
                      dt,
                      N=2,
                      initial_condition=None,
                      mu=1,
                      sigma=1,
                      **kwargs,
                      ):

    """
    Generates realizations :math:`Y(t)` of a multivariate, white noise replicator model:

    .. math::

        dX_i = \mu_i X_i dt + \sigma_i X_i Z_i(t) dt

    with

    .. math::

        tau_i dZ_i = - Z_i+dW_i

    where :math:`Z_i(t)` are Ornstein Uhlenbeck processes and the limit :math:`tau_i\rightarrow 0` is performed. This
    is equivalent to a Stratonovich interpretation of the SDE:

    .. math::

        dX_i = \mu_i X_i dt + \sigma_i X_i dW_i dt

    or the Ito-Interpretation of the SDE

    .. math::

        dX_i = (\mu_i + \sigma_i^2/2) X_i dt + \sigma_i X_i dW_i dt


    Finally the trajectories are normalized.

    .. math::

        Y_i(t)=X_i(t) / \sum_j X_(t)


    Parameters
    ----------
    T : float
        Time interval.
    dt : float
        Time step size. Will be overridden if ``steps`` is provided instead.
    N : int, default 2
        Number of species, must be > 1.
    mu : float, np.ndarray of shape (N,)
    sigma : float, np.ndarray of shape (N,)
    initial_condition : str or numpy.ndarray of shape (``N``,), default = None
        If ``initial_condition is None``, process will be initiated at X = 1.
        Else, process will be initiated as ``X[:,0] = initial_condition``.
    **kwargs
        Additional keyword arguments that will be passed
        to :func:`stopro.stopro.geometric_brownian_motion`.

    Returns
    -------
    result : dict


    .. code:: python

        {
            'X': 'numpy.ndarray of shape (samples, dimension, steps+1)
                  such that  X[i,j,k] is time point k of component j of realization i',
                't': 'numpy.ndarray of shape (samples,steps+1)' such that t[i,k] is time point k of realization i,
                'dt': 'float, time increment',
                'steps': 'int, numper of time steps',
                'N': 'int, number of components of process',
                'savedsteps': 'int, number of returned trajectory points in X'
                'gap': 'int, gap between saved steps'
                'noise_covariance': 'numpy.ndarray of shape (dimension, dimension) such that covariance[i,j] = < dW_i dW_k>'
                'sigma': 'numpy.ndarray of shape (dimension), prefactors of noise terms dW_i',
                'mu': 'numpy.ndarray of shape (dimension), linear force terms for process components',
                'initial_condition': 'numpy.ndarray of shape (samples, dimension), initial condition for the process'
        }


    """



    assert N > 1, "The number of species N must be greater than 1"

    if "gap" in kwargs:
        gap = kwargs['gap']
        del kwargs["gap"]
    else:
        gap = 1


    if initial_condition is not None:
        initial_condition = initial_condition / np.sum(initial_condition,axis=0)
    else:
        initial_condition = np.ones(N)/N

    if not hasattr(mu,'__len__'):
        mu = mu * np.ones(N)

    if not hasattr(sigma,'__len__'):
        sigma = sigma * np.ones(N)

    res = geometric_brownian_motion(T,dt,N=N,mu=mu+0.5*sigma**2,sigma=sigma,gap=1,initial_condition=initial_condition,**kwargs)
    Y = res["X"]
    t = res["t"]

    steps = res["steps"]

    if gap > 1:
        t = t[np.arange(0,steps+1,gap)]
        Y = Y[:,:,np.arange(0,steps+1,gap)]
        savedsteps = int((steps+1)/gap)
    else:
        savedsteps = steps

    gront = np.sum (Y,axis=1)

    res["X"] = Y / gront[:,None,:]
    res["t"] = t
    res["mu"] = mu
    res["sigma"] = sigma
    res["N"] = N

    return res

def colored_replicator(T,
                      dt,
                      N=2,
                      mu=1.0,
                      sigma=1.0,
                      tau=1.0,
                      initial_condition=None,
                      **kwargs):

    """
    Generates realizations :math:`Y(t)` of a multivariate, colored stochastic replicator model:

    .. math::

        dX_i = \mu_i X_i dt + \sigma_i X_i Z_i(t) dt

    with

    .. math::

        tau_i dZ_i = - Z_i+dW_i

    where :math:`Z_i(t)` are Ornstein Uhlenbeck processes.

    The solution is

    .. math::

        Y_i(t)=X_i(t) / \sum_j X_(t)


    Parameters
    ----------
    T : float
        Time interval.
    dt : float
        Time step size. Will be overridden if ``steps`` is provided instead.
    N : int, default 2
        Number of species, must be > 1.
    mu : float, np.ndarray of shape (N,)
    sigma : float, np.ndarray of shape (N,)
    tau : float, np.ndarray of shape (N,)
    initial_condition : str or numpy.ndarray of shape (``N``,), default = None
        If ``initial_condition is None``, process will be initiated at X = 1.
        Else, process will be initiated as ``X[:,0] = initial_condition``.
    **kwargs
        Additional keyword arguments that will be passed
        to :func:`stopro.stopro.integrated_brownian_motion`.

    Returns
    -------
    result : dict


    .. code:: python

        {
            'X': 'numpy.ndarray of shape (samples, dimension, steps+1)
                  such that  X[i,j,k] is time point k of component j of realization i',
                't': 'numpy.ndarray of shape (samples,steps+1)' such that t[i,k] is time point k of realization i,
                'dt': 'float, time increment',
                'steps': 'int, numper of time steps',
                'N': 'int, number of components of process',
                'savedsteps': 'int, number of returned trajectory points in X'
                'gap': 'int, gap between saved steps'
                'noise_covariance': 'numpy.ndarray of shape (dimension, dimension) such that covariance[i,j] = < dW_i dW_k>'
                'sigma': 'numpy.ndarray of shape (dimension), prefactors of noise terms dW_i',
                'mu': 'numpy.ndarray of shape (dimension), linear force terms for process components',
                'initial_condition': 'numpy.ndarray of shape (samples, dimension), initial condition for the process'
        }


    """
    if "gap" in kwargs:
        gap = kwargs['gap']
        del kwargs["gap"]
    else:
        gap = 1

    res = integrated_ornstein_uhlenbeck(T,dt,theta=1.0/tau,sigma=1.0/tau,N=N,gap=1,initial_condition="stationary",**kwargs)

    X = res["X"]
    t = res["t"]
    D = res["N"]

    if initial_condition is None:
        if D > 1:
            x0 = np.ones(D)
        else:
            x0 = 1
    else:
        if D > 1:
            if not hasattr(initial_condition,'__len__'):
                x0 = initial_condition * np.ones(D)
            else:
                x0 = initial_condition
        else:
            x0 = initial_condition

    if D > 1:
        if not hasattr(mu,'__len__'):
            mu = mu * np.ones(D)

        if not hasattr(sigma,'__len__'):
            sigma = sigma * np.ones(D)

        for i in range(D):
            X[:,i,:] = x0[i]*np.exp( ( mu[i] ) * t + sigma[i]*X[:,i,:])
    else:
        X = x0 * np.exp( (mu - 0.5*sigma**2)*t + sigma*X)

    steps = res["steps"]

    if gap > 1:
        t = t[np.arange(0,steps+1,gap)]
        X = X[:,:,np.arange(0,steps+1,gap)]
        savedsteps = int((steps+1)/gap)
    else:
        savedsteps = steps


    gront = np.sum (X,axis=1)

    res["X"] = X / gront[:,None,:]
    res["t"] = t
    res["mu"] = mu
    res["sigma"] = sigma
    res["tau"] = tau
    res["N"] = N
    res["initial_condition"] = x0
    return res

def exponential_ornstein_uhlenbeck(T,dt,mean=1,coeff_var=1,**kwargs):
    """
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


    if isinstance(coeff_var,(float, int)):
        N = 1
    else:
        N = len(coeff_var)

    A = mean / np.sqrt(1+coeff_var**2)
    B = np.sqrt(np.log(1+coeff_var**2))

    res = ornstein_uhlenbeck(T,dt,N=N,**kwargs)

    X=res["X"] ;

    if isinstance(coeff_var,np.ndarray):
        res["X"] = A[None,:,None]*np.exp(B[None,:,None]*X)

    elif isinstance(coeff_var,(float, int)):
        res["X"] = A*np.exp(B*X)

    res["mean"] = mean
    res["coeff_var"] = coeff_var

    return res


def moran(T,population_size,N=2,alpha=1,normalize=False,initial_condition=None,samples=1):

    """
    Generates realizations of the multispecies, stochasic Moran process of a population of individuals of N different species
    that interact according to the following reaction scheme:
    .. math::

        X_i + X_j \rightarrow 2X_i \quad \text{at rate} \quad \alpha_i 

    Parameters
    ----------
    T : float
        Time interval.
    population_size : int
        Total population size, i.e. the number of individuals altogether
    N : int, default 2
        Number of species, must be > 1.
    initial_condition : str or numpy.ndarray of shape (``N``,), default = None
        If ``initial_condition is None``, process will be initiated at X_i = population_size/N.
        Else, process will be initiated as ``X = initial_condition``.
    alpha : float or numpy.ndarray of shape (``N``,), default = 1
        Replication rate of species
    samples : int, default = 1
        The number of samples generated
    normalize : boolean, default = False
        If True returns fractions of the population, instead of absolute numbers

    Returns
    -------
    result : dict

        Result in the following structure:

    .. code:: python

            {
                'Xt': 'tuple of size samples, each element contains pair (t,X) with array of times and vector X of species abundance'
                'samples': 'int, number of realizations',
                'N': 'int, number of species',
                'initial_condition': 'initial condition',
                'population_size': 'population size',
                'alpha': 'vector with replication rates of species'
                
            }

    """

    V = [(x,y) for x in range(N) for y in range(N)]

    Xt=[]
    
    if normalize: 
        M = population_size
    else:
        M = 1
    
    if initial_condition is None:
        n0 = [int(population_size/N)] * N;
    else:
        n0 = initial_condition
        
    for i in range(samples):
            
        n = n0.copy();
        
        t = [0];
        X = [np.array(n.copy())/M];
    
        while t[-1] < T:        
            r = (np.outer(alpha*n,n)/population_size).flatten();
            rtot = np.sum(r)
            P = r/rtot;
            dt = np.random.exponential(1.0/rtot);
            t.append(t[-1]+dt)
            dn = V[np.random.choice(range(N*N),p = P)];

            n[dn[0]]+=1;
            n[dn[1]]-=1;
            X.append(np.array(n.copy())/M)
        
        Xt.append((t,np.array(X).T))
        
    return {
            'initial_condition': initial_condition,
            'samples':samples,
            'alpha': alpha,
            'N' : N,
            'Xt':Xt,
            'population_size': population_size,
            }


