# -*- coding: utf-8 -*-
"""
Contains functions simulating the stochastic processes.
"""

import numpy as np

# Wiener Process

def wiener(T,dt,dimension=1,samples=1,covariance=None,mixing_matrix=None):
    """Generates realizations of a multivariate Wiener process

    Returns realizations (samples) on the time interval [0,T] at increments of size dt.
    You can specify the number of realizations, the covariance in case the processes
    is multidimensional
    
    Parameters
    ----------
    dimension : int
        The dimension of the process

    samples : int
        The number of samples generated

    covariance: N x N matrix
        In case of a multivariate process the covariance matrix (N is the number of components), 
        which must be positive semidefinite. If specified overrides dimension parameter.

    mixing_matrix: N x M matrix
        This matrix, let's call it S with elements S_ij is used to generate an 
        N-dimensional covariant Wiener processes W (with components W_i, i=1,...,N) by superposition
        of independent components V_j of an M-dimensional Wiener process V : W_i = sum_j S_ij * V_j.
        The covariance of W is given by S*S^T.
        Specifying the mixing matrix overrides the covariance parameter.
    
    Returns
    -------
    dict
        {
            'X': dictionary of the realizations so that e.g. X[i][j] is component j of realization i
            't': array of times
            'dt': time increment
            'steps': steps,
            'covariance': covariance matrix
        }
            
    """
    # I like docstrings to be in numpy format. See here: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
   
    steps = int (T / dt)

    if covariance is not None:
        covariance = np.array(covariance)    
        (n,m) = np.shape(covariance)
        assert n==m, "covariance must square"
        assert np.all(np.linalg.eigvals(covariance) >= 0), "covariance is not positive definite"
        dimension = n
        S = np.linalg.cholesky(covariance)
    else:
        covariance = np.identity(dimension)
        S = covariance
        
    if mixing_matrix is not None:
        S = np.array(mixing_matrix)
        (n,m) = np.shape(S)
        covariance = S @ S.T
        dimension = m
        
    t = np.linspace(0,T,steps+1); X = {}
    
    for i in range(samples):
        dw = S @ np.random.randn(dimension,steps+1)
        dw[:,0] = 0
        W = np.sqrt(dt)*np.cumsum(dw,axis=1)
        X[i] = W

    return {
        'X':X,
        't':t,
        'dt':dt,
        'steps':steps,
        'covariance':covariance
    }

def ornsteinuhlenbeck(T,dt,variability=1,timescale=1,dimension=1,samples=1,stationary=False,**kwargs):
    
    """Generates realizations of the multivariate Ornstein-Uhlenbeck Process X(t)

    The OUP is the solution to the SDE dX = -theta X dt + sigma dW, where W is the Wiener Process
    
    Returns realizations on the time interval [0,T] at increments dt.
    
    Parameters:
    
    - dimension: the dimensionality of the process, defaults to 1
    - samples: number of realizations, defaults to 1
    - variability: (standard deviation), defaults to 1
    - timescale: (autocorrelation time of the process), defaults to 1 
    - stationary: if TRUE the processes are initialized so they are stationary, if FALSE (default) all realizations start at the origin.
    
    The latter are related to the standard parameters theta and sigma by
    
    timescale = 1 / theta
    variability = sigma sqrt(2/timescale)
    
    optional arguments:
    
    - instead of timescale and variability you can specify the standard parameters theta (force constant) and sigma (noise magnitude), in this case variability and timescale are overridden.
    
    - steps (specifies number of time steps instead of the dt increment, 
      in this case dt is set to dt = T / steps)
    
    - covariance = COV where COV is a real, square, positive definite covariance matrix. The covariance specifies the covariance of the Wiener increments dW.
    
    - mixing_matrix = S where S is an NxM matrix which is used to generate N Wiener processes W by superposition of M independent Wiener processes V, so dW = S x dV. 
    
    you can either provide a covariance matrix OR a mixing_matrix, but not both
    
    """
    
    # hier auch eher die parameter oben in der funktionsdefinition
    # spezifizieren, dann wieder if theta is None
    if 'theta' in kwargs or 'sigma' in kwargs:
        theta = kwargs["theta"] if 'theta' in kwargs else 1
        sigma = kwargs["sigma"] if 'sigma' in kwargs else 1
        timescale = 1 / theta
        variability = sigma / np.sqrt (2 * theta)
    else:
        theta = 1 / timescale;
        sigma = variability * np.sqrt (2 / timescale)
        
    steps = int( T / dt );
    covariance = np.identity(dimension)
    S = covariance;
    target_dimension = dimension

    if 'steps' in kwargs:    
        steps = kwargs["steps"]
        dt = T/steps;

    assert not ('covariance' in kwargs and 'mixing_matrix' in kwargs), "you cannot specify both, covariance AND mixing_matrix"

    if 'covariance' in kwargs:    
        covariance = kwargs["covariance"]
        (n,m) = np.shape(covariance)
        assert n==m, "covariance must square"
        assert np.all(np.linalg.eigvals(covariance) >= 0), "covariance is not positive definite"
        dimension = n
        target_dimension = n
        S = np.linalg.cholesky(covariance)

    elif 'mixing_matrix' in kwargs:
        S = kwargs["mixing_matrix"]
        (n,m) = np.shape(S)
        covariance = S @ S.T
        dimension = m
        target_dimension = n

    sqdt = np.sqrt(dt)
    t = np.linspace(0,T,steps+1)
    X = np.zeros((samples,target_dimension,steps+1))


    for i in range(samples):
        x = np.zeros((target_dimension,steps+1))
        dw = S @ np.random.randn(dimension, steps+1)
        x[:,0] = 0

        if stationary:
            x[:,0] = S @ np.random.randn(dimension)*sigma/np.sqrt(2*theta)

        for j in range(steps):
            x[:,j+1] = x[:,j] + (-theta) * dt * x[:,j]+ sigma * sqdt * dw[:,j]

        X[i] = x

    # das hier ist eher unueblich (dict returnen) aber kann man auch machen
    return {'variability':variability,'timescale':timescale,'noise_covariance':covariance,'steps':steps,'dt':dt,'t':t,'X':X}

def exponential_ornsteinuhlenbeck(T,dt,mean=1,coeff_var=1,timescale=1,dimension=1,samples=1,stationary=False,**kwargs):

    """Generates a non-negative, multivariate stochastic process X(t) that is the exponential of an Ornstein-Uhlenbeck Process Z(t) like so X(t) = A exp( B Z(t) ). The constants A, B are chosen such that the process X(t) has the specified mean (default 1) and coefficient of variation (default 1). The Ornstein Uhlenbeck process chosen here is the solution to dZ = - Z dt + sqrt(2)*dW

    Returns realizations on the
    time interval [0,T] at increments dt.

    Parameters:

    - timescale: timescale of the process (correlation time)
    - dimension: the dimension of the process
    - samples: number of realizations generated
    - mean: The mean values of the process, defaults to 1
    - coeff_var: coefficient of variation, defaults to 1
    - stationary: if set to TRUE the processes are initialized so they are stationary, if FALSE all realizations start at the origin.

    Optional parameters:

    - steps (specifies number of time steps instead of the dt increment,
      in this case dt is set to dt = T / steps)

    - covariance = COV where COV is a real, square, positive definite covariance matrix. The covariance specifies the covariance of the Wiener increments dW.

    - mixing_matrix = S where S is an NxM matrix which is used to generate N Wiener processes W by superposition of M independent Wiener processes V, so dW = S x dV.

    you can either provide a covariance matrix OR a mixing_matrix, but not both

    """
    A =  mean / np.sqrt(1+coeff_var**2); B = np.sqrt(np.log(1+coeff_var**2))

    res = ornsteinuhlenbeck(T,dt,
                            timescale=timescale,
                            dimension=dimension,
                            samples=samples,
                            stationary=stationary,**kwargs)

    res["X"] = A*np.exp(B*res["X"])

    res["mean"] = mean
    res["coeff_var"] = coeff_var

    return res

