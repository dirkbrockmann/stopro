# -*- coding: utf-8 -*-
"""
Contains functions simulating elementary stochastic processes.
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
    T : float
        terminal time of time interval [0,T]

    dt : float 
        time increment

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
            'steps': steps
            'covariance': covariance matrix
        }
            
    """
 
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

# Ornstein-Uhlenbeck Process

def ornsteinuhlenbeck(T,dt,variability=1,timescale=1,dimension=1,samples=1,stationary=False,covariance=None, mixing_matrix=None,sigma=None,theta=None):
    """Generates realizations of a multivariate Ornstein-Uhlenbeck process (OUP)

    The OUP is the solution to the stochastic differential equation 
    
    dX = -theta X dt + sigma dW, where dW is the differential of the Wiener Process
    
    Returns realizations (samples) on the time interval [0,T] at increments of size dt.
    You can specify the number of realizations, the covariance in case the processes
    is multidimensional
    
    Parameters
    ----------
    T : float
        terminal time of time interval [0,T]

    dt : float 
        time increment

    variability: float
        the standard deviation of the process in equilibrium, related to the parameters sigma, theta
        according to variability = sigma / sqrt (2 * theta).
    
    timescale: float
        the timescale of the process according to the auto-correlation time, related to the parameters sigma, theta
        according to timescale = 1/theta
    
    dimension : int
        The dimension of the process

    samples : int
        The number of samples generated

    stationary : bool
        If True, a stationary OUP is computed, i.e. initial conditions are drawn from the equilibrium pdf. 
        If false (default), initial conditions are X(0)=0 for all components of the process.

    covariance: N x N matrix
        In case of a multivariate process the covariance matrix (N is the number of components), 
        which must be positive semidefinite. If specified overrides dimension parameter.

    mixing_matrix: N x M matrix
        This matrix, let's call it S with elements S_ij is used to generate an 
        N-dimensional covariant Wiener processes W (with components W_i, i=1,...,N) by superposition
        of independent components V_j of an M-dimensional Wiener process V : W_i = sum_j S_ij * V_j.
        The covariance of W is given by S*S^T.
        Specifying the mixing matrix overrides the covariance parameter.
    
    sigma: float
        The prefactor of the noise term. If specified timescale and variability are overridden. If parameter theta
        is not specified, theta is set to 1.

    theta: float
        The prefactor of the noise term. If specified timescale and variability are overridden. If parameter theta
        is not specified, theta is set to 1.
    
    
    Returns
    -------
    dict
        {
            'X': dictionary of the realizations so that e.g. X[i][j] is component j of realization i
            't': array of times
            'dt': time increment
            'steps': steps
            'variability' : variability of the process
            'timescale' : timescale of the process
            'theta' : force factor in SDE
            'sigma' : prefactor of noise in SDE
            'noise_covariance': covariance matrix of the underlying Wiener process
        }
            
    """
    
    steps = int( T / dt );
    
    if theta is not None or sigma is not None:
        if theta is not None and sigma is None:
            sigma = 1

        if theta is None and sigma is not None:
            theta = 1

        timescale = 1 / theta
        variability = sigma / np.sqrt (2 * theta)
    else:
        theta = 1 / timescale;
        sigma = variability * np.sqrt (2 / timescale)      
    

    if covariance is not None:    
        (n,m) = np.shape(covariance)
        assert n==m, "covariance must square"
        assert np.all(np.linalg.eigvals(covariance) >= 0), "covariance is not positive definite"
        dimension = n
        target_dimension = n
        S = np.linalg.cholesky(covariance)
    else:
        covariance = np.identity(dimension)
        S = covariance;
        target_dimension = dimension

    if mixing_matrix is not None:
        S = mixing_matrix
        (n,m) = np.shape(S)
        covariance = S @ S.T
        dimension = m
        target_dimension = n
        
    sqdt = np.sqrt(dt)
    t = np.linspace(0,T,steps+1)
    X = {}
    
    
    for i in range(samples):        
        x = np.zeros([target_dimension,steps+1])
        dw = S @ np.random.randn(dimension, steps+1)
        x[:,0]=0
        
        if stationary:
            x[:,0] = S @ np.random.randn(dimension)*sigma/np.sqrt(2*theta)
        
        for j in range(steps):
            x[:,j+1] = x[:,j] + (-theta) * dt * x[:,j]+ sigma * sqdt * dw[:,j]
        
        X[i]=x
        
    return {
        'X':X,
        't':t,
        'dt':dt,
        'steps':steps,
        'variability':variability,
        'timescale':timescale,
        'theta':theta,
        'sigma':sigma,
        'noise_covariance':covariance
    }

# Exponential Ornstein-Uhlenbeck Process
    
def exponential_ornsteinuhlenbeck(T,dt,mean=1,coeff_var=1,timescale=1,dimension=1,samples=1,stationary=False,covariance=None, mixing_matrix=None):
    """Generates realizations of an exponential multivariate Ornstein-Uhlenbeck process (eOUP)

    The eOUP X(t) is the exponential of an ordinary OUP Z(t) like so X(t) = A exp( B Z(t) ). 
    The constants A, B are chosen such that the process X(t) has the specified mean (default 1) 
    and coefficient of variation (default 1). 
   
    The Ornstein Uhlenbeck process chosen here is the solution to dZ = - Z dt + sqrt(2)*dW
     
    Returns realizations (samples) on the time interval [0,T] at increments of size dt.
    You can specify the number of realizations, the covariance in case the processes
    is multidimensional
    
    Parameters
    ----------
    T : float
        terminal time of time interval [0,T]

    dt : float 
        time increment

    mean: float
        the mean <X(t)> of the process
    
    coeff_var: float
        the coefficient of variation (standard deviation divided by the mean) of the process X(t)
    
    dimension : int
        The dimension of the process

    samples : int
        The number of samples generated

    stationary : bool
        If True, a stationary OUP is computed, i.e. initial conditions are drawn from the equilibrium pdf. 
        If false (default), initial conditions are X(0)=0 for all components of the process.

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
            'steps': steps
            'variability' : variability of the process
            'timescale' : timescale of the process
            'theta' : force factor in SDE
            'sigma' : prefactor of noise in SDE
            'mean' : mean of the process
            'coeff_var' : coefficient of variation of the process
            'noise_covariance': covariance matrix of the underlying Wiener process
        }
            
    """

    A =  mean / np.sqrt(1+coeff_var**2); B = np.sqrt(np.log(1+coeff_var**2))

    res = ornsteinuhlenbeck(T,dt,
                            timescale=timescale,
                            dimension=dimension,
                            samples=samples,
                            stationary=stationary,
                            covariance=covariance,
                            mixing_matrix=mixing_matrix)

    for i in range(len(res["X"])):
        res["X"][i]=A*np.exp(B*res["X"][i])
    
    res["mean"] = mean
    res["coeff_var"] = coeff_var

    return res

