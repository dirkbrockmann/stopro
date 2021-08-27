import numpy as np

# Wiener Process

def wiener(T,dt,dimension=1,samples=1,**kwargs):
    """Generates a multivariate Wiener Process

    Returns realizations on the
    time interval [0,T] at increments of size dt.
    
    optional arguments:
    
    - steps (specifies number of time steps instead of dt increment, 
      in this case dt is set to dt = T / steps)
    
    - covariance = COV where COV is a real, square, positive definite covariance matrix.
    
    - mixing_matrix = S where S is an NxM matrix which is used to generate N Wiener processes W by superposition of M independent Wiener processes V, so W = S x V. 
    
    you can either provide a covariance matrix OR a mixing_matrix, but not both
    """

    steps = int( T / dt );
    covariance = np.identity(dimension)
    S = covariance;
    
    assert not ('covariance' in kwargs and 'mixing_matrix' in kwargs), "you cannot specify both, covariance AND mixing_matrix"
    
    if 'steps' in kwargs:    
        steps = kwargs["steps"]
        dt = T/steps;

    if 'covariance' in kwargs:    
        covariance = kwargs["covariance"]
        (n,m) = np.shape(covariance)
        assert n==m, "covariance must square"
        assert np.all(np.linalg.eigvals(covariance) >= 0), "covariance is not positive definite"
        dimension = n
        S = np.linalg.cholesky(covariance)

    elif 'mixing_matrix' in kwargs:
        S = kwargs["mixing_matrix"]
        (n,m) = np.shape(S)
        covariance = S @ S.T
        dimension = m
        
    t = np.linspace(0,T,steps+1); X = {}
    
    for i in range(samples):
        dw = S @ np.random.randn(dimension,steps+1);
        dw[:,0]=0;
        W = np.sqrt(dt)*np.cumsum(dw,axis=1);
        X[i]=W;
        
    return {'covariance':covariance,'steps':steps,'dt':dt,'t':t,'X':X}

def ornsteinuhlenbeck(T,dt,theta=1,sigma=1,dimension=1,samples=1,stationary=False,**kwargs):
    
    """Generates a multivariate Ornstein-Uhlenbeck Process X(t)

    The OUP is the solution to the SDE dX = -theta X dt + sigma dW, where W is the Wiener Process
    
    Returns realizations on the
    time interval [0,T] at increments dt.
    
    optional arguments:
    
    - stationary: if set to TRUE the processes are initialized to they are stationarly, of FALSE all realizations start at the origin.
    
    - steps (specifies number of time steps instead of the dt increment, 
      in this case dt is set to dt = T / steps)
    
    - covariance = COV where COV is a real, square, positive definite covariance matrix. The covariance specifies the covariance of the Wiener increments dW.
    
    - mixing_matrix = S where S is an NxM matrix which is used to generate N Wiener processes W by superposition of M independent Wiener processes V, so dW = S x dV. 
    
    you can either provide a covariance matrix OR a mixing_matrix, but not both
    
    """

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
        
    sqdt = np.sqrt(dt);
    t = np.linspace(0,T,steps+1); X = {}
    
    
    for i in range(samples):        
        x = np.zeros([target_dimension,steps+1]);
        dw = S @ np.random.randn(dimension, steps+1);
        x[:,0]=0;
        
        if stationary:
            x[:,0] = np.random.randn(target_dimension)*sigma/np.sqrt(2*theta)
        
        for j in range(steps):
            x[:,j+1] = x[:,j] + (-theta) * dt * x[:,j]+ sigma * sqdt * dw[:,j];
        
        X[i]=x;
        
    return {'theta':theta,'sigma':sigma,'covariance':covariance,'steps':steps,'dt':dt,'t':t,'X':X}
    
def exponential_ornsteinuhlenbeck(T,dt,mean=1,coeff_var=1,dimension=1,samples=1,stationary=False,**kwargs):
    """Generates a non-negative, multivariate stochastic process X(t) that is the exponential of an Ornstein-Uhlenbeck Process Z(t) like so X(t) = A exp( B Z(t) ). The constants A, B are chosen such that the process X(t) has the specified mean (default 1) and coefficient of variation (default 1).
    
    Returns realizations on the
    time interval [0,T] at increments dt.
    
    optional arguments:
    
    - stationary: if set to TRUE the processes are initialized to they are stationarly, of FALSE all realizations start at the origin.
    
    - steps (specifies number of time steps instead of the dt increment, 
      in this case dt is set to dt = T / steps)
    
    - covariance = COV where COV is a real, square, positive definite covariance matrix. The covariance specifies the covariance of the Wiener increments dW.
    
    - mixing_matrix = S where S is an NxM matrix which is used to generate N Wiener processes W by superposition of M independent Wiener processes V, so dW = S x dV. 
    
    you can either provide a covariance matrix OR a mixing_matrix, but not both
    
    """
    A =  mean / np.sqrt(1+coeff_var**2); B = np.sqrt(np.log(1+coeff_var**2))

    res = ornsteinuhlenbeck(T,dt,dimension=dimension,samples=samples,stationary=stationary,a=1,b=np.sqrt(2),**kwargs);

    for i in range(len(res["X"])):
        res["X"][i]=A*np.exp(B*res["X"][i]);           
    
    res["mean"]=mean;
    res["coeff_var"]=coeff_var;

    return res

    
    


           