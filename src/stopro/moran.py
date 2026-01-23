import numpy as np
from math import inf
from math import isinf
from scipy.integrate import odeint
from scipy.special import logsumexp

def moran_particle_dynamics(n0,T,alpha, 
                            dt = 1, 
                            sigma = False, 
                            A = None, 
                            timescale = 1):
    
    N = len(n0)
    system_size = np.sum(n0)
    V = [(x,y) for x in range(N) for y in range(N)]      
    t = [0]
    X = [tuple(n0)]
    n = n0.copy()

    if sigma is not False:
        alpha = exponential_ornstein_uhlenbeck(T = T, dt = dt, coeff_var = sigma, mixing_matrix = A, mean = alpha, timescale = timescale)['X'][0]

    while t[-1] < T:        

        if np.ndim(alpha)>1:

            r = np.zeros((N*N, int(alpha.shape[1])))
            for i in range(int(alpha.shape[1])):
                r[:,i] = (np.outer(alpha[:,i]*n,n)/system_size).flatten()
            rtot = np.sum(r, axis = 0)
            if len(rtot) == 1: break
            tau = stochastic_tau(rtot, dt)
            t.append(t[-1]+(1+tau)*dt)
            r = r[:,tau]
            rtot = rtot[tau]
            alpha = alpha[:,(1+tau):]

        else:

            r = (np.outer(alpha*n,n)/system_size).flatten()
            rtot = np.sum(r)
            tau = np.random.exponential(1.0/rtot)
            t.append(t[-1]+tau)

        P = r/rtot
        dn = V[np.random.choice(range(N*N),p = P)]

        n[dn[0]]+=1
        n[dn[1]]-=1
        X.append(tuple(n))
        
    return (t,X)
        
def moran_diffusion_approximation(n0,T,alpha,dt):
    
    N = len(n0)
    system_size = np.sum(n0)
    
    t = [0]
    X = [tuple(n0)]
    
    n = list(n0)
        
    while t[-1] < T:
        W = np.random.normal(size=(N,N))
        Q = alpha*W-(alpha*W).T
        S = np.sqrt(np.outer(n,n)) * Q * np.sqrt(dt/system_size)
        phi = np.sum(n*alpha/system_size);
        dn = n*(alpha-phi)*dt + np.sum(S,axis=0);
        n += dn;
        n[n<0] = 0
        t.append(t[-1]+dt)
        X.append(tuple(n))
    
    return (t,X)
    
def moran(T,n0,alpha,
          system_size=None,
          diffusion_approximation=False,
          dt=None, 
          normalize=False,
          samples=1,
          sigma = False, 
          A = None, 
          timescale = 1):
    """
    
    Parameters
    ----------
    T : float
        Time interval.
    n0: numpy.ndarray of shape (``M``,)
        Initial abundances / fractions of species
    system_size : int or None or inf, default is None
        Total population size, i.e. the number of individuals altogether
        If inf deterministic solution is found, required additional keyword dt to be set
    alpha : numpy.ndarray of shape (``N``,), default = 1
        Replication rate of species
    samples : int, default = 1
        The number of samples generated, ignored if system_size = inf (deterministic system)
    normalize : boolean, default = False
        If True returns fractions of the population, instead of absolute abundance
        It True n0 will be automatically rescaled to be normalized to unity
    diffusion_approximation: False
        If True computes the realizations of the diffusion approximation process for the system
        requires to set keyword dt
    dt: float
        Needs to be specified if the diffusion approximation is used. This is the time increment in that
        case

    Returns
    -------
    result : numpy.ndarray:
        each array element is a sample, each sample is a tuple (t,X) or time array t and array of state vectors X.
    

    """ 
    
    if system_size is not None:
        if isinf(system_size):
            rep = lambda x,t,a : x*(a-np.sum(a*x))
            x0 = np.array(n0/np.sum(n0))
            t = np.linspace(0,T,int(T/dt))
            sol = odeint(rep,x0,t,args=(alpha,))
            X = [tuple(v) for v in sol]
            return (t,X)
        else:
            normalize = True
            x0 = n0/np.sum(n0)
            n0 = system_size*x0
            n0 = n0.astype(int)
    else:
        n0 = n0.astype(int)
        system_size = np.sum(n0)
        
    if diffusion_approximation:
        res = [moran_diffusion_approximation(n0,T,alpha,dt) for i in range(samples)]
    else:
        res = [moran_particle_dynamics(n0,T,alpha, dt, sigma, A, timescale) for i in range(samples)]
    
    if normalize:
        for i in range(len(res)):
            res[i]=(res[i][0],[tuple(list(v)/sum(v)) for v in res[i][1]])
    
    return res

    