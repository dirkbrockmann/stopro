import numpy as np
from math import inf
from math import isinf
from scipy.integrate import odeint
from scipy.special import logsumexp

def clv_particle_dynamics(n0,T,alpha,beta,omega,
                          dt = 1, 
                          sigma = False, 
                          A = None, 
                          timescale = 1):
    
    N = len(n0)
    t = [0]
    X = [tuple(n0)]
    n = n0.copy()

    if sigma is not False:
        alpha = exponential_ornstein_uhlenbeck(T = T, dt = dt, coeff_var = sigma, mixing_matrix = A, mean = alpha, timescale = timescale)['X'][0]
    
    while t[-1] < T:

        beta_i =  n * (beta@n)/omega

        if np.ndim(alpha)>1:

            alpha_i = n[:,None] * alpha
            rtot = np.sum(alpha_i, axis = 0) + np.sum(beta_i)
            if len(rtot) == 1: break
            tau = stochastic_tau(rtot, dt)
            t.append(t[-1]+(1+tau)*dt)
            alpha_t = alpha_i[:,tau]
            rtot = rtot[tau]
            alpha = alpha[:,(1+tau):]

        else:

            alpha_i = n * alpha
            rtot = np.sum(alpha_i) + np.sum(beta_i)
            tau = np.random.exponential(1.0/rtot)
            t.append(t[-1]+tau)
            alpha_t = alpha_i 
            
        P1 = (alpha_t+beta_i)/rtot
        selected_species = np.random.choice(range(N),p = P1)
        P2 = alpha_t[selected_species]/(alpha_t[selected_species]+beta_i[selected_species])

        if (np.random.rand() < P2):
            n[selected_species]+=1
        else:
            n[selected_species]-=1

        X.append(tuple(n))
    
    return (t,X)

def clv_diffusion_approximation(n0,T,alpha,beta,omega,dt):

    N = len(n0)
    t = [0]
    X = [tuple(n0)]
    n = list(n0)
        
    while t[-1] < T:
        beta_i = (beta@n)/omega
        alpha_mean = np.sum(n*alpha)/omega
        
        W = np.random.normal(size=N)
        dn = n*(alpha-beta_i)*dt + np.sqrt(n*(alpha+beta_i)*dt)*W;
        n += dn;
        n[n<0] = 0
        t.append(t[-1]+dt)
        X.append(tuple(n))
    
    return (t,X)

def competitive_lotka_volterra(T,x0,alpha,beta,system_size,     
          diffusion_approximation=False,
          normalize=False, 
          samples=1,
          dt = 1, 
          sigma = False, 
          A = None, 
          timescale = 1 
          ):

    """
    Generates realizations of the multispecies, stochasic competitve Lokta-Volterra Model 
    of a population of individuals of M different species 
    that replicate and compete
    
    The parameter Omega is the system size
    Parameters
    ----------
    T : float
        Time interval.
    x0: numpy.ndarray of shape (``M``,)
        Initial state, depending on the state_variable keyword (below) this is abundance,
        density, fraction, or capacitance
    alpha : numpy.ndarray of shape (``N``,)
        Replication rates of species
    beta : numpy.ndarray of shape (``N``,``N``)
        Competition rate between species
    system_size : int or inf
        System size, i.e. a measure for the approximate total abundance of species,
        it's the normalization of the competitive rate that makes sure beta is of the order of unity
        If inf deterministic solution is computed, required additional keyword dt to be set
    diffusion_approximation: True/False, default is False
        If True computes the realizations of the diffusion approximation process for the system
        requires to set keyword dt
    normalize : boolean, default = False
        If True normalizes state variable with system_size parameter Omega
    samples : int, default = 1
        The number of samples generated, ignored if system_size = inf (deterministic system)
    dt: float
        Needs to be specified if the diffusion approximation is used. This is the time increment in that
        case

    Returns
    -------
    result : numpy.ndarray:
        each array element is a sample, each sample is a tuple (t,X) or time array t and array of state vectors X.
    

    """ 
    if isinf(system_size):
        t = np.linspace(0,T,int(T/dt))
        ode1 = lambda x,t,a,b : x*(a-b@x)
        sol = odeint(lambda x,t,a,b : x*(a-b@x),x0,t,args=(alpha,beta))                    
        X = [tuple(v) for v in sol]
        return (t,X)
    
    if normalize:
        n0 = system_size*x0
        n0 = n0.astype(int)
    else:
        n0 = x0
                
    if diffusion_approximation:
        res = [clv_diffusion_approximation(n0,T,alpha,beta,system_size,dt) for i in range(samples)]
    else:
        res = [clv_particle_dynamics(n0,T,alpha,beta,system_size, dt, sigma, A, timescale) for i in range(samples)]
    
    if normalize:
        for i in range(len(res)):
            res[i]=(res[i][0],[tuple(np.array(v)/system_size) for v in res[i][1]])
        
    return res

