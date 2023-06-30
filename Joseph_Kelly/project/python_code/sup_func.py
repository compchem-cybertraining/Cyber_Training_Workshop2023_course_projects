import numpy as np

def init_elec(n, N):
    '''
    States are indexed starting from 0
    Args:
        n (int): index of state to initialize population
        N (int): number of electronic states in system
    Returns:
        coeffs (np.array<complex doubles>): list of coefficients for each state
    '''
    Hn = np.sum(1/np.arange(1,N+1))
    alphaN = (N-1)/(Hn-1)
    shift = (alphaN-1)/(N*alphaN)
    coeffs = np.zeros(N)
    coeffs[n] = 1
    coeffs = coeffs/alphaN + shift
    coeffs = np.sqrt(coeffs)*np.exp(1j*2*np.pi*np.random.rand(N))
    return coeffs

def init_nucl_gaus(qi, pi, gamma):
    '''
    Args:
        qi (float): center for gaussian distribution of initial positions
        pi (float): center for gaussian distribution of initial momenta
        gamma (float): width of distribution
    '''
    qstart = np.random.normal(qi,np.sqrt(1/(2*gamma)))
    pstart = np.random.normal(pi,np.sqrt(gamma/2))
    return qstart, pstart

def pop_estimator(coeffs):
    N = len(coeffs)
    Hn = np.sum(1/np.arange(1,N+1))
    alphaN = (N-1)/(Hn-1)
    Ps = coeffs*np.conj(coeffs)
    phis = 1/N+alphaN*(Ps-1/N)
    return phis

def coher_estimator(coeffs):
    return 1

def sqr_mod(coeffs):
    return (coeffs*np.conj(coeffs)).real

def init_system(istate, nstates, qi, pi, gamma):
    ecoeffs = init_elec(istate, nstates)
    qstart, pstart = init_nucl_gaus(qi, pi, gamma)
    return ecoeffs, qstart, pstart

def step_elec(c_dia, q, V_dia, dt):
    nstates = len(c_dia)
    H_dia = np.zeros((nstates,nstates))
    # print(sqr_mod(c_dia))
    for i in range(nstates):
        for j in range(nstates):
            H_dia[i,j] = V_dia(q,i,j)

    evals, evecs = np.linalg.eig(H_dia)
    H_adi = np.diag(np.exp(-1j*evals*dt/2))
    c_dia = evecs @ H_adi @ np.conj(evecs.T) @ c_dia 

    c_adi = np.conj(evecs.T) @ c_dia
    return c_dia, c_adi

def step_full(c_dia, q, p, V_dia, dV_adi, active_state, dt, m):
    c_dia, c_adi = step_elec(c_dia, q, V_dia, dt)
    q = q + dt * p/(2*m)
    p = p - dt *dV_adi[active_state](q)
    q = q + dt * p/(2*m)
    c_dia, c_adi = step_elec(c_dia, q, V_dia, dt) #, c_adi
    return c_dia, q, p, c_adi


def hop(c_adi, q, p, V_adi, active_state, m):
    new_state = np.argmax(pop_estimator(c_adi))
    # print(new_state)
    if new_state != active_state:
        hop_E = V_adi[new_state](q) - V_adi[active_state](q) 
        KE = p**2/(2*m)
        if KE >= hop_E:
            p = np.sqrt(p**2-2*m*hop_E)
            active_state = new_state
        else:
            p = -p
            active_state = active_state
    return p, active_state











