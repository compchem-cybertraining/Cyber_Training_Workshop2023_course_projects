import matplotlib.pyplot as plt
import numpy as np

import potentials as pot
import sup_func as sp


#####################################
############Define System############
#####################################

#number of adiabatic states
nstates = 2

#initial adiabatic state
istate = 0

#dimension of system (1D,2D, or 3D)
dim = 1

#number of trajectories
ntraj = 1000

#time step
dt = 20 #in au
nsteps = 300
time = np.arange(nsteps)*0.5 #time in fs

#nuclear initialization
m = 2000
qi = -15
pi = 10.95  #20
gamma = 0.5  #0.1

#Adiabatic States
V1 = pot.tully1g_adi
V2 = pot.tully1e_adi

#diabatic state function
V_dia = pot.tully1_dia  #tully1_dia

#forces on adiabatic states
dV1 = pot.tully1g_grad_adi
dV2 = pot.tully1e_grad_adi

#random seed
np.random.seed(2)

#####################################
##########Initialize System##########
#####################################

q_final = []
p_final = []

for l in range(ntraj):
    ecoeffs, q, p = sp.init_system(istate, nstates, qi, pi, gamma)
    print(l)
    active_state = istate
    for k in range(nsteps):
        ecoeffs, q, p, c_adi = sp.step_full(ecoeffs, q, p, V_dia, [dV1, dV2], active_state, dt, m) #, c_adi
        p, active_state = sp.hop(c_adi, q, p, [V1, V2], active_state, m)

    q_final.append(q)
    p_final.append(p)

plt.hist(q_final,bins=40,density=True)
plt.title('Position Distribution')
plt.show()

plt.hist(p_final,bins=40,density=True)
plt.title('Momentum Distribution')
plt.show()
