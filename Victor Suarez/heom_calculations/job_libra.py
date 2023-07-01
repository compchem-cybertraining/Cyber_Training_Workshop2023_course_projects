import libra_py.data_savers as data_savers

import cmath
import math
import os
import sys
import copy
import time
import h5py

import numpy as np
import matplotlib.pyplot as plt

from liblibra_core import *

import util.libutil as comn
from libra_py import units
import libra_py.dynamics.heom.compute as compute
from libra_py import ft



#===================== HEOM output ====================

def init_heom_data(saver, hdf5_output_level, _nsteps, _nquant):

    if hdf5_output_level>=1:
        # Time axis (integer steps)
        saver.add_dataset("timestep", (_nsteps,) , "I")  

        # Time axis
        saver.add_dataset("time", (_nsteps,) , "R")  
        

    if hdf5_output_level>=3:
        # System's density matrix
        saver.add_dataset("denmat", (_nsteps, _nquant, _nquant), "C") 






def init_heom_savers(params, nquant):

    #================ Create savers ==================    
    prefix = params["prefix"]

    # Create an output directory, if not present    
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    properties_to_save = params["properties_to_save"]


    _savers = {"hdf5_saver":None, "txt_saver":None, "mem_saver":None }

    #====== HDF5 ========
    hdf5_output_level = params["hdf5_output_level"]
    
    if hdf5_output_level > 0:                
        _savers["hdf5_saver"] = data_savers.hdf5_saver(F"{prefix}/data.hdf", properties_to_save) 
        _savers["hdf5_saver"].set_compression_level(params["use_compression"], params["compression_level"])
        init_heom_data(_savers["hdf5_saver"], hdf5_output_level, params["nsteps"], nquant)

    #====== TXT ========
    if params["txt_output_level"] > 0:
        pass
    
    #====== MEM =========
    mem_output_level = params["mem_output_level"]

    if mem_output_level > 0:
        _savers["mem_saver"] =  data_savers.mem_saver(properties_to_save)
        init_heom_data(_savers["mem_saver"], mem_output_level, params["nsteps"], nquant)

    return _savers                         
    





def save_heom_hdf5(step, saver, params, denmat):
    
    dt = params["dt"]    
    hdf5_output_level = params["hdf5_output_level"]

    
    if hdf5_output_level>=1:
        # Timestep 
        saver.save_scalar(step, "timestep", step) 

        # Actual time
        saver.save_scalar(step, "time", step * dt)        

    if hdf5_output_level>=3:
        # Average adiabatic density matrices
        saver.save_matrix(step, "denmat", denmat) 



def save_heom_data(_savers, step, print_freq, params, rho_unpacked):

    #================ Saving the data ==================

    #if step%print_freq==0:
    #    print(F" step= {step}")
        
    # Save properties
    if _savers["hdf5_saver"] != None:            
        save_heom_hdf5(step, _savers["hdf5_saver"], params, rho_unpacked[0])
        
    if _savers["txt_saver"] != None:            
        pass
        
    if _savers["mem_saver"] != None:            
        prms = dict(params)
        prms["hdf5_output_level"] = prms["mem_output_level"]
        save_heom_hdf5(step, _savers["mem_saver"], prms, rho_unpacked[0])



def aux_print_matrices(step, x):
    print(F"= step = {step} =")
    nmat = len(x)
    nrows = x[0].num_of_rows
    ncols = x[0].num_of_cols
    for imat in range(nmat):
        print(F"== imat = {imat} ==")
        for row in range(nrows):
            line = ""
            for col in range(ncols):
                line = line + F"{x[imat].get(row, col)}  "
            print(line)


def update_filters(rho_scaled, params, aux_memory):
    """

    This function takes the current ADMs hierarchy, computes the corresponding time-derivatives
    (based on the input params["adm_list"]), then it updates the params["adm_list"] to indicate
    the equations for which the |dADM/dt (max element)| is larger than a given threshold,
    params["adm_deriv_tolerance"].

    This function also runs over all ADMs to determine those which have |ADM (max element)| larger
    than the params["adm_tolerance"]. This will determine the update of the params["nonzero"] lists.
    The ADMs that are determined to be "zero" (discarded until later) can also be set to 0.0, the
    option controlled by params["do_zeroing"].
    In the end, the variable `rho` is updated with the correspondingly updated variables

    aux_memory should have allocated:
    - rho_unpacked_scaled
    - drho_unpacked_scaled

    """


    unpack_mtx(aux_memory["rho_unpacked_scaled"], rho_scaled)

    drho_scaled = compute_heom_derivatives(rho_scaled, params)
    unpack_mtx(aux_memory["drho_unpacked_scaled"], drho_scaled)

    # Filtering of the derivatives - defines the active EOMs, params["adm_list"]
    trash = filter(aux_memory["drho_unpacked_scaled"], params["adm_list"], params["adm_deriv_tolerance"], 0)

    # Filtering of the densities - defines the list of nonzero derivatives, params["nonzero"]
    params["nonzero"] = filter(aux_memory["rho_unpacked_scaled"], trash, params["adm_tolerance"], params["do_zeroing"])

    pack_mtx(aux_memory["rho_unpacked_scaled"], rho_scaled)



def transform_adm(rho, rho_scaled, aux_memory, params, direction):
    """
    aux_memory should have allocated:
    - rho_unpacked
    - rho_unpacked_scaled

    direction:
     1 :  raw -> scaled
    -1 :  scaled -> raw

    """

    nn_tot = len(aux_memory["rho_unpacked"])

    # Forward
    if direction == 1:

        unpack_mtx(aux_memory["rho_unpacked"], rho)

        if params["do_scale"]==0:
            for n in range(nn_tot):
                aux_memory["rho_unpacked_scaled"][n] = CMATRIX(aux_memory["rho_unpacked"][n])

        elif params["do_scale"]==1:
            scale_rho(aux_memory["rho_unpacked"], aux_memory["rho_unpacked_scaled"], params)

        pack_mtx(aux_memory["rho_unpacked_scaled"], rho_scaled)


    # Backward
    elif direction == -1:

        unpack_mtx(aux_memory["rho_unpacked_scaled"], rho_scaled)

        # We want to save only actual ADMs so we need to convert the
        # scaled one back to the unscaled
        if params["do_scale"]==0:
            for n in range(nn_tot):
                aux_memory["rho_unpacked"][n] = CMATRIX(aux_memory["rho_unpacked_scaled"][n])

        elif params["do_scale"]==1:
            # rho_unpacked_scaled -> rho_unpacked
            inv_scale_rho(aux_memory["rho_unpacked"], aux_memory["rho_unpacked_scaled"], params)

        pack_mtx(aux_memory["rho_unpacked"], rho)



T = 300.
beta = 1 / (T * 3.16681156345e-6)

V = 0.001 / beta
E0 = E0_sub / beta
lam = 10. / beta
gam = 0.1 / beta

print(2 * 3.1415 * V**2 * (beta / (4 * 3.1415 * lam))**0.5 * np.exp(-beta * (lam - E0)**2 / (4 * lam)))

delt = delt_sub
nsteps = nsteps_sub
step_switch = step_switch_sub

# Hamiltonian
Ham1 = CMATRIX(2,2)
Ham1.set(0, 0, 0.5 * E0);  Ham1.set(0, 1, 0.);
Ham1.set(1, 0, 0.);   Ham1.set(1, 1, -0.5 * E0)
Ham1.scale(-1,-1, (1.0+0.0j))

Ham2 = CMATRIX(2,2)
Ham2.set(0, 0, 0.5 * E0);  Ham2.set(0, 1, V);
Ham2.set(1, 0, V);   Ham2.set(1, 1, -0.5 * E0)
Ham2.scale(-1,-1, (1.0+0.0j))

# Initial density matrix
rho_init = CMATRIX(2,2); rho_init.set(0, 0, 1.0+0.0j) # starting state = initial state

# Parameters
dyn_params = { # hierarchy parameters
           "KK":K_sub, "LL":L_sub,

          # bath parameters
           "gamma": gam,
           "eta": 0.5 * lam,
           "temperature": T,
           "el_phon_couplings":initialize_el_phonon_couplings(2),

          # dynamics parameters
           "dt":delt, "nsteps":nsteps,
           "verbosity":-1, "progress_frequency":0.1,

          # computational efficiency parameters
           "truncation_scheme":4, "do_scale":0,
           "adm_tolerance":1e-10, "adm_deriv_tolerance":1e-15,
           "filter_after_steps":1,"do_zeroing":1,
           "num_threads":1,

          # data management parameters
           "prefix":"out_0",
           "hdf5_output_level":0, "txt_output_level":0, "mem_output_level":3,
           "properties_to_save": [ "timestep", "time", "denmat"],
           "use_compression":0, "compression_level":[0,0,0]
             }

# Run the actual calculations
#compute.run_dynamics(dyn_params, Ham, rho_init)

params = dict(dyn_params)

# Parameters and dimensions
critical_params = [  ]
default_params = { "KK":0, "LL":10,
                   "gamma": 1.0/(0.1 * units.ps2au),
                   "eta": 2.0 * 50.0 * units.inv_cm2Ha,
                   "temperature": 300.0,
                   "el_phon_couplings":initialize_el_phonon_couplings(Ham1.num_of_cols),

                   "dt":0.1*units.fs2au, "nsteps":10,
                   "verbosity":-1, "progress_frequency":0.1,

                   "truncation_scheme":1, "do_scale":0,
                   "adm_tolerance":1e-6,  "adm_deriv_tolerance":1e-12,
                   "filter_after_steps":1, "do_zeroing":0,
                   "num_threads":1,

                   "prefix":"out",
                   "hdf5_output_level":0, "txt_output_level":0, "mem_output_level":3,
                   "properties_to_save": [ "timestep", "time", "denmat"],
                   "use_compression":0, "compression_level":[0,0,0]
                 }

comn.check_input(params, default_params, critical_params)

nsteps = params["nsteps"]
print_freq = int(params["progress_frequency"]*nsteps)


#============= System ======================
params.update({"Ham" : Ham1})
nquant = Ham1.num_of_cols


#============== HEOM topology ==============


KK = dyn_params["KK"]
LL = dyn_params["LL"]

all_vectors = intList2()
vec_plus = intList2()
vec_minus = intList2()

gen_hierarchy(nquant * (KK+1), LL, params["verbosity"], all_vectors, vec_plus, vec_minus)
params.update( { "nvec":all_vectors, "nvec_plus":vec_plus, "nvec_minus":vec_minus } )

nn_tot = len(all_vectors)


all_indices = []
init_nonzero = []
for n in range(nn_tot):
    all_indices.append(n)
    init_nonzero.append(1)


#============ Bath update =====================
gamma_matsubara = doubleList()
c_matsubara = complexList()

setup_bath(KK, params["eta"], params["gamma"], params["temperature"], gamma_matsubara, c_matsubara)
params.update({ "gamma_matsubara": gamma_matsubara, "c_matsubara":c_matsubara  } )

if params["verbosity"]>=1:
    for k in range(KK+1):
        print(F" k = {k} gamma_matsubara[{k}] = {gamma_matsubara[k]}  c_matsubara[{k}] = {c_matsubara[k]}")

#============= Initialization ============

rho = CMATRIX((nn_tot)*nquant, nquant)  # all rho matrices stacked on top of each other
rho_scaled = CMATRIX((nn_tot)*nquant, nquant)  # all rho matrices stacked on top of each other
#drho = CMATRIX((nn_tot)*nquant, nquant)  # all rho matrices stacked on top of each other

aux_memory = {"rho_unpacked" : CMATRIXList(),
              "rho_unpacked_scaled" : CMATRIXList(),
              "drho_unpacked" : CMATRIXList(),
              "drho_unpacked_scaled" : CMATRIXList()
             }
for n in range(nn_tot):
    aux_memory["rho_unpacked"].append( CMATRIX(nquant, nquant))
    aux_memory["rho_unpacked_scaled"].append( CMATRIX(nquant, nquant))
    aux_memory["drho_unpacked"].append( CMATRIX(nquant, nquant))
    aux_memory["drho_unpacked_scaled"].append( CMATRIX(nquant, nquant))

# Initial conditions
x_ = Py2Cpp_int(list(range(nquant)))
y_ = Py2Cpp_int(list(range(nquant)))
push_submatrix(rho, rho_init, x_, y_)

#unpack_mtx(aux_memory["rho_unpacked"], rho)


#========== Scale working ADMs ====================
if params["verbosity"]>=2 and params["do_scale"]==1:
    print("Scaling factors")
    for n in range(nn_tot):
        for m in range(nquant):
            for k in range(KK+1):
                n_mk = all_vectors[n][m*(KK+1)+k]
                scl = 1.0/math.sqrt( FACTORIAL(n_mk) * FAST_POW(abs(c_matsubara[k]), n_mk))
                print(F" n={n} m={m} k={k}  scaling_factor={scl}")

# raw -> scaled
transform_adm(rho, rho_scaled, aux_memory, params, 1)

#========== Filter scaled ADMs ======================
params.update({ "nonzero": Py2Cpp_int(init_nonzero), "adm_list": Py2Cpp_int( all_indices ) } )
update_filters(rho_scaled, params, aux_memory)



if params["verbosity"]>=2:
    print("nonzero = ", Cpp2Py(params["nonzero"]))
    print("adm_list = ", Cpp2Py(params["adm_list"]))

    if params["verbosity"]>=4:
        print("ADMs")
        aux_print_matrices(0, aux_memory["rho_unpacked"])
        print("Scaled ADMs")
        aux_print_matrices(0, aux_memory["rho_unpacked_scaled"])


# Initialize savers
#_savers = save.init_heom_savers(params, nquant)
_savers = init_heom_savers(params, nquant)

#============== Propagation =============


start = time.time()
for step in range(step_switch):

    #================ Saving and printout ===================
    # scaled -> raw
    transform_adm(rho, rho_scaled, aux_memory, params, -1)

    # Save the variables
    #save.save_heom_data(_savers, step, print_freq, params, aux_memory["rho_unpacked"])
    save_heom_data(_savers, step, print_freq, params, aux_memory["rho_unpacked"])

    if step%print_freq==0:
        print(F" step= {step}")

        if params["verbosity"]>=3:
            print("nonzero = ", Cpp2Py(params["nonzero"]))
            print("adm_list = ", Cpp2Py(params["adm_list"]))

        if params["verbosity"]>=4:
            print("ADMs")
            aux_print_matrices(0, aux_memory["rho_unpacked"])
            print("Scaled ADMs")
            aux_print_matrices(0, aux_memory["rho_unpacked_scaled"])




    #============== Update the list of active equations = Filtering  ============
    if step % params["filter_after_steps"] == 0:

        # To assess which equations to discard, lets estimate the time-derivatives of rho
        # for all the matrices
        params["adm_list"] = Py2Cpp_int( all_indices )

        update_filters(rho_scaled, params, aux_memory)


    #================= Propagation for one timestep ==================================
    rho_scaled = RK4(rho_scaled, params["dt"], compute_heom_derivatives, params)

params.update({"Ham" : Ham2})

for step in range(step_switch, params["nsteps"]):

    #================ Saving and printout ===================
    # scaled -> raw
    transform_adm(rho, rho_scaled, aux_memory, params, -1)

    # Save the variables
    #save.save_heom_data(_savers, step, print_freq, params, aux_memory["rho_unpacked"])
    save_heom_data(_savers, step, print_freq, params, aux_memory["rho_unpacked"])

    if step%print_freq==0:
        print(F" step= {step}")

        if params["verbosity"]>=3:
            print("nonzero = ", Cpp2Py(params["nonzero"]))
            print("adm_list = ", Cpp2Py(params["adm_list"]))

        if params["verbosity"]>=4:
            print("ADMs")
            aux_print_matrices(0, aux_memory["rho_unpacked"])
            print("Scaled ADMs")
            aux_print_matrices(0, aux_memory["rho_unpacked_scaled"])




    #============== Update the list of active equations = Filtering  ============
    if step % params["filter_after_steps"] == 0:

        # To assess which equations to discard, lets estimate the time-derivatives of rho
        # for all the matrices
        params["adm_list"] = Py2Cpp_int( all_indices )

        update_filters(rho_scaled, params, aux_memory)


    #================= Propagation for one timestep ==================================
    rho_scaled = RK4(rho_scaled, params["dt"], compute_heom_derivatives, params)

end = time.time()
print(F"Calculations took {end - start} seconds")


# For the mem_saver - store all the results into HDF5 format only at the end of the simulation
#if _savers["mem_saver"] != None:
#    prefix = params["prefix"]
#    _savers["mem_saver"].save_data( F"{prefix}/mem_data.hdf", params["properties_to_save"], "w")
#    return _savers["mem_saver"]

def save_data():
    if _savers["mem_saver"] != None:
        prefix = params["prefix"]
        _savers["mem_saver"].save_data( F"{prefix}/mem_data.hdf", params["properties_to_save"], "w")
        return _savers["mem_saver"]

save_data()

