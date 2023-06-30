import sys
import cmath
import math
import os
import h5py
import matplotlib.pyplot as plt   # plots
import numpy as np
import time
import warnings

from liblibra_core import *
import util.libutil as comn
from libra_py import units
import libra_py.models.Holstein as Holstein
import libra_py.models.Morse as Morse
from libra_py import dynamics_plotting
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot

import libra_py.dynamics.exact.compute as dvr
import libra_py.dynamics.exact.save as dvr_save

import libra_py.data_savers as data_savers


#from matplotlib.mlab import griddata
#%matplotlib inline 
warnings.filterwarnings('ignore')

colors = {}
colors.update({"11": "#8b1a0e"})  # red       
colors.update({"12": "#FF4500"})  # orangered 
colors.update({"13": "#B22222"})  # firebrick 
colors.update({"14": "#DC143C"})  # crimson   
colors.update({"21": "#5e9c36"})  # green
colors.update({"22": "#006400"})  # darkgreen  
colors.update({"23": "#228B22"})  # forestgreen
colors.update({"24": "#808000"})  # olive      
colors.update({"31": "#8A2BE2"})  # blueviolet
colors.update({"32": "#00008B"})  # darkblue  
colors.update({"41": "#2F4F4F"})  # darkslategray

clrs_index = ["11", "21", "31", "41", "12", "22", "32", "13","23", "14", "24"]





def compute_model(q, params, full_id):

    model = params["model"]
    res = None
    
    if model==1:        
        res = Holstein.Holstein2(q, params, full_id) 
    elif model==2:
        pass #res = compute_model_nbra(q, params, full_id)
    elif model==3:
        pass #res = compute_model_nbra_files(q, params, full_id)        
    elif model==4:
        res = Morse.general(q, params, full_id)    

    return res

def potential(q, params):
    full_id = Py2Cpp_int([0,0]) 
    
    return compute_model(q, params, full_id)




model_params1 = {"model":1, "model0":1, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.000, "nstates":2}
model_params2 = {"model":1, "model0":1, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.001, "nstates":2}
model_params3 = {"model":1, "model0":1, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.01, "nstates":2}
model_params4 = {"model":1, "model0":1, "E_n":[0.0, -0.01], "x_n":[0.0,  0.5],"k_n":[0.002, 0.008],"V":0.001, "nstates":2}

all_model_params = [model_params1, model_params2, model_params3, model_params4]




#################################
# Give the model used an index
model_indx = 3
model_params = all_model_params[model_indx]
#################################



dyn_general = { "nsteps":200, "ntraj":1000, "nstates":2,
                "dt":10.0, "num_electronic_substeps":1, "isNBRA":0, "is_nbra":0,
                "progress_frequency":0.1, "which_adi_states":range(2), "which_dia_states":range(2),      
                "mem_output_level":4,
                "properties_to_save":[ "timestep", "time", "q", "p", "f", "Cadi", "Cdia", "Epot_ave", "Ekin_ave", "Etot_ave",
                "se_pop_adi", "se_pop_dia", "sh_pop_adi", "hvib_adi", "hvib_dia", "St", "basis_transform", "D_adi" ],
                "prefix":"adiabatic_md", "prefix2":"adiabatic_md"
              }



# Uncomment one of the options in each of the categories below:
#====== How to update Hamiltonian ===================
#dyn_general.update({"ham_update_method":0}) # don't update any Hamiltonians
dyn_general.update({"ham_update_method":1})  # recompute only diabatic Hamiltonian, common choice for model Hamiltonians 
#dyn_general.update({"ham_update_method":2})  # recompute only adiabatic Hamiltonian; use with file-based or on-the-fly workflows


#====== How to transform the Hamiltonians between representations ============
#dyn_general.update( {"ham_transform_method":0 }) # don't do any transforms; usually for NBRA or on-the-fly workflows, 
                                                 # so you don't override the read values
dyn_general.update( {"ham_transform_method":1 }) # diabatic->adiabatic according to internal diagonalization
#dyn_general.update( {"ham_transform_method":2 }) # diabatic->adiabatic according to internally stored basis transformation matrix
#dyn_general.update( {"ham_transform_method":3 }) # adiabatic->diabatic according to internally stored basis transformation matrix
#dyn_general.update( {"ham_transform_method":4 }) # adiabatic->diabatic according to local diabatization method

#====== How do get the time-overlaps in the dynamics ========
#dyn_general.update( {"time_overlap_method":0 })  # don't update time-overlaps - maybe they are already pre-computed and read
dyn_general.update( {"time_overlap_method":1 }) # explicitly compute it from the wavefunction info; common for model Hamiltonians

#================== How to compute NACs ===============================
dyn_general.update({"nac_update_method":1})  # explicit NAC calculations - let's just focus on this one for now
#dyn_general.update({"nac_update_method":2, "nac_algo":0})  # HST algo
#dyn_general.update({"nac_update_method":2, "nac_algo":1})  # NPI algo

#============== How to compute vibronic Hamiltonian ==============
#dyn_general.update( {"hvib_update_method":0 }) # don't update Hvib; maybe because we read it from files
dyn_general.update( {"hvib_update_method":1 }) # recompute diabatic and adiabatic Hvib from the Ham and NACs in those reps

#=========== Ehrenfest or state-resolved options ===========
# This is what we use with any of the TSH-based methods - in all cases here, we would 
# use "rep_force":1 so that we are guided by the forces derived from the adiabatic surfaces.
# In Ehrenfest cases though, the forces can be computed using only diabatic properties though 
dyn_general.update( {"force_method":1, "rep_force":1} ) # state-resolved (e.g. TSH) with adiabatic forces
#dyn_general.update( {"force_method":2, "rep_force":1} ) # for Ehrenfest in adiabatic rep
#dyn_general.update( {"force_method":2, "rep_force":0} ) # for Ehrenfest in diabatic rep


#============ Types of surface hopping acceptance and momenta rescaling opntions =================
#dyn_general.update({"hop_acceptance_algo":10, "momenta_rescaling_algo":100 })  # accept and rescale based on total energy, do not reverse on frustrated
dyn_general.update({"hop_acceptance_algo":10, "momenta_rescaling_algo":101 })  # accept and rescale based on total energy, reverse on frustrated
#dyn_general.update({"hop_acceptance_algo":20, "momenta_rescaling_algo":200 })  # accept and rescale based on NAC vectors, do not reverse on frustrated
#dyn_general.update({"hop_acceptance_algo":20, "momenta_rescaling_algo":201 })  # accept and rescale based on NAC vectors, reverse on frustrated
#dyn_general.update({"hop_acceptance_algo":21, "momenta_rescaling_algo":200 })  # accept and rescale based on force differences, do not reverse on frustrated
#dyn_general.update({"hop_acceptance_algo":21, "momenta_rescaling_algo":201 })  # accept and rescale based on force differences, reverse on frustrated


#============ Surface hopping opntions =================
#dyn_general.update({"tsh_method":-1 }) # adiabatic, no surface hopping
#dyn_general.update({"tsh_method":10 }) # FSSH_var
#dyn_general.update({"tsh_method":0 }) # FSSH
dyn_general.update({"tsh_method":1 }) # GFSH
#dyn_general.update({"tsh_method":2 }) # MSSH
#dyn_general.update({"tsh_method":3, "rep_lz":0 })  # LZ options
#dyn_general.update({"tsh_method":4, "rep_lz":0 }) # ZN
#dyn_general.update({"tsh_method":5 }) # DISH

#=========== Decoherence options =================
dyn_general.update({ "decoherence_algo":-1}) # no (additional) decoherence
#dyn_general.update({ "decoherence_algo":0}) # msdm  
#dyn_general.update({ "decoherence_algo":1}) # IDA
#dyn_general.update({ "decoherence_algo":2}) # A-FSSH, not yet ready
#dyn_general.update({ "decoherence_algo":3}) # BCSH
#dyn_general.update({ "decoherence_algo":4}) # mfsd

#=========== Decoherence times (for decoherence options 0 or 4) ==================
A = MATRIX(2,2); A.set(0, 0, 10.0); A.set(1,1, 10.0)
dyn_general.update({"decoherence_times_type":-1 }) # No decoherence times, infinite decoherence times
#dyn_general.update( { "decoherence_times_type":1, "decoherence_C_param": 1.0, "decoherence_eps_param":0.1 } )  # EDC + default params
#dyn_general.update( { "decoherence_times_type":2, "schwartz_decoherence_inv_alpha":A } ) # Schwartz version 1
#dyn_general.update( { "decoherence_times_type":3, "schwartz_decoherence_inv_alpha":A } ) # Schwartz version 2

#======= Various decoherence-related parameters =====================
dyn_general.update( {"dephasing_informed":0, "decoherence_rates":MATRIX(2,2), "ave_gaps":MATRIX(2,2) } )
                               
#=========== Phase correction of SSY =================
dyn_general.update({"do_ssy":1 }) # do no use it - that's the default

#=========== What to integrate ==================
# solve TD-SE in diabatic representation
#dyn_general.update({"rep_tdse":0, "electronic_integrator":-1 })   # no propagation
#dyn_general.update({"rep_tdse":0, "electronic_integrator":0 })    # Lowdin exp_ with 2-point Hvib_dia
#dyn_general.update({"rep_tdse":0, "electronic_integrator":1 })    # based on QTAG propagator
#dyn_general.update({"rep_tdse":0, "electronic_integrator":2 })    # based on modified QTAG propagator (Z at two times)
#dyn_general.update({"rep_tdse":0, "electronic_integrator":3 })    # non-Hermitian integrator with 2-point Hvib_dia

# solve TD-SE in adiabatic representation
#dyn_general.update({"rep_tdse":1, "electronic_integrator":-1 })  # no propagation
#dyn_general.update({"rep_tdse":1, "electronic_integrator":0 })   # ld, with crude splitting,  with exp_
#dyn_general.update({"rep_tdse":1, "electronic_integrator":1 })   # ld, with symmetric splitting, with exp_
dyn_general.update({"rep_tdse":1, "electronic_integrator":2 })   # ld, original, with exp_
#dyn_general.update({"rep_tdse":1, "electronic_integrator":3 })   # 1-point, Hvib integration, with exp_
#dyn_general.update({"rep_tdse":1, "electronic_integrator":4 })   # 2-points, Hvib integration, with exp_
#dyn_general.update({"rep_tdse":1, "electronic_integrator":5 })   # 2-points, Hvib, integration with the second-point correction of Hvib, with exp_
#dyn_general.update({"rep_tdse":1, "electronic_integrator":6 })   # same as 4, but without projection matrices (T_new = I)
#dyn_general.update({"rep_tdse":1, "electronic_integrator":10 })  # same as 0, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":11 })  # same as 1, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":12 })  # same as 2, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":13 })  # same as 3, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":14 })  # same as 4, but with rotations
#dyn_general.update({"rep_tdse":1, "electronic_integrator":15 })  # same as 5, but with rotations

# solve QCLE in diabatic representation
#dyn_general.update({"rep_tdse":3, "electronic_integrator":0 })  # mid-point Hvib, using exp_

# solve QCLE in adiabatic representation
#dyn_general.update({"rep_tdse":3, "electronic_integrator":0 })  # mid-point Ham with the second-point correction of Hvib, using exp_
#dyn_general.update({"rep_tdse":3, "electronic_integrator":1 })  # using Zhu Liouvillian THIS IS NOT JUST A DIFFERENT INTEGRATOR!!!!
#dyn_general.update({"rep_tdse":3, "electronic_integrator":10 }) # same as 0 but with rotations


#################################
# Give the recipe above an index
method_indx = 0
#################################





# ============== How nuclear DOFs are initialized =================
#icond_nucl = 0  # Coords and momenta are set exactly to the given value
#icond_nucl = 1  # Coords are set, momenta are sampled
#icond_nucl = 2  # Coords are sampled, momenta are set
icond_nucl = 3  # Both coords and momenta are samples

nucl_params = { "ndof":1, "q":[-2.0], "p":[0.0], 
                "mass":[2000.0], "force_constant":[0.01], 
                "init_type":icond_nucl }

#============= How electronic DOFs are initialized ==================
#icond_elec = 2  # amplitudes all have the same phase
icond_elec = 3  # amplitudes gain random phase 

#============= Also select the representation ========================
# rep = 0 # diabatic wfc
rep = 1 # adiabatic wfc

elec_params = {"verbosity":2, "init_dm_type":0,
               "ndia":2, "nadi":2, 
               "rep":rep, "init_type":icond_elec, "istates":[0.0, 1.0]
              }


#============== This computes the enumerated index of the initial condition =============
icond_indx = rep * 8 + (icond_elec-2)*4 + icond_nucl





dyn_params = dict(dyn_general)
dyn_params.update({ "prefix":F"model{model_indx}-method{method_indx}-icond{icond_indx}", 
                    "prefix2":F"model{model_indx}-method{method_indx}-icond{icond_indx}" })

print(F"Computing model{model_indx}-method{method_indx}-icond{icond_indx}")    

rnd = Random()
res = tsh_dynamics.generic_recipe(dyn_params, compute_model, model_params, elec_params, nucl_params, rnd)





#============ Plotting ==================
#method_indx = 0
#model_indx = 1
#init_cond_indx = 6

pref = F"model{model_indx}-method{method_indx}-icond{icond_indx}"

plot_params = { "prefix":pref, "filename":"mem_data.hdf", "output_level":3,
                "which_trajectories":[0, 1, 2], "which_dofs":[0], "which_adi_states":[0,1], "which_dia_states":[0,1], 
                "frameon":True, "linewidth":3, "dpi":300,
                "axes_label_fontsize":(8,8), "legend_fontsize":8, "axes_fontsize":(8,8), "title_fontsize":8,
                "what_to_plot":["coordinates", "momenta",  "forces", "energies", "phase_space", "se_pop_adi",
                                "se_pop_dia", "sh_pop_adi", "traj_resolved_adiabatic_ham", "traj_resolved_diabatic_ham", 
                                "time_overlaps", "basis_transform"
                               ], 
                "which_energies":["potential", "kinetic", "total"],
                "save_figures":1, "do_show":1
              }

tsh_dynamics_plot.plot_dynamics(plot_params)


