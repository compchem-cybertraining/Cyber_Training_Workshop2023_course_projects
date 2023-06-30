import os
import sys
import libra_py.packages.cp2k.methods as CP2K_methods
from libra_py.workflows.nbra import step2


path = os.getcwd()
params = {}

# number of processors
params['nprocs'] = 1

# The mpi executable
# The executable can be srun, mpiexe, mpirun, etc. If you're using slurm for job submission and
# you want to use Intel compiled version of CP2K, use 'srun' as your executable. Since we are using GNU-based
# executable, we use the OpenMP 'mpirun'
params['mpi_executable'] = 'srun shifter --entrypoint'

# CP2K executable 
# You can choose any of these compiled versions but please note that you should first load all the required 
# dependencies in the "submit" file and not here. Also, note that the Intel compiled version can obly run on specific nodes on UB CCR.
# Other users need to adapt this based on their own environments.

# CP2K v23.1 compiled with GNU compilers v11.2.0 + Intel MKL v2020.2 + OpenMP v4.1.1, Runs on all general compte and faculty cluster nodes. Other dependencies for this compiled version are:
# DFLAGS = -D__parallel  -D__MKL -D__FFTW3  -D__SCALAPACK -D__FFTW3  -D__LIBINT -D__LIBXC -D__HAS_smm_dnn -D__COSMA -D__ELPA  -D__QUIP -D__GSL -D__PLUMED2 -D__HDF5 -D__LIBVDWXC -D__SPGLIB -D__LIBVORI -D__SPFFT -D__SPLA
params['cp2k_exe'] = 'cp2k'

# Leave this part for Libra
params['istep'] = 
params['fstep'] = 

# Lowest and highest orbital, Here HOMO is 24
params['lowest_orbital'] = 166 - 20
params['highest_orbital'] = 167 + 20

# extended tight-binding calculation type
params['isxTB'] = False
# unrestricted spin configuration
params['isUKS'] = True
# Periodic calculations flag
params['is_periodic'] = True
# Set the cell parameters for periodic calculations
if params['is_periodic']:
    params['A_cell_vector'] = [8.7373056412, 0.0000000000, 0.0000000000]
    params['B_cell_vector'] = [0.0000000000, 10.0889711380, 0.0000000000]
    params['C_cell_vector'] = [0.0000000000, 0.0000000000, 21.1782078743 ]
    params['periodicity_type'] = 'XYZ'
    # Set the origin for generating the translational vectors (for creating Bloch type functions)
    origin = [0,0,0]
    params['translational_vectors'] = CP2K_methods.generate_translational_vectors(origin, [2,2,2],
                                                                                  params['periodicity_type'])
    tr_vecs = params['translational_vectors']
    
    print('The translational vectors for the current periodic system are:\n')
    print(tr_vecs)
    print(F'Will compute the S^AO between R(0,0,0) and {tr_vecs.shape[0]+1} translational vectors')

# The AO overlaps in spherical or Cartesian coordinates
params['is_spherical'] =  True
# Remove the molden files, which are large files for some systems, 
# after the computaion is done for tha system
params['remove_molden'] = True

# Cube visualization using VMD
# For the TiO2 unit cell we do not visualize the cube files, 
params['cube_visualization'] = False

# The results are stored in this folder
params['res_dir'] = path + '/../res'
params['all_pdosfiles'] = path + '/../all_pdosfiles'
params['all_logfiles'] = path + '/../all_logfiles'


params['cp2k_ot_input_template'] = path + '/../es_ot_temp.inp'
params['cp2k_diag_input_template'] = path + '/../es_diag_temp.inp'

# The trajectory xyz file path
# Note that since it will be run in one of the jobs folders
# we need to put one more .. so that it can recognize the file
params['trajectory_xyz_filename'] = path + '/../traj.xyz'

params['restart_file'] = path + '/../restart.wfn'

step2.run_cp2k_libint_step2(params)

