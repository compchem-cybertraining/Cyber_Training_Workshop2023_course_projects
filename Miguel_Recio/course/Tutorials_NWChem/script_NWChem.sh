#!/bin/bash
#SBATCH --partition=valhalla  
#SBATCH ---qos=valhalla
#SBATCH --clusters=faculty
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=24
#SBATCH -C CPU-E5-2650v4
#SBATCH --clusters=faculty
#SBATCH --account=cyberwksp21
eval "$(/projects/academic/cyberwksp21/Software/nwchem_conda0/bin/conda shell.bash hook)"
mpirun -n ${SLURM_NTASKS} nwchem [input]
