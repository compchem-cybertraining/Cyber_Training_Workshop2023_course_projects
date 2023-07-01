#!/bin/bash
#SBATCH --account=cyberwksp21
#SBATCH --partition=valhalla  --qos=valhalla
#SBATCH --clusters=faculty
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH -C CPU-E5-2650v4
module purge
eval "$(/projects/academic/cyberwksp21/Software/nwchem_conda0/bin/conda shell.bash hook)"
mpirun -n ${SLURM_NTASKS} nwchem box.inp > prep.out
