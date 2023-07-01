#!/bin/bash
#SBATCH --account=cyberwksp21
#SBATCH --partition=valhalla  --qos=valhalla
#SBATCH --clusters=faculty
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
module purge
mpirun -n 16 nwchem benzene_inp.nw
