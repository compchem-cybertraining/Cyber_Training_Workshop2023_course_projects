import os
import sys
import libra_py.packages.cp2k.methods as CP2K_methods


run_slurm = True
submit_template = 'submit_template.slm'
run_python_file = 'run_template.py'
istep = 195
fstep = 301
njobs = 1
submission_exe = 'sbatch'

print('Distributing jobs...')
CP2K_methods.distribute_cp2k_libint_jobs(submit_template, run_python_file, istep, fstep, njobs, run_slurm, submission_exe)

