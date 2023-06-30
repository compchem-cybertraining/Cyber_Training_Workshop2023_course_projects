import os
import time
import numpy as np
from liblibra_core import *
from libra_py import molden_methods, data_conv
import libra_py.packages.cp2k.methods as CP2K_methods


molden_filename = "c6h6-b3lyp.molden" 
is_spherical = True
nprocs = 4

# ===== removing all lines with "Sym="
os.system("sed -i '/Sym=/d' c6h6-b3lyp.molden")
# ===== create integrations shell
t1 = time.time()
print("Creating shell using molden_methods.molden_file_to_libint_shell")
shell_1, l_vals = molden_methods.molden_file_to_libint_shell(molden_filename, is_spherical)
print('Done with creating shell. Elapsed time:', time.time()-t1)
# ===== read eigenvectors
t1 = time.time()
print('Reading energies and eigenvectors....')
eig_vect_1, energies_1 = molden_methods.eigenvectors_molden(molden_filename, nbasis(shell_1),l_vals)
print('Done with reading energies and eigenvectors. Elapsed time:', time.time()-t1)
# ===== compute overlaps
t1 = time.time()
print('Computing atomic orbital overlap matrix...')
AO_S = compute_overlaps(shell_1,shell_1,nprocs)
# ===== resorting eigenvectors
new_indices = CP2K_methods.resort_molog_eigenvectors(l_vals)
eigenvectors_1 = []
for j in range(len(eig_vect_1)):
    # the new and sorted eigenvector
    eigenvector_1 = eig_vect_1[j]
    eigenvector_1 = eigenvector_1[new_indices]
    # append it to the eigenvectors list
    eigenvectors_1.append(eigenvector_1)
eigenvectors_1 = np.array(eigenvectors_1)
# ===== computing the MO overlaps
AO_S = data_conv.MATRIX2nparray(AO_S)
S = np.linalg.multi_dot([eigenvectors_1, AO_S, eigenvectors_1.T])
print("Diagonal elements of the MO overlap matrix")
print(np.diag(S))
print("Done! Elpased time for computing overlaps:", time.time()-t1)
