import cmath
import math
import os
import h5py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from liblibra_core import *

import util.libutil as comn
from libra_py import units
import libra_py.dynamics.heom.compute as compute
from libra_py import ft

time, pop0_0, pop0_5, pop0_10, pop0_15, pop0_20 = None, None, None, None, None, None

with h5py.File(F"E0_0.0/out_0/mem_data.hdf", 'r') as f:
    time = list(f["time/data"][:] * units.au2fs)
    pop0_0 = list(f["denmat/data"][:, 0,0])

with h5py.File(F"E0_5.0/out_0/mem_data.hdf", 'r') as f:
    pop0_5 = list(f["denmat/data"][:, 0,0])
 
with h5py.File(F"E0_10.0/out_0/mem_data.hdf", 'r') as f:
    pop0_10 = list(f["denmat/data"][:, 0,0])
 
with h5py.File(F"E0_15.0/out_0/mem_data.hdf", 'r') as f:
    pop0_15 = list(f["denmat/data"][:, 0,0])
    
with h5py.File(F"E0_20.0/out_0/mem_data.hdf", 'r') as f:
    pop0_20 = list(f["denmat/data"][:, 0,0])

figure = plt.figure(figsize = (12,6))
mpl.rcParams['axes.linewidth'] = 1.5
plt.rcParams.update({'font.size': 17})
#plt.title('Plot Title',fontsize = 20)
plt.xlabel('Time, fs', fontsize = 20)
plt.ylabel('Population', fontsize = 20)
plt.plot(time, pop0_0,'-',linewidth = '2', label = r'E$_0$ = 0 * $\beta$', color = 'red')
plt.plot(time, pop0_5,'-',linewidth = '2', label = r'E$_0$ = 5 * $\beta$', color = 'darkorange')
plt.plot(time, pop0_10,'-',linewidth = '2', label = r'E$_0$ = 10 * $\beta$', color = 'green')
plt.plot(time, pop0_15,'-',linewidth = '2', label = r'E$_0$ = 15 * $\beta$', color = 'blue')
plt.plot(time, pop0_20,'-',linewidth = '2', label = r'E$_0$ = 20 * $\beta$', color = 'darkviolet')

plt.gca().spines['right'].set_color('none'), plt.gca().spines['top'].set_color('none')
plt.tick_params(axis='both', which='major', labelsize = 18, size = 7, width = 2)
plt.legend()
plt.savefig('fig2')

