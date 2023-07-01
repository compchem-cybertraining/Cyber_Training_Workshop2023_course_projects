import cmath
import math
import os
import h5py

import numpy as np
import matplotlib.pyplot as plt

from liblibra_core import *

import util.libutil as comn
from libra_py import units
import libra_py.dynamics.heom.compute as compute
from libra_py import ft


plt.rc('axes', titlesize=38)      # fontsize of the axes title
plt.rc('axes', labelsize=38)      # fontsize of the x and y labels
plt.rc('legend', fontsize=38)     # legend fontsize
plt.rc('xtick', labelsize=38)    # fontsize of the tick labels
plt.rc('ytick', labelsize=38)    # fontsize of the tick labels

plt.rc('figure.subplot', left=0.2)
plt.rc('figure.subplot', right=0.95)
plt.rc('figure.subplot', bottom=0.13)
plt.rc('figure.subplot', top=0.88)

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

time, pop0, pop1, coherence01 = None, None, None, None

with h5py.File(F"out_0/mem_data.hdf", 'r') as f:
    time = list(f["time/data"][:] * units.au2fs)
    pop0 = list(f["denmat/data"][:, 0,0])
    pop1 = list(f["denmat/data"][:, 1,1])
    coherence01 = list(f["denmat/data"][:, 0,1])

plt.figure(1, figsize=(24, 12), dpi=300, frameon=False)
plt.subplot(1,1,1)
plt.title('Density matrix evolution', fontsize=44)
plt.xlabel('Time, fs')
plt.ylabel('Population')
ax = plt.gca()
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(length=8, width=3)
plt.plot(time, pop0, label='$\\rho_{00}$', linewidth=10, color = colors["11"])
#plt.plot(time, pop1, label='$\\rho_{11}$', linewidth=10, color = colors["21"])
#plt.plot(time, coherence01, label='$\\rho_{01}$', linewidth=10, color = colors["41"])
plt.legend()
plt.savefig('figure')
