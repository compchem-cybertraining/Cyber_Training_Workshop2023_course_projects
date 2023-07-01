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

time, pop0, pop1, coherence01 = None, None, None, None

with h5py.File("out_0/mem_data.hdf", 'r') as f:
    time = list(f["time/data"][:] * units.au2fs)
    pop0 = list(f["denmat/data"][:, 0,0])
    pop1 = list(f["denmat/data"][:, 1,1])
    coherence01 = list(f["denmat/data"][:, 0,1])

pop0_cut = np.array(pop0[int(len(pop0)*0.75):])
dpop0 = (pop0_cut[1:]-pop0_cut[:-1])/(time[1]-time[0])

print(np.average(dpop0) * units.au2fs)
print(np.std(dpop0) * units.au2fs)
