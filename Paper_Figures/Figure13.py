# %% Maximum relative Bz error
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import scipy.constants as constants
import string
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from charges import *
from field_calculations import *

e = constants.e
c = constants.c
width = 3.519
tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)
fig = plt.figure(figsize=(width, width/1.618))

std_dev = np.zeros((3, 151))
coeff_std_dev = np.zeros((3, 151))
max_rel_error = np.zeros((3, 151))  # Only plots max rel error
X_plot = np.linspace(0, 15e-9, 151, endpoint=True)
for count, x in enumerate(X_plot):
    print(count)
    X, Y, Z = np.meshgrid(x, 0, 0, indexing='ij')
    amplitude = 10e-9
    speed = 1
    B_total = np.zeros((3, 100))
    for k, num_e in enumerate([16, 64, 256]):
        charges = [None]*num_e
        phases = np.linspace(0, 2*np.pi, num_e, endpoint=False)
        for i in range(100):
            for j, phase in enumerate(phases):
                charges[j] = OrbittingCharge(pos_charge=True, phase=phase, amplitude=amplitude,
                                             max_speed=speed, start_zero=False)
            field = MovingChargesField(charges)
            B_total[k][i] = field.calculate_B(
                t=0, X=X, Y=Y, Z=Z, pcharge_field='Total', plane=True)[2]
            phases += 2*np.pi/(100*num_e)
    B_total[0] *= 16
    B_total[1] *= 4
    B_total[2] *= 1
    for i in range(3):
        std_dev[i][count] = np.sqrt(
            sum((B_total[i]-np.average(B_total[i]))**2)/len(B_total[i]))
        coeff_std_dev[i][count] = std_dev[i][count]/np.average(B_total[i])
        max_rel_error[i][count] = abs(
            (max(B_total[i])-np.average(B_total[i]))/np.average(B_total[i])*100)

fig = plt.figure(figsize=(width, width/1.618))

plt.semilogy(X_plot*1e9, max_rel_error[0], label='$n=16$')
plt.semilogy(X_plot*1e9, max_rel_error[1], label='$n=64$')
plt.semilogy(X_plot*1e9, max_rel_error[2], label='$n=256$')

plt.xlabel('$x$ [nm]')
plt.ylabel('Maximum Relative Error [\%]')
plt.xlim(0, 15)
plt.ylim(0, 10000)
plt.legend()
savename = 'Figure13'
plt.savefig('Figs/'+savename+'.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0.02, dpi=500)
plt.show()
