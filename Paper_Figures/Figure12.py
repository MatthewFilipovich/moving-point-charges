# %% Field point Bz error
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

x = 9.9e-9
y = 0
z = 0
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
fig = plt.figure(figsize=(width, width/1.618))

amplitude = 10e-9
speed = 1
B_total = np.zeros((3, 101))
for k, num_e in enumerate([16, 64, 256]):
    charges = [None]*num_e
    phases = np.linspace(0, 2*np.pi, num_e, endpoint=False)

    for i in range(101):
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
plt.plot(B_total[0], label='n=16')
plt.plot(B_total[1], label='n=64')
plt.plot(B_total[2], label='n=256')
plt.gca().ticklabel_format(style='sci', scilimits=(-3, 4), axis='both')

plt.xlim(0, 100)
plt.xticks((0, 25, 50, 75, 100),
           ('0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'))
plt.xlabel('$n\phi$  [rad]')
plt.ylabel('$B_z$ [T]')
plt.legend(loc=2)
savename = 'Figure12'
plt.savefig('Figs/'+savename+'.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0.02, dpi=500)
