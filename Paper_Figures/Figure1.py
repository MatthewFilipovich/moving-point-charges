# %% xz quiver oscillating charge Ex, By, Ez
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
lim = 50e-9
grid_size = 1000
Emax = 1e-2
Bmax = 1e-1
Vmax = 1e-1
Amax = 1e-9
Smax = 1e12

log_scale = 1e-2
Emin = Emax*log_scale
Bmin = Bmax*log_scale
Vmin = Vmax*log_scale
Amin = Amax*log_scale
Smin = Smax*log_scale

charges = (OscillatingCharge(pos_charge=True, direction=(
    1, 0, 0), start_position=(-2e-9, 0, 0), max_speed=0.5*c),)
t = 0
X, Y, Z = np.meshgrid(np.linspace(-lim, lim, grid_size), 0,
                      np.linspace(-lim, lim, grid_size), indexing='ij')
field = MovingChargesField(charges)

E_total = field.calculate_E(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Total', plane=True)
B_total = field.calculate_B(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Total', plane=True)
E_acc = field.calculate_E(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Acceleration', plane=True)
B_acc = field.calculate_B(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Acceleration', plane=True)
E_vel = field.calculate_E(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Velocity', plane=True)
B_vel = field.calculate_B(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Velocity', plane=True)

fig, ax = plt.subplots(figsize=(width*0.7, width*0.7))
u = E_total[0]
v = E_total[2]
im = plt.imshow(np.sqrt(u**2+v**2).T, origin='lower',
                extent=[-lim, lim, -lim, lim], vmax=7)
plt.xticks([-lim, -lim/2, 0, lim/2, lim], [-50, -25, 0, 25, 50])
plt.yticks([-lim, -lim/2, 0, lim/2, lim], [-50, -25, 0, 25, 50])
im.set_norm(mpl.colors.LogNorm(vmin=1e5, vmax=1e8))


grid_size = 19
lim = 46e-9
X, Y, Z = np.meshgrid(np.linspace(-lim, lim, grid_size), 0,
                      np.linspace(-lim, lim, grid_size), indexing='ij')
field = MovingChargesField(charges)

E_total = field.calculate_E(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Total', plane=True)
B_total = field.calculate_B(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Total', plane=True)
E_acc = field.calculate_E(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Acceleration', plane=True)
B_acc = field.calculate_B(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Acceleration', plane=True)
E_vel = field.calculate_E(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Velocity', plane=True)
B_vel = field.calculate_B(
    t=t, X=X, Y=Y, Z=Z, pcharge_field='Velocity', plane=True)

u = E_total[0]
v = E_total[2]
r = np.power(np.add(np.power(u, 2), np.power(v, 2)), 0.5)
cb = fig.colorbar(im, fraction=0.046, pad=0.04)
cb.ax.set_ylabel('$|\mathbf{E}|$ [N/C]', rotation=270, labelpad=12)

Q = plt.quiver(X, Z, u/r, v/r, scale_units='xy')
plt.xlabel('$x$ [nm]')
plt.ylabel('$z$ [nm]')

ax.scatter(-2e-9, 0, s=3, c='red', marker='o')

savename = 'Figure1'
plt.savefig('Figs/'+savename+'.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0.02, dpi=500)
plt.show()
