# %% xy current approximation Ex, Ey, Bz
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

amplitude = 10e-9
speed = 1
lim = 20e-9
grid_size = 1000
E = [None]*3
B = [None]*3

for i, num_e in enumerate((16, 64, 256)):
    phases = np.linspace(0, 2*np.pi, num_e, endpoint=False)
    charges = [None]*num_e
    for j, phase in enumerate(phases):
        charges[j] = OrbittingCharge(pos_charge=True, phase=phase, amplitude=amplitude,
                                     max_speed=speed, start_zero=False)
    w = charges[0].w
    t = 0
    X, Y, Z = np.meshgrid(np.linspace(-lim, lim, grid_size),
                          np.linspace(-lim, lim, grid_size), 0, indexing='ij')
    field = MovingChargesField(charges)
    E[i] = field.calculate_E(
        t=0, X=X, Y=Y, Z=Z, pcharge_field='Velocity', plane=True)
    B[i] = field.calculate_B(
        t=0, X=X, Y=Y, Z=Z, pcharge_field='Velocity', plane=True)

axes = [None]*9
fig = plt.figure(figsize=(width, width))

left_shift = 0.125
bottom_shift = 0.12
cbar_shift = 0.2
height_shift = 0.05
plot_width = (1-left_shift-cbar_shift)/3

axes[0] = fig.add_axes(
    [left_shift, bottom_shift+2*plot_width+2*height_shift, plot_width, plot_width])
axes[1] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        2*plot_width+2*height_shift, plot_width, plot_width])
axes[2] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        2*plot_width+2*height_shift, plot_width, plot_width])

axes[3] = fig.add_axes(
    [left_shift, bottom_shift+plot_width+height_shift, plot_width, plot_width])
axes[4] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        plot_width+height_shift, plot_width, plot_width])
axes[5] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        plot_width+height_shift, plot_width, plot_width])

axes[6] = fig.add_axes([left_shift, bottom_shift, plot_width, plot_width])
axes[7] = fig.add_axes(
    [left_shift+plot_width, bottom_shift, plot_width, plot_width])
axes[8] = fig.add_axes(
    [left_shift+2*plot_width, bottom_shift, plot_width, plot_width])

Emax = 1e10
Bmax = 1e-7
log_scale = 1e-2
Emin = Emax*log_scale
Bmin = Bmax*log_scale

ims = [None]*9
for i, ax in enumerate(axes):
    ims[i] = ax.imshow(np.zeros((1, 1)), cmap='viridis', origin='lower')
    ax.set_aspect('equal')
    ax.text(-0.02, 1.06, '(' +
            string.ascii_lowercase[i]+')', transform=ax.transAxes, size=10)
    ims[i].set_extent((1e9*X[0, 0, 0], 1e9*X[-1, 0, 0],
                       1e9*Y[0, 0, 0], 1e9*Y[0, -1, 0]))
    ax.set_xticks((-10, 0, 10))
    ax.set_yticks((-10, 0, 10))

for i in range(6):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Emin, linscale=1,
                                          vmin=-Emax, vmax=Emax))
for i in range(6, 9):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Bmin, linscale=1,
                                          vmin=-Bmax, vmax=Bmax))

for i in range(6):
    axes[i].set_xticklabels([])
for i in (1, 2, 4, 5, 7, 8):
    axes[i].set_yticklabels([])
for i in range(9):
    axes[i].tick_params(axis="y", direction="inout")
    axes[i].tick_params(axis="x", direction="inout")

axes[3].set_ylabel('$y$ [nm]', labelpad=0)
axes[7].set_xlabel('$x$ [nm]')

ims[0].set_data(E[0][0].T*16)
ims[1].set_data(E[1][0].T*4)
ims[2].set_data(E[2][0].T*1)
ims[3].set_data(E[0][1].T*16)
ims[4].set_data(E[1][1].T*4)
ims[5].set_data(E[2][1].T*1)
ims[6].set_data(B[0][2].T*16)
ims[7].set_data(B[1][2].T*4)
ims[8].set_data(B[2][2].T*1)

axes[0].text(0.26, 1.25, '$\mathbf{Total}$',
             transform=axes[0].transAxes, size=10)
axes[1].text(0.1, 1.25, '$\mathbf{Coulomb}$',
             transform=axes[1].transAxes, size=10)
axes[2].text(0.08, 1.25, '$\mathbf{Radiation}$',
             transform=axes[2].transAxes, size=10)

for i, label in zip((2, 5, 8), ('$E_x$ [N/C]', '$E_y$ [N/C]', '$B_z$ [T]')):
    Ecax = inset_axes(axes[i],
                      width="6%",
                      height="100%",
                      loc='lower left',
                      bbox_to_anchor=(1.05, 0., 1, 1),
                      bbox_transform=axes[i].transAxes,
                      borderpad=0,
                      )
    E_cbar = plt.colorbar(ims[i], cax=Ecax)
    E_cbar.ax.set_ylabel(label, rotation=270, labelpad=12)

savename = 'Figure11'
plt.savefig('Figs/'+savename+'.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0.02, dpi=500)
