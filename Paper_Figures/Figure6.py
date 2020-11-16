# %% xz oscillating dipole Ex, By, Ez
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
Emax = 1e7
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

charges = (OscillatingCharge(pos_charge=True, direction=(1, 0, 0), start_position=(-2e-9, 0, 0), max_speed=0.5*c),
           OscillatingCharge(pos_charge=False, direction=(-1, 0, 0), start_position=(2e-9, 0, 0), max_speed=0.5*c))
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

ims = [None]*9
for i, ax in enumerate(axes):
    ims[i] = ax.imshow(np.zeros((1, 1)), cmap='viridis', origin='lower')
    ax.set_aspect('equal')
    ax.text(-0.02, 1.06, '(' +
            string.ascii_lowercase[i]+')', transform=ax.transAxes, size=10)
    ims[i].set_extent((1e9*X[0, 0, 0], 1e9*X[-1, 0, 0],
                       1e9*Z[0, 0, 0], 1e9*Z[0, 0, -1]))
    ax.set_xticks((-25, 0, 25))
    ax.set_yticks((-25, 0, 25))

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

axes[3].set_ylabel('$z$ [nm]', labelpad=0)
axes[7].set_xlabel('$x$ [nm]')

ims[0].set_data(E_total[0].T)
ims[1].set_data(E_vel[0].T)
ims[2].set_data(E_acc[0].T)
ims[3].set_data(E_total[2].T)
ims[4].set_data(E_vel[2].T)
ims[5].set_data(E_acc[2].T)
ims[6].set_data(B_total[1].T)
ims[7].set_data(B_vel[1].T)
ims[8].set_data(B_acc[1].T)

axes[0].text(0.26, 1.25, '$\mathbf{Total}$',
             transform=axes[0].transAxes, size=10)
axes[1].text(0.1, 1.25, '$\mathbf{Coulomb}$',
             transform=axes[1].transAxes, size=10)
axes[2].text(0.08, 1.25, '$\mathbf{Radiation}$',
             transform=axes[2].transAxes, size=10)

for i, label in zip((2, 5, 8), ('$E_x$ [N/C]', '$E_z$ [N/C]', '$B_y$ [T]')):
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

savename = 'Figure6'
plt.savefig('Figs/'+savename+'.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0.02, dpi=500)

# %% Animation
charges = (OscillatingCharge(pos_charge=True, direction=(1, 0, 0), start_position=(-2e-9, 0, 0), max_speed=0.5*c, start_zero=True),
           OscillatingCharge(pos_charge=False, direction=(-1, 0, 0), start_position=(2e-9, 0, 0), max_speed=0.5*c, start_zero=True))
field = MovingChargesField(charges)


def _update_animation(frame):
    text = "\rProcessing frame {0}/{1}.".format(frame+1, 240)
    sys.stdout.write(text)
    sys.stdout.flush()

    t = frame*dt
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
    ims[0].set_data(E_total[0].T)
    ims[1].set_data(E_vel[0].T)
    ims[2].set_data(E_acc[0].T)
    ims[3].set_data(E_total[2].T)
    ims[4].set_data(E_vel[2].T)
    ims[5].set_data(E_acc[2].T)
    ims[6].set_data(B_total[1].T)
    ims[7].set_data(B_vel[1].T)
    ims[8].set_data(B_acc[1].T)
    return ims,


def _init_animate():
    """Necessary for matplotlib animate."""
    pass


dt = 2*np.pi/charges[0].w/24
ani = FuncAnimation(fig, _update_animation, interval=1000/24,
                    frames=240, blit=False, init_func=_init_animate)
ani.save('Animations/'+savename+'.mp4',
         writer=animation.FFMpegWriter(fps=24), dpi=500)
