# %% xz syncotrone  Ex, Ey, Ez, Bx By, Bz
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

charges = (OrbittingCharge(pos_charge=True,
                           second_charge=False, max_speed=0.5*c),)
t = 0
X, Y, Z = np.meshgrid(np.linspace(-lim, lim, grid_size), 0,
                      np.linspace(-lim, lim, grid_size),  indexing='ij')
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

axes = [None]*18
fig = plt.figure(figsize=(width, width))

left_shift = 0.125
bottom_shift = 0.12
cbar_shift = 0.2
height_shift = 0.05
plot_width = 0.225

axes[0] = fig.add_axes(
    [left_shift, bottom_shift+5*plot_width+5*height_shift, plot_width, plot_width])
axes[1] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        5*plot_width+5*height_shift, plot_width, plot_width])
axes[2] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        5*plot_width+5*height_shift, plot_width, plot_width])

axes[3] = fig.add_axes(
    [left_shift, bottom_shift+4*plot_width+4*height_shift, plot_width, plot_width])
axes[4] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        4*plot_width+4*height_shift, plot_width, plot_width])
axes[5] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        4*plot_width+4*height_shift, plot_width, plot_width])

axes[6] = fig.add_axes(
    [left_shift, bottom_shift+3*plot_width+3*height_shift, plot_width, plot_width])
axes[7] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        3*plot_width+3*height_shift, plot_width, plot_width])
axes[8] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        3*plot_width+3*height_shift, plot_width, plot_width])

axes[9] = fig.add_axes(
    [left_shift, bottom_shift+2*plot_width+2*height_shift, plot_width, plot_width])
axes[10] = fig.add_axes([left_shift+plot_width, bottom_shift +
                         2*plot_width+2*height_shift, plot_width, plot_width])
axes[11] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                         2*plot_width+2*height_shift, plot_width, plot_width])

axes[12] = fig.add_axes(
    [left_shift, bottom_shift+plot_width+height_shift, plot_width, plot_width])
axes[13] = fig.add_axes([left_shift+plot_width, bottom_shift +
                         plot_width+height_shift, plot_width, plot_width])
axes[14] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                         plot_width+height_shift, plot_width, plot_width])

axes[15] = fig.add_axes([left_shift, bottom_shift, plot_width, plot_width])
axes[16] = fig.add_axes(
    [left_shift+plot_width, bottom_shift, plot_width, plot_width])
axes[17] = fig.add_axes(
    [left_shift+2*plot_width, bottom_shift, plot_width, plot_width])


ims = [None]*18
for i, ax in enumerate(axes):
    ims[i] = ax.imshow(np.zeros((1, 1)), cmap='viridis', origin='lower')
    ax.set_aspect('equal')
    ax.text(-0.02, 1.06, '(' +
            string.ascii_lowercase[i]+')', transform=ax.transAxes, size=10)
    ims[i].set_extent((1e9*X[0, 0, 0], 1e9*X[-1, 0, 0],
                       1e9*Z[0, 0, 0], 1e9*Z[0, 0, -1]))
    ax.set_xticks((-25, 0, 25))
    ax.set_yticks((-25, 0, 25))

for i in range(9):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Emin, linscale=1,
                                          vmin=-Emax, vmax=Emax))
for i in range(9, 18):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Bmin, linscale=1,
                                          vmin=-Bmax, vmax=Bmax))

for i in range(15):
    axes[i].set_xticklabels([])
for i in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17):
    axes[i].set_yticklabels([])
for i in range(18):
    axes[i].tick_params(axis="y", direction="inout")
    axes[i].tick_params(axis="x", direction="inout")

axes[9].set_ylabel('$z$ [nm]', y=1.1, labelpad=0)
axes[16].set_xlabel('$x$ [nm]')

ims[0].set_data(E_total[0].T)
ims[1].set_data(E_vel[0].T)
ims[2].set_data(E_acc[0].T)
ims[3].set_data(E_total[1].T)
ims[4].set_data(E_vel[1].T)
ims[5].set_data(E_acc[1].T)
ims[6].set_data(E_total[2].T)
ims[7].set_data(E_vel[2].T)
ims[8].set_data(E_acc[2].T)
ims[9].set_data(B_total[0].T)
ims[10].set_data(B_vel[0].T)
ims[11].set_data(B_acc[0].T)
ims[12].set_data(B_total[1].T)
ims[13].set_data(B_vel[1].T)
ims[14].set_data(B_acc[1].T)
ims[15].set_data(B_total[2].T)
ims[16].set_data(B_vel[2].T)
ims[17].set_data(B_acc[2].T)

axes[0].text(0.26, 1.25, '$\mathbf{Total}$',
             transform=axes[0].transAxes, size=10)
axes[1].text(0.1, 1.25, '$\mathbf{Coulomb}$',
             transform=axes[1].transAxes, size=10)
axes[2].text(0.08, 1.25, '$\mathbf{Radiation}$',
             transform=axes[2].transAxes, size=10)

for i, label in zip((2, 5, 8, 11, 14, 17), ('$E_x$ [N/C]', '$E_y$ [N/C]', '$E_z$ [N/C]', '$B_x$ [T]', '$B_y$ [T]', '$B_z$ [T]')):
    Ecax = inset_axes(axes[i],
                      width="6%",  # width = 5% of parent_bbox width
                      height="100%",  # height : 50%
                      loc='lower left',
                      bbox_to_anchor=(1.05, 0., 1, 1),
                      bbox_transform=axes[i].transAxes,
                      borderpad=0,
                      )
    E_cbar = plt.colorbar(ims[i], cax=Ecax)
    E_cbar.ax.set_ylabel(label, rotation=270, labelpad=12)


savename = 'Figure8'
plt.savefig('Figs/'+savename+'.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0.02, dpi=500)

# %% Animation
charges = (OrbittingCharge(pos_charge=True, start_zero=True,
                           second_charge=False, max_speed=0.5*c),)
field = MovingChargesField(charges)


axes = [None]*18
width = 7.2888
fig = plt.figure(figsize=(width, width))

left_shift = 0.125
bottom_shift = 0.06
cbar_shift = 0.2
height_shift = 0.03
plot_width = 0.127

axes[0] = fig.add_axes(
    [left_shift, bottom_shift+5*plot_width+5*height_shift, plot_width, plot_width])
axes[1] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        5*plot_width+5*height_shift, plot_width, plot_width])
axes[2] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        5*plot_width+5*height_shift, plot_width, plot_width])

axes[3] = fig.add_axes(
    [left_shift, bottom_shift+4*plot_width+4*height_shift, plot_width, plot_width])
axes[4] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        4*plot_width+4*height_shift, plot_width, plot_width])
axes[5] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        4*plot_width+4*height_shift, plot_width, plot_width])

axes[6] = fig.add_axes(
    [left_shift, bottom_shift+3*plot_width+3*height_shift, plot_width, plot_width])
axes[7] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        3*plot_width+3*height_shift, plot_width, plot_width])
axes[8] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        3*plot_width+3*height_shift, plot_width, plot_width])

axes[9] = fig.add_axes(
    [left_shift, bottom_shift+2*plot_width+2*height_shift, plot_width, plot_width])
axes[10] = fig.add_axes([left_shift+plot_width, bottom_shift +
                         2*plot_width+2*height_shift, plot_width, plot_width])
axes[11] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                         2*plot_width+2*height_shift, plot_width, plot_width])

axes[12] = fig.add_axes(
    [left_shift, bottom_shift+plot_width+height_shift, plot_width, plot_width])
axes[13] = fig.add_axes([left_shift+plot_width, bottom_shift +
                         plot_width+height_shift, plot_width, plot_width])
axes[14] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                         plot_width+height_shift, plot_width, plot_width])

axes[15] = fig.add_axes([left_shift, bottom_shift, plot_width, plot_width])
axes[16] = fig.add_axes(
    [left_shift+plot_width, bottom_shift, plot_width, plot_width])
axes[17] = fig.add_axes(
    [left_shift+2*plot_width, bottom_shift, plot_width, plot_width])


ims = [None]*18
for i, ax in enumerate(axes):
    ims[i] = ax.imshow(np.zeros((1, 1)), cmap='viridis', origin='lower')
    ax.set_aspect('equal')
    ax.text(-0.02, 1.06, '(' +
            string.ascii_lowercase[i]+')', transform=ax.transAxes, size=10)
    ims[i].set_extent((1e9*X[0, 0, 0], 1e9*X[-1, 0, 0],
                       1e9*Z[0, 0, 0], 1e9*Z[0, 0, -1]))
    ax.set_xticks((-25, 0, 25))
    ax.set_yticks((-25, 0, 25))

for i in range(9):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Emin, linscale=1,
                                          vmin=-Emax, vmax=Emax))
for i in range(9, 18):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Bmin, linscale=1,
                                          vmin=-Bmax, vmax=Bmax))

for i in range(15):
    axes[i].set_xticklabels([])
for i in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17):
    axes[i].set_yticklabels([])
for i in range(18):
    axes[i].tick_params(axis="y", direction="inout")
    axes[i].tick_params(axis="x", direction="inout")

axes[9].set_ylabel('$z$ [nm]', y=1.1, labelpad=0)
axes[16].set_xlabel('$x$ [nm]')

for i, label in zip((2, 5, 8, 11, 14, 17), ('$E_x$ [N/C]', '$E_y$ [N/C]', '$E_z$ [N/C]', '$B_x$ [T]', '$B_y$ [T]', '$B_z$ [T]')):
    Ecax = inset_axes(axes[i],
                      width="6%",  # width = 5% of parent_bbox width
                      height="100%",  # height : 50%
                      loc='lower left',
                      bbox_to_anchor=(1.05, 0., 1, 1),
                      bbox_transform=axes[i].transAxes,
                      borderpad=0,
                      )
    E_cbar = plt.colorbar(ims[i], cax=Ecax)
    E_cbar.ax.set_ylabel(label, rotation=270, labelpad=12)


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
    ims[3].set_data(E_total[1].T)
    ims[4].set_data(E_vel[1].T)
    ims[5].set_data(E_acc[1].T)
    ims[6].set_data(E_total[2].T)
    ims[7].set_data(E_vel[2].T)
    ims[8].set_data(E_acc[2].T)
    ims[9].set_data(B_total[0].T)
    ims[10].set_data(B_vel[0].T)
    ims[11].set_data(B_acc[0].T)
    ims[12].set_data(B_total[1].T)
    ims[13].set_data(B_vel[1].T)
    ims[14].set_data(B_acc[1].T)
    ims[15].set_data(B_total[2].T)
    ims[16].set_data(B_vel[2].T)
    ims[17].set_data(B_acc[2].T)
    return ims,


def _init_animate():
    """Necessary for matplotlib animate."""
    pass


dt = 2*np.pi/charges[0].w/24
ani = FuncAnimation(fig, _update_animation, interval=1000/24,
                    frames=240, blit=False, init_func=_init_animate)
ani.save('Animations/'+savename+'.mp4',
         writer=animation.FFMpegWriter(fps=24), dpi=500)
