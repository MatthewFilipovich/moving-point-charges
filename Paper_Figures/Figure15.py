# %% xz Decelerating charge with time Ex, Ez, By
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
from field_calculations import *
from charges import *

c = constants.c
width = 6.4
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

initial_speed = 0.99*c
x_stop = 30e-9

deceleration = 0.5*initial_speed**2/x_stop
t_end = initial_speed/deceleration
charges = LinearDeceleratingCharge(
    pos_charge=True, deceleration=deceleration, initial_speed=initial_speed, stop_t=t_end)

X, Y, Z = np.meshgrid(np.linspace(-lim, lim, grid_size), 0,
                      np.linspace(-lim, lim, grid_size), indexing='ij')
field = MovingChargesField(charges)

E_acc = [None]*6
B_acc = [None]*6
xpos = np.zeros(6)
for i in range(6):
    E_acc[i] = field.calculate_E(
        t=(i)*t_end/5, X=X, Y=Y, Z=Z, pcharge_field='Total', plane=True)
    B_acc[i] = field.calculate_B(
        t=(i)*t_end/5, X=X, Y=Y, Z=Z, pcharge_field='Total', plane=True)
    xpos[i] = charges.xpos(t=np.array([i*t_end/5]))

axes = [None]*18
fig = plt.figure(figsize=(width, width))

left_shift = 0.06
bottom_shift = 0.06
cbar_shift = 0.1
height_shift = 0.028
plot_width = (1-left_shift-cbar_shift)/6

axes[0] = fig.add_axes(
    [left_shift, bottom_shift+2*plot_width+2*height_shift, plot_width, plot_width])
axes[1] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        2*plot_width+2*height_shift, plot_width, plot_width])
axes[2] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        2*plot_width+2*height_shift, plot_width, plot_width])
axes[3] = fig.add_axes([left_shift+3*plot_width, bottom_shift +
                        2*plot_width+2*height_shift, plot_width, plot_width])
axes[4] = fig.add_axes([left_shift+4*plot_width, bottom_shift +
                        2*plot_width+2*height_shift, plot_width, plot_width])
axes[5] = fig.add_axes([left_shift+5*plot_width, bottom_shift +
                        2*plot_width+2*height_shift, plot_width, plot_width])

axes[6] = fig.add_axes(
    [left_shift, bottom_shift+plot_width+height_shift, plot_width, plot_width])
axes[7] = fig.add_axes([left_shift+plot_width, bottom_shift +
                        plot_width+height_shift, plot_width, plot_width])
axes[8] = fig.add_axes([left_shift+2*plot_width, bottom_shift +
                        plot_width+height_shift, plot_width, plot_width])
axes[9] = fig.add_axes([left_shift+3*plot_width, bottom_shift +
                        plot_width+height_shift, plot_width, plot_width])
axes[10] = fig.add_axes([left_shift+4*plot_width, bottom_shift +
                         plot_width+height_shift, plot_width, plot_width])
axes[11] = fig.add_axes([left_shift+5*plot_width, bottom_shift +
                         plot_width+height_shift, plot_width, plot_width])

axes[12] = fig.add_axes([left_shift, bottom_shift, plot_width, plot_width])
axes[13] = fig.add_axes(
    [left_shift+plot_width, bottom_shift, plot_width, plot_width])
axes[14] = fig.add_axes(
    [left_shift+2*plot_width, bottom_shift, plot_width, plot_width])
axes[15] = fig.add_axes(
    [left_shift+3*plot_width, bottom_shift, plot_width, plot_width])
axes[16] = fig.add_axes(
    [left_shift+4*plot_width, bottom_shift, plot_width, plot_width])
axes[17] = fig.add_axes(
    [left_shift+5*plot_width, bottom_shift, plot_width, plot_width])

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

for i in range(6):
    axes[i].scatter(xpos[i]*1e9, 0, s=3, c='red', marker='o')
    axes[i+6].scatter(xpos[i]*1e9, 0, s=3, c='red', marker='o')
    axes[i+12].scatter(xpos[i]*1e9, 0, s=3, c='red', marker='o')

for i in range(12):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Emin, linscale=1,
                                          vmin=-Emax, vmax=Emax))
for i in range(12, 18):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Bmin, linscale=1,
                                          vmin=-Bmax, vmax=Bmax))

for i in range(12):
    axes[i].set_xticklabels([])
for i in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17):
    axes[i].set_yticklabels([])
for i in range(18):
    axes[i].tick_params(axis="y", direction="inout")
    axes[i].tick_params(axis="x", direction="inout")

axes[6].set_ylabel('$z$ [nm]', labelpad=0)
axes[14].set_xlabel('$x$ [nm]', x=1)

ims[0].set_data(E_acc[0][0].T)
ims[1].set_data(E_acc[1][0].T)
ims[2].set_data(E_acc[2][0].T)
ims[3].set_data(E_acc[3][0].T)
ims[4].set_data(E_acc[4][0].T)
ims[5].set_data(E_acc[5][0].T)
ims[6].set_data(E_acc[0][2].T)
ims[7].set_data(E_acc[1][2].T)
ims[8].set_data(E_acc[2][2].T)
ims[9].set_data(E_acc[3][2].T)
ims[10].set_data(E_acc[4][2].T)
ims[11].set_data(E_acc[5][2].T)
ims[12].set_data(B_acc[0][1].T)
ims[13].set_data(B_acc[1][1].T)
ims[14].set_data(B_acc[2][1].T)
ims[15].set_data(B_acc[3][1].T)
ims[16].set_data(B_acc[4][1].T)
ims[17].set_data(B_acc[5][1].T)

for i, label in zip((5, 11, 17), ('$E_x$ [N/C]', '$E_z$ [N/C]', '$B_y$ [T]')):
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

savename = 'Figure15'
plt.savefig('Figs/'+savename+'.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0.02, dpi=500)

# %% Animation

axes = [None]*9
width = 3.519
fig = plt.figure(figsize=(width, width))
left_shift = 0.125
bottom_shift = 0.12
cbar_shift = 0.2
height_shift = 0.05
plot_width = 0.225

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

ims = [None]*9
for i, ax in enumerate(axes):
    ims[i] = ax.imshow(np.zeros((1, 1)), cmap='viridis', origin='lower')
    ax.set_aspect('equal')
    ax.text(-0.02, 1.07, '(' +
            string.ascii_lowercase[i]+')', transform=ax.transAxes, size=10)
    ims[i].set_extent((1e9*X[0, 0, 0], 1e9*X[-1, 0, 0],
                       1e9*Z[0, 0, 0], 1e9*Z[0, 0, -1]))
    ax.set_xticks((-25, 0, 25))
    ax.set_yticks((-25, 0, 25))

for i in range(6):
    axes[i].set_xticklabels([])
for i in (1, 2, 4, 5, 7, 8):
    axes[i].set_yticklabels([])
for i in range(9):
    axes[i].tick_params(axis="y", direction="inout")
    axes[i].tick_params(axis="x", direction="inout")

axes[3].set_ylabel('$z$ [nm]', labelpad=0)
axes[7].set_xlabel('$x$ [nm]')

for i in range(6):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Emin, linscale=1,
                                          vmin=-Emax, vmax=Emax))
for i in range(6, 9):
    ims[i].set_norm(mpl.colors.SymLogNorm(linthresh=Bmin, linscale=1,
                                          vmin=-Bmax, vmax=Bmax))


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


dt = t_end/239
ani = FuncAnimation(fig, _update_animation, interval=1000/24,
                    frames=240, blit=False, init_func=_init_animate)
ani.save('Animations/'+savename+'.mp4',
         writer=animation.FFMpegWriter(fps=24), dpi=500)
