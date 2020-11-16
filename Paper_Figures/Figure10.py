# %% xy syncotrone  V, Ax, Ay, S
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
Vmax = 1e-0
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
X, Y, Z = np.meshgrid(np.linspace(-lim, lim, grid_size),
                      np.linspace(-lim, lim, grid_size), 0, indexing='ij')
field = MovingChargesField(charges)

V, Ax, Ay, _ = field.calculate_potentials(
    t=t, X=X, Y=Y, Z=Z, plane=True)
S = field.calculate_Poynting(
    t=t, X=X, Y=Y, Z=Z, plane=True)

axes = [None]*4
fig = plt.figure(figsize=(width, width))

left_shift = 0.125
right_shift = 0.48
bottom_shift = 0.12
height_shift = 0.06
plot_width = 0.225

axes[0] = fig.add_axes(
    [left_shift, bottom_shift+height_shift+plot_width, plot_width, plot_width])
axes[1] = fig.add_axes([left_shift+right_shift, bottom_shift +
                        height_shift+plot_width, plot_width, plot_width])
axes[2] = fig.add_axes([left_shift, bottom_shift, plot_width, plot_width])
axes[3] = fig.add_axes(
    [left_shift+right_shift, bottom_shift, plot_width, plot_width])

ims = [None]*4
for i, ax in enumerate(axes):
    ims[i] = ax.imshow(np.zeros((1, 1)), cmap='viridis', origin='lower')
    ax.set_aspect('equal')
    ax.text(-0.02, 1.07, '(' +
            string.ascii_lowercase[i]+')', transform=ax.transAxes, size=10)
    ims[i].set_extent((1e9*X[0, 0, 0], 1e9*X[-1, 0, 0],
                       1e9*Y[0, 0, 0], 1e9*Y[0, -1, 0]))
    ax.set_xticks((-25, 0, 25))
    ax.set_yticks((-25, 0, 25))

ims[0].set_norm(mpl.colors.SymLogNorm(linthresh=Amin, linscale=1,
                                      vmin=-Amax, vmax=Amax))
ims[1].set_norm(mpl.colors.LogNorm(vmin=Vmin, vmax=Vmax))
ims[2].set_norm(mpl.colors.SymLogNorm(linthresh=Amin, linscale=1,
                                      vmin=-Amax, vmax=Amax))
ims[3].set_norm(mpl.colors.LogNorm(vmin=Smin, vmax=Smax))

for i in range(3):
    axes[i].tick_params(axis="y", direction="inout")
    axes[i].tick_params(axis="x", direction="inout")

axes[2].set_ylabel('$y$ [nm]', y=1.1, labelpad=0)
axes[2].set_xlabel('$x$ [nm]')
axes[3].set_xlabel('$x$ [nm]')
axes[0].set_xticklabels([])
axes[1].set_xticklabels([])
axes[1].set_yticklabels([])
axes[3].set_yticklabels([])

ims[0].set_data(Ax.T)
ims[1].set_data(V.T)
ims[2].set_data(Ay.T)
ims[3].set_data(S.T)

for i, label in zip((0, 1, 2, 3), ('$A_x$ [V$\cdot$s/m]', '$\Phi$ [V]', '$A_y$ [V$\cdot$s/m]', '$S_\mathrm{rad}$ [W/m$^2$]')):
    cax = inset_axes(axes[i],
                     width="6%",
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.05, 0., 1, 1),
                     bbox_transform=axes[i].transAxes,
                     borderpad=0,
                     )
    cbar = plt.colorbar(ims[i], cax=cax)
    cbar.ax.set_ylabel(label, rotation=270, labelpad=11)

savename = 'Figure10'
plt.savefig('Figs/'+savename+'.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0.02, dpi=500)

# %% Animation
charges = (OrbittingCharge(pos_charge=True, start_zero=True,
                           second_charge=False, max_speed=0.5*c),)
field = MovingChargesField(charges)


def _update_animation(frame):
    text = "\rProcessing frame {0}/{1}.".format(frame+1, 240)
    sys.stdout.write(text)
    sys.stdout.flush()

    t = frame*dt
    V, Ax, Ay, _ = field.calculate_potentials(
        t=t, X=X, Y=Y, Z=Z, plane=True)
    S = field.calculate_Poynting(
        t=t, X=X, Y=Y, Z=Z, plane=True)
    ims[0].set_data(Ax.T)
    ims[1].set_data(V.T)
    ims[2].set_data(Ay.T)
    ims[3].set_data(S.T)
    return ims,


def _init_animate():
    """Necessary for matplotlib animate."""
    pass


dt = 2*np.pi/charges[0].w/24
ani = FuncAnimation(fig, _update_animation, interval=1000/24,
                    frames=240, blit=False, init_func=_init_animate)
ani.save('Animations/'+savename+'.mp4',
         writer=animation.FFMpegWriter(fps=24), dpi=500)
