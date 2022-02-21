"""
Miscellaneous plotting helpers.
"""

# std
from fractions import Fraction

# third-party
import numpy as np
from astropy import units as u
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import FuncFormatter
from matplotlib.collections import LineCollection

# local
from recipes.array import fold
from recipes import pprint as pp
from recipes.transforms import sph2cart


# ---------------------------------------------------------------------------- #
π = np.pi


_POW10 = {-1: 0.1,
          0:  1,
          1:  10,
          2:  100}


def format_x10(n, _pos=None):
    n = int(n)
    return str(_POW10.get(n, None) or rf'10^{{{n}}}').join('$$')


# def float_as_pi_frac(n, _pos=None):
#     return pi_frac.latex(n / np.pi)


x10_formatter = FuncFormatter(format_x10)

pi_frac = pp.formatters.FractionOfPi()
pi_frac_formatter = FuncFormatter(pi_frac.latex)
pi_radian_formatter = FuncFormatter(pi_frac.from_radian)

# ---------------------------------------------------------------------------- #


def theta_tickmarks(ax, n=None, direction='inout', length=0.02, width=0.72,
                    color='k'):
    """
    Tick marks for polar plots.
    """
    # adapted from: https://stackoverflow.com/a/44657941/1098683

    assert direction in {'in', 'out', 'inout'}

    d = {'in':    [0, -1],
         'out':   [0, 1],
         'inout': [-1, 1]}[direction]

    if n is None:
        n = len(ax.xaxis.get_majorticklocs()) + 1
    
    
    tick = ax.get_rmax() + length * np.array(d)
    for t in np.linspace(0, 2 * np.pi, n):
        ax.plot([t, t], tick, lw=width, color=color, clip_on=False)

# ---------------------------------------------------------------------------- #
from mpl_toolkits.mplot3d.axes3d import Axes3D

def plot_line_cmap(ax, x, cbar=True, **kws):
    # colorline
    
    ndim = x.shape[0]
    assert ndim in (2, 3)

    Lines = LineCollection if ndim == 2 else Line3DCollection
    lc = Lines(fold.fold(x.T, 2, 1, pad=False), **kws)
    ax.add_collection(lc)
    
    if isinstance(ax, Axes3D):
        ax.auto_scale_xyz(*x)
    else:
        ax.autoscale_view()

    cb = None
    if cbar:
        cb = ax.figure.colorbar(lc,
                                ax=ax, pad=0.03,
                                ticks=np.linspace(0, np.pi, 7),
                                format=pi_frac_formatter)
        cb.ax.set_ylabel(r'$\theta\ [rad]$', rotation=0, labelpad=20,
                         usetex=True)

    return lc, cb


def plot_line_family(ax, x, u, cbar=True, **kws):

    segments = np.empty((*u.shape, x.ndim))
    segments[..., :-1] = x
    segments[..., -1] = u

    lc = LineCollection(segments, **kws)
    ax.add_collection(lc)
    ax.autoscale_view()

    if cbar:
        cb = ax.figure.colorbar(lc,
                                ax=ax, pad=0.03,
                                ticks=np.linspace(0, np.pi, 7),
                                format=pi_frac_formatter)
        cb.ax.set_ylabel(r'$\theta\ [rad]$', rotation=0, labelpad=20, usetex=True)

    return lc, cb


def plot_isopotential_solution(roche, axes, r, theta, phi, u, iso_radii, primary,
                               cmap='jet'):
    """
    Helper for plotting isopotential solutions.
    """

    # segments = np.empty((*u.shape, 2))
    # segments[..., 0] = r
    # segments[..., 1] = u

    # plot
    ax, ax1 = axes

    # Potentials (radial)
    family = plot_radial_potentials(roche, ax, r, theta, phi, u, iso_radii, cmap)

    # isopotential curve
    contour = plot_iso(roche, ax1, r, theta, phi, primary, cmap)

    return family, contour


def plot_iso(roche, ax, r, theta, phi, primary, cmap):

    secondary = not primary
    invert = (-1, 1)[primary]
    mu = secondary + invert * roche.u.mu
    centre = 0  # getattr(roche.u, f'r{(2, 1)[primary]}').x / roche.u.a
    rmax = invert * (roche.u.l1.x / roche.u.a) - mu + 1

    # isopotential curve

    # roche lobe
    x, y, _ = sph2cart(r, phi, theta)
    x = invert * x + centre

    ax.plot(x, y, 'x', color='orangered', ms=3, zorder=10)
    contour, _ = plot_line_cmap(ax, np.array([x, y]),
                                cbar=True, array=theta, cmap=cmap)
    ax.plot(invert * rmax + centre, 0, 'r*')
    ax.plot(centre, 0, 'g+')

    # ax.plot(x, y)
    ax.grid()
    ax.set_xlabel(r'$x\ [a]$', usetex=True)
    ax.set_ylabel(r'$y\ [a]$', usetex=True)
    # ax1.set_aspect('equal')

    return contour


def plot_radial_potentials(roche, ax, r, theta, phi, u, iso_radii, cmap):
    # Potentials (radial)
    family, _ = plot_line_family(ax, r, u, cbar=False, array=theta, cmap=cmap)
    ax.plot(iso_radii, np.zeros_like(iso_radii), 'mx')
    ax.plot((roche.u.l1 - roche.u.r1).x, 0, 'r*')

    ax.grid()
    ax.set_xlabel(r'$r\ [a]$', usetex=True)
    ax.set_ylabel(
        rf'$\Psi(r,\theta,\phi={pi_frac.latex(Fraction(phi/π)).strip("$")})\ '
        r'\left[\frac{GM}{2a}\right]$',
        usetex=True
    )
    return family


def plot_multipole_rθ(L=range(6, 7)):

    from mCV.bfield import solve_theta_r
    from mCV.bfield import pi_radian_formatter, get_theta_fieldlines, PhysicalMultipole
    from scipy.special import factorial, lpmv
    from scrawl.dualaxes import DualAxes


    fig = plt.figure(figsize=[15.65,  4.8])


    ax = DualAxes(fig, 1, 1, 1)
    fig.add_subplot(ax)
    ax.setup_ticks()


    res = 100
    rn = 1
    for l in L:
        self = PhysicalMultipole(degree=l)

        θ = get_theta_fieldlines(l)
        sinθ, cosθ = np.sin(θ), np.cos(θ)
        r = rn * self._fieldline_radial(θ)

        line, = ax.plot(θ, r, '-',  label=fr'$\ell = {l}$')
        #line, = ax.plot(θ, sl, '--', color=line.get_color())
    # line, = ax.plot(θ, rprime, ':', color=line.get_color())

        thc = self.theta_c
        #print(f'l = {l}\nzeros = {thc}\n')
        ax.plot(thc, np.zeros_like(thc), 'x',  color=line.get_color())
        ax.plot(self.θ_max, self.rmax, '*',  color=line.get_color())

    ax.grid()
    ax.legend()

    xticks = np.linspace(0, np.pi/2, 7)
    ax.set(xlabel=r'$\theta$', xticks=xticks)
    ax.set_ylabel(r'$r(\theta)$', rotation=0, labelpad=20)

    # ax.parasite.yaxis.offsetText.set_visible(False)
    ax.parasite.set(xlabel=r'$\theta$', xticks=xticks)
    ax.parasite.xaxis.set_major_formatter(pi_radian_formatter)

    # rx = solve_theta_r(l, 0.856)
    # ax.axhline(0.856)
    # ax.vlines(rx, *ax.get_ylim())