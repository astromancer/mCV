

# std
from fractions import Fraction

# third-party
import numpy as np
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
pi_frac = pp.formatters.FractionOfPi()
pi_frac_formatter = FuncFormatter(pi_frac.latex)

# ---------------------------------------------------------------------------- #


def plot_line_cmap(ax, x, cbar=True, **kws):
    nd = x.shape[0]
    assert nd in (2, 3)

    Lines = LineCollection if nd == 2 else Line3DCollection
    lc = Lines(fold.fold(x.T, 2, 1, pad=False), **kws)

    ax.add_collection(lc)
    ax.auto_scale_xyz(*x)

    if cbar:
        cb = ax.figure.colorbar(lc,
                                ax=ax, pad=0.03,
                                ticks=np.linspace(0, np.pi, 7),
                                format=pi_frac_formatter)
        cb.ax.set_ylabel(r'$\theta\ [rad]$', rotation=0, labelpad=20, usetex=True)

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


def plot_isopotential_solution(roche, axes, r, theta, phi, u, radii, primary,
                               cmap='jet'):
    """
    Helper for plotting isopotential solutions.
    """

    secondary = not primary
    invert = (-1, 1)[primary]
    mu = secondary + invert * roche.u.mu
    centre = 0  # getattr(roche.u, f'r{(2, 1)[primary]}').x / roche.u.a
    rmax = invert * (roche.u.l1.x / roche.u.a) - mu + 1

    # segments = np.empty((*u.shape, 2))
    # segments[..., 0] = r
    # segments[..., 1] = u

    # plot
    ax, ax1 = axes

    # roche lobe
    x, y, z = sph2cart(radii, phi, theta)
    x = invert * x + centre

    # radial curve
    ax1.plot(x, y, 'x', color='orangered', ms=3, zorder=10)
    lc, cb = plot_line_cmap(ax1, np.array([x, y]),
                            cbar=True, array=theta, cmap=cmap)
    ax1.plot(invert * rmax + centre, 0, 'r*')
    ax1.plot(centre, 0, 'g+')

    # ax1.plot(x, y)
    ax1.grid()
    ax1.set_xlabel(r'$x\ [a]$', usetex=True)
    ax1.set_ylabel(r'$y\ [a]$', usetex=True)
    # ax1.set_aspect('equal')

    # Potentials (radial)
    lc, _ = plot_line_family(ax, r, u, cbar=False, array=theta, cmap=cmap)
    ax.plot(radii, np.zeros_like(radii), 'mx')
    ax.plot((roche.u.l1 - roche.u.r1).x, 0, 'r*')

    ax.grid()
    ax.set_xlabel(r'$r\ [a]$', usetex=True)
    ax.set_ylabel(
        rf'$\Psi(r,\theta,\phi={pi_frac.latex(Fraction(phi/π)).strip("$")})\ '
        r'\left[\frac{GM}{2a}\right]$',
        usetex=True)
