import numpy as np

from recipes.transforms import sph2cart
from astropy import units as u
from astropy.constants import M_sun, R_sun


def Chandrasekhar(M):
    """

    Parameters
    ----------
    M

    Returns
    -------

    References
    ----------
    An introduction to the Study of stellar structure
    Chandrasekhar, S.
    http://adsabs.harvard.edu/abs/1939C&T....55..412C

    """
    raise NotImplementedError


def Nauenberg(M):
    """
    The Nauenberg Mass-Radius relation for White Dwarfs.

    Parameters
    ----------
    M

    Returns
    -------
    R - white dwarf radius in solar radii

    References
    ----------
    "Analytic Approximations to the Mass-Radius Relation and Energy of Zero-Temperature Stars"
    Nauenberg (1972)
    http://adsabs.harvard.edu/full/1972ApJ...175..417N
    """

    mu = 2  # the average molecular weight per electron of the star.
    # He-4 C-12 and O-16 which predominantly compose white dwarf all have atomic
    # number equal to half their atomic weight, one should take μe equal to 2
    M3 = 5.816 / mu ** 2
    mm3 = np.cbrt(M / M3)
    return (0.0225 / mu) * np.sqrt(1. - mm3 ** 4) / mm3


def NauenbergStd(M, Mstd):
    # linear propagation of uncertainty
    mu = 2
    M3 = 5.816 / mu ** 2
    k = 0.0225 / mu
    mcb = np.cbrt(M / M3)
    return abs(k / 18 / M3 * (1 - 2 / mcb) / np.sqrt(1 - mcb) / mcb ** 5 * Mstd)


def carvalho_M(r):
    c = [20.86, 0.66, 2.48e-5, -2.43e-5]
    np.polyval(c, r)


class WhiteDwarf:
    def __init__(self, Msol, centre=(0, 0, 0)):
        """M: WD mass in units of (solar mass)"""
        #
        self.R = Nauenberg(Msol)  # unit of solar radii
        self.centre = np.array(centre, ndmin=3).T

        # in cgs
        # M = Msol * M_sun
        # R = Rsol * R_sun

    def get_data(self, res, scale):
        """generate spherical surface with radius scale around center"""
        r = np.ones(res) * scale
        phi = np.linspace(0, 2 * np.pi, res)
        theta = np.linspace(0, np.pi, res)
        return sph2cart(r, theta, phi, 'grid') + self.centre

    def plot2D(self, ax, scale=1, **kws):
        from matplotlib.patches import Circle

        # WD surface
        props = dict(ec='c', fc='none', lw=1)  # defaults
        props.update(**kws)
        cir = Circle(self.centre[:-1],
                     # FIXME: position and radius different units
                     self.R * scale, **props)
        ax.add_patch(cir)

    def plot_wireframe(self, ax, res=25, scale=1, **kws):
        return ax.plot_wireframe(*self.get_data(res, scale), **kws)

    def plot_surface(self, ax, res=25, scale=1, **kws):
        return ax.plot_surface(*self.get_data(res, scale), **kws)
