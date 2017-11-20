import numpy as np

from .misc import sph2cart
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
    pass

def Nauenberg(M):
    """

    Parameters
    ----------
    M

    Returns
    -------
    R - white dwarf radius

    References
    ----------
    "Analytic Approximations to the Mass-Radius Relation and Energy of Zero-Temperature Stars"
    Nauenberg (1972)
    http://adsabs.harvard.edu/full/1972ApJ...175..417N
    """

    musol = 0.615 # solar mean atomic weight
    M3 = 5.816 / (musol ** 2.0)
    M_M3 = M / M3
    R = (0.0225 / musol) * ((1.0 - M_M3 ** 0.75) ** 0.5) / M_M3 ** (1. / 3)
    return R


class WhiteDwarf():
    def __init__(self, Msol, centre=(0, 0, 0)):
        """M - solar masses"""
        #
        self.R = Rsol = Nauenberg(Msol)
        self.centre = np.array(centre, ndmin=3).T

        # in cgs
        # M = Msol * M_sun
        # R = Rsol * R_sun

    def plot_wireframe(self, ax, res=25, **kw):
        r = np.ones(res)
        phi = np.linspace(0, 2 * np.pi, res)
        theta = np.linspace(0, np.pi, res)
        x, y, z = sph2cart(r, theta, phi, 'grid')
        return ax.plot_wireframe(x, y, z, **kw)

    def plot_surface(self, ax, res=25, **kw):
        r = np.ones(res)
        phi = np.linspace(0, 2 * np.pi, res)
        theta = np.linspace(0, np.pi, res)
        x, y, z = sph2cart(r, theta, phi, 'grid') + self.centre
        return ax.plot_surface(x, y, z, **kw)
