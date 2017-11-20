import warnings
from collections import namedtuple

import numpy as np
from scipy.optimize import brentq  # root finding
# from scipy.optimize import fmin #Nelder-Mead simplex algorithm

from astropy.utils import lazyproperty
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .misc import pol2cart, rotation_matrix_2D


def atleast_2(seq):
    """

    Parameters
    ----------
    seq : {number, array-like}

    Returns
    -------

    """
    # seq = np.atleast_1d(seq)
    if np.size(seq) == 1:
        seq = np.ravel([seq, seq])
    if np.size(seq) != 2:
        raise ValueError('Input should be of size 1 or 2')
    return seq


def _solve_lagrange(x, mu):
    """
    This equation is derived from that giving the force on a test mass in
    the roche potential within the orbital plane and along the line of
    centres.  It is more amenable to numerical methods than the full binary
    potential equation. The 3 unique roots of this equation are the L3, L1,
    and L2 Lagrangian points.
    """
    x1 = (x - mu)  # distance from primary
    x2 = (x - mu + 1)  # distance from secondary
    x1sq = x1 ** 2
    x2sq = x2 ** 2
    return x * x1sq * x2sq \
           - mu * np.sign(x2) * x1sq \
           - (1 - mu) * np.sign(x1) * x2sq


def _binary_potential_polar1(r, mu, theta, psi0):
    """
    Expresses the binary potential in polar coordinates centred on the primary.

    Parameters
    ----------
    r: {float, array-like}
        radial distance from primary
    mu: float
        location of secondary wrt CoM
    theta:  {float, array-like}
        azimuthal angle
    psi0: float
        Value of gravitational potential at L1

    Returns
    -------

    """
    r2 = r ** 2
    rcos_t = r * np.cos(theta)
    return mu / r + 0.5 * r2 \
           + (1 - mu) / np.sqrt(r2 - 2 * rcos_t + 1) \
           + (mu - 1) * rcos_t + 0.5 * (mu - 1) ** 2 + psi0


def _binary_potential_polar2(r, mu, theta, psi0):
    """
    Expresses the binary potential in polar coordinates centred on the secondary.

    Parameters
    ----------
    r: {float, array-like}
        radial distance from primary
    mu: float
        location of secondary wrt CoM
    theta:  {float, array-like}
        azimuthal angle
    psi0: float
        Value of gravitational potential at L1

    Returns
    -------

    """

    r2 = r ** 2
    rcos_t = r * np.cos(theta)

    return mu / np.sqrt(r2 + 2 * rcos_t + 1) \
           + (1 - mu) / r + 0.5 * r2 \
           + mu * rcos_t + 0.5 * mu * mu + psi0


LagrangePoint = namedtuple('LagrangePoint', ('x', 'y'))


class BinaryPotential():
    # TODO: Consider ThreeBody base class?? then use this class as a inherited convenience
    # TODO: efficiency ==> use lazyproperties for L points

    # @classmethod
    # def from_q()

    # default tolerance on Lagrange point solutions
    xtol = 1.e-9

    def __init__(self, q):
        """
        Calculate the orbital parameters of a binary star. Internal units
        are in terms of the semi-major axis of the orbit.
        """

        # # TODO: a=None, m1=None, m2=None
        # The class can be
        #  initialized by specifying any two of the following:
        # q:   mass ratio
        # a:   semi-major axis
        #
        # (m1, m2):   component masses      #TODO
        # P:   orbital period               #TODO

        self.q = q

        # Potential at L1
        self.psi0 = self(*self.l1, 0)

        # self.omega        # Orbital angular velocity

    def __call__(self, x, y, z):
        """
        Evaluate gravitational potential at Cartesian (x, y, z) coordinate.
        """

        # TODO: units??
        return self.gravitational(x, y, z)

    @property
    def q(self):
        """mass ratio primary to secondary"""
        return self._q

    @q.setter
    def q(self, q):
        if q <= 0:
            raise ValueError("Mass ratio 'q' should be a positive integer.")

        self._q = float(q)

        # reset all the lazy propertios that depend on q
        del (self.mu, self.l1, self.l2, self.l3, self.l4, self.l5, self.psi0)

    @lazyproperty
    def mu(self):
        """position of the secondary wrt CoM in units of a"""
        return 1. / (self.q + 1.)

    def r1(self):
        """position of the primary wrt CoM in units of a"""
        return self.mu - 1

    r2 = mu

    @lazyproperty
    def psi0(self):
        """Gravitational potential at L1"""
        return self(*self.l1, 0)

    def centrifugal(self, x, y, z):
        """
        Potential due to centrifugal force at Cartesian position (x, y, z).
        """
        # self.omega ** 2 *
        return 0.5 * (x * x + y * y)

    def primary(self, x, y, z):
        """
        Potential due to primary at Cartesian position (x, y, z).
        """
        mu = self.mu
        return -mu / np.sqrt((x + 1 - mu) ** 2 + y * y + z * z)

    def secondary(self, x, y, z):
        """
        Potential due to secondary at Cartesian position (x, y, z).
        """
        mu = self.mu
        return (mu - 1.) / np.sqrt((x - mu) ** 2 + y * y + z * z)

    def gravitational(self, x, y, z):
        """
        graviatational potential of binary system in rotating coordinate
        frame (Cartesian)
        Takes xyz grid in units of the semi-major axis (a = r1 + r2)
        """
        x, y, z = map(np.asarray, (x, y, z))
        mu = self.mu

        # q = self.q
        # r1, r2 = self.r1, self.r2

        y2 = y * y
        z2 = z * z
        yz2 = y2 + z2  # for efficiency, only calculate these once

        _phi = mu / np.sqrt((x + 1 - mu) ** 2 + yz2) \
               + (1 - mu) / np.sqrt((x - mu) ** 2 + yz2) \
               + 0.5 * (x * x + y2)
        # k = -G*(M1+M2)/(2*(r1+r2))
        # k = -1
        return -_phi

    psi = gravitational

    def _solve_lagrange123(self, interval, xtol=xtol):
        """
        Solve for L1, L2, or L3 (depending on given interval)

        Parameters
        ----------
        interval

        Returns
        -------

        """
        lx = brentq(_solve_lagrange, *interval, (self.mu,), xtol)
        return LagrangePoint(lx, 0)

    @lazyproperty
    def l1(self):
        mu = self.mu
        delta = 1e-6
        interval = mu - 1 + delta, mu - delta
        return self._solve_lagrange123(interval)

    @lazyproperty
    def l2(self):
        mu = self.mu
        delta = 1e-6
        interval = mu + delta, 10 * mu
        return self._solve_lagrange123(interval)

    @lazyproperty
    def l3(self):
        r1 = self.q * self.mu
        delta = 1e-6
        interval = -10 * r1, -r1 - delta
        return self._solve_lagrange123(interval)

    @lazyproperty
    def l4(self):
        return LagrangePoint(self.mu - 0.5, np.sqrt(3) / 2)

    @lazyproperty
    def l5(self):
        return LagrangePoint(self.mu - 0.5, -np.sqrt(3) / 2)

    @lazyproperty
    def lagrangians2D(self):
        return np.c_[[getattr(self, 'l%i' % i)
                      for i in range(1, 6)]]

    @lazyproperty
    def lagrangians3D(self):
        x, y = self.lagrangians2D.T
        z = self.psi(x, y, 0)
        return np.c_[x, y, z]



class RocheLobeBase(BinaryPotential):

    # default resolution
    res_default = rres, ares = (30, 20)

    def solve(self, res=30, full=True):
        raise NotImplementedError

    def _solver(self, func, interval, res):
        """
        Solve for the shape of the Roche Lobes in polar coordinates centred on the star.
        """
        Theta = np.linspace(0, np.pi, res)
        A = np.ma.empty(res)
        for i, theta in enumerate(Theta):
            try:
                A[i] = brentq(func, *interval, (self.mu, theta, self.psi0))
            except ValueError as err:
                warnings.warn('ValueError %s\nMasking element %i' % (str(err), i))
                A[i] = np.ma.masked  # exceptions are masked for robustness
        return Theta, A

    def solve3D(self, res, scale=1.):
        """
        Solve 1D roche lobe, and use rotational symmetry (along the line of centres)
        to generate 3D Roche lobe surface.
        """

        resr, resaz = atleast_2(res)
        x, y = self.solve(resr, full=False)
        z = np.zeros_like(x)

        # scale x coordinate
        x *= scale
        # make 3D
        X = np.tile(x, (resaz, 1))

        # Make use of the rotational symmetry around the x-axis to construct
        # the Roche lobe in 3D from the 2D solution
        radians = np.linspace(0, 2 * np.pi, resaz)
        rm = rotation_matrix_2D(radians)
        Y, Z = np.einsum('ijk,jl->ikl', rm, [y, z]) * scale
        return X, Y, Z

    # def plot(self):

    def plot2D(self, ax=None, res=rres, scale=1., **kws):
        """

        Parameters
        ----------
        ax
        res
        scale

        Returns
        -------

        """
        if ax is None:
            fig, ax = plt.subplots()

        x1, y1 = self.solve(res)
        lobe, = ax.plot(x1, y1, **kws)

        return ax, lobe


    def plot_wireframe(self, ax=None, res=res_default, scale=1., **kw):
        """

        Parameters
        ----------
        ax
        res : {int, tuple}
                resolution in r and/or theta
        scale
        kw

        Returns
        -------

        """

        poly3 = ax.plot_wireframe(*self.solve3D(res, scale), **kw)
        return ax, poly3

    def plot_surface(self, ax=None, res=res_default, scale=1., **kw):
        """
        res : {int, tuple}
                resolution in r and/or theta
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        x, y, z = self.solve3D(res, scale)
        poly3 = ax.plot_surface(x, y, z, **kw)

        return ax, poly3



class PrimaryRocheLobe(RocheLobeBase):
    def solve(self, res=30, full=True):
        """ """
        rmax = self.l1.x - self.mu + 1
        theta, r = self._solver(_binary_potential_polar1, (1e-6, rmax), res)
        r[theta == 0] = self.l1.x - self.mu + 1

        # coordinate conversion
        x, y = pol2cart(r, theta)
        x += self.mu - 1  # shift to CoM

        if full:
            # use reflection symmetry
            return np.r_[x, x[::-1]], np.r_[y, -y[::-1]]

        return x, y


class SecondaryRocheLobe(RocheLobeBase):
    def solve(self, res=30, full=True):
        """Returns the secondary Roche Lobe in 2D"""
        rmax = self.l1.x
        theta, r = self._solver(_binary_potential_polar2, (1e-6, rmax), res)
        r[theta == np.pi] = self.mu - self.l1.x

        # coordinate conversion
        x, y = pol2cart(r, theta)
        x += self.mu  # shift to CoM

        if full:
            # use reflection symmetry
            return np.r_[x[::-1], x], np.r_[y[::-1], -y]

        return x[::-1], y[::-1]



class RocheLobe():

    rres, ares = res_default = RocheLobeBase.res_default

    def __init__(self, q):
        """
        """
        #BinaryPotential.__init__(self, q)

        self.primary = PrimaryRocheLobe(q)
        self.secondary = SecondaryRocheLobe(q)
        self._axes = None

    @property
    def axes(self):
        if self._axes is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            self._axes = ax
        return self._axes

    def plot2D(self, ax=None, res=rres, scale=1.):

        if ax is None:
            fig, ax = plt.subplots()

        _, pri = self.primary.plot2D(ax, res, scale, label='Primary')
        _, sec = self.secondary.plot2D(ax, res, scale, label='Secondary')

        mu = self.primary.mu
        lag, = ax.plot(*self.primary.lagrangians2D.T, 'm.')
        cenx = [mu, mu - 1]
        cent, = ax.plot(cenx, [0, 0], 'g+', label='centres')
        com, = ax.plot(0, 0, 'rx', label='CoM')
        artists = pri, sec, lag, com,  cent

        ax.grid()

        return ax, artists

    def label_lagrangians(self, ax, txtoff=(0.02, 0.02)):

        # Label lagrangian points
        texts = []
        for i, xy in enumerate(self.primary.lagrangians2D):
            text = ax.annotate('L%i' % (i + 1), xy, xy + txtoff)
            texts.append(text)

        return texts

    def plot_wireframe(self, ax=None, res=res_default, scale=1., **kws):
        """

        Parameters
        ----------
        ax
        res : {int, tuple}
                resolution in r and/or theta
        scale
        kw

        Returns
        -------

        """
        if ax is None:
            ax = self.axes

        poly1 = self.primary.plot_wireframe(ax, res, scale, **kws)
        poly2 = self.secondary.plot_wireframe(ax, res, scale, **kws)

        ax.set_aspect('equal')
        ax.figure.tight_layout()

        return ax, (poly1, poly2)

    def plot_surface(self, ax=None, res=res_default, scale=1., **kws):
        """
        res : {int, tuple}
                resolution in r and/or theta
        """
        if ax is None:
            ax = self.axes

        poly1 = self.primary.plot_surface(ax, res, scale, **kws)
        poly2 = self.secondary.plot_surface(ax, res, scale, **kws)

        ax.set_aspect('equal')
        ax.figure.tight_layout()

        return ax,  (poly1, poly2)

    plot3D = plot_surface


def L1(q, xtol=1.e-9):
    # convenience function
    return RocheLobe(q).l1


if __name__ == '__main__':
    # test primary solver
    fig, ax = plt.subplots()
    for q in np.linspace(0, 1, 11):
        try:
            roche = RocheLobe(q)
            x1, y1 = roche.primary()
            pri, = ax.plot(x1, y1, label='q = %.1f' % q)
            print(q)
        except Exception as e:
            print(q, e)
        ax.legend()