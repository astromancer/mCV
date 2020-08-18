"""
Methods for solving and plotting Roche lobe surfaces in 2d and 3d.
"""

# TODO: note the basic equations + references here

# Both stars treated as point masses.

# TODO: I'm sure others have published more recent work on solving these
# equations.  Do lit review and harvest best methods
# TODO: allow scale unit to be specified: eg. a, R1, R2, km, m etc...
# TODO: improve documentation

# TODO: options for which coordinate system to return values in
# TODO:

# todo: Seidov 20xx
# http://iopscience.iop.org.ezproxy.uct.ac.za/article/10.1086/381315/pdf
# def q_of l1(q):
# def q_of_l2(q):
# def q_of_l3(q):

import numbers
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G
# from scipy.optimize import fmin #Nelder-Mead simplex algorithm
from astropy.utils import lazyproperty
from scipy.optimize import brentq  # root finding

from recipes.transforms import pol2cart, rotation_matrix_2d

from mCV.utils import duplicate_if_scalar

# Default altitudinal and azimuthal resolution for RocheLobes
RESOLUTION = namedtuple('Resolution', ('alt', 'azim'))(30, 20)

#
LagrangePoint = namedtuple('LagrangePoint', ('x', 'y'))
LagrangePoint3D = namedtuple('LagrangePoint3D', ('x', 'y', 'z'))


def semi_major_axis(p, m1, m2):
    """
    Calculate the semi major axis from the orbital period and masses

    Parameters
    ----------
    p: float
        The orbital period
    m1
    m2

    Returns
    -------

    """

    # TODO: check if p, m1, m2 have units. if not, assume SI, use G value so
    # that the result does not have erroneous units.

    return np.cbrt(p * p * G * (m1 + m2) / (4.0 * np.pi * np.pi))


def L1(q, xtol=1.e-9):
    """
    Inner Lagrange points in units of binary separation a from origin CoM

    Parameters
    ----------
    q
    xtol

    Returns
    -------

    """
    return RochePotential(q).l1


def binary_potential_com(q, x, y, z):
    """
    Expresses the binary potential in Cartesian coordinates with origin at the
    CoM

    Parameters
    ----------
    q
    x
    y
    z

    Returns
    -------

    """

    x, y, z = map(np.asarray, (x, y, z))
    mu = 1. / (q + 1.)

    y2 = y * y
    z2 = z * z
    yz2 = y2 + z2  # for efficiency, only calculate these once

    return -(mu / np.sqrt((x + 1 - mu) ** 2 + yz2)
             + (1 - mu) / np.sqrt((x - mu) ** 2 + yz2)
             + 0.5 * (x * x + y2))


def _binary_potential_polar_com(r, theta, mu, psi0):
    r2 = r ** 2
    rcosθ = r * np.cos(theta)
    _μ1 = 1 - mu
    return -(mu / np.sqrt(r2 - 2 * _μ1 * rcosθ + _μ1 * _μ1) +
             _μ1 / np.sqrt(r2 - 2 * mu * rcosθ + mu * mu) +
             0.5 * r2
             - psi0)


def _binary_potential_polar1(r, theta, mu, psi0):
    """
    Expresses the binary potential in polar coordinates centred on the
    primary point mass.

    Parameters
    ----------
    r: {float, array-like}
        radial distance from primary
    mu: float
        location of secondary wrt CoM
    theta:  {float, array-like}
        azimuthal angle
    psi0: float
        Reference value of gravitational potential. Eg. Potential at L1 for
        Roche lobe.

    Returns
    -------

    """
    r2 = r ** 2
    rcosθ = r * np.cos(theta)
    _μ1 = 1 - mu
    return -(mu / r  #
             + _μ1 / np.sqrt(r2 - 2 * rcosθ + 1)  #
             - _μ1 * rcosθ + 0.5 * _μ1 * _μ1 + 0.5 * r2
             - psi0)


def _binary_potential_polar2(r, theta, mu, psi0):
    return _binary_potential_polar1(r, theta, 1 - mu, psi0)


# def _binary_potential_polar2(r, theta, mu, psi0):
#     """
#     Expresses the binary potential in polar coordinates centered on the
#     secondary point mass.  Use to solve for the equipotential surfaces.
#
#     Parameters
#     ----------
#     r: {float, array-like}
#         radial distance from secondary point mass
#     mu: float
#         location of secondary wrt CoM
#     theta:  {float, array-like}
#         azimuthal angle
#     psi0: float
#         Reference value of gravitational potential.
#
#     Returns
#     -------
#
#     """
#
#     # FIXME: this is identical to the above equation with mu = 1 - mu and
#     # theta = theta[::-1]
#
#     r2 = r ** 2
#     rcosθ = r * np.cos(theta)
#     return (1 - mu) / r + 0.5 * r2 \
#            + mu / np.sqrt(r2 + 2 * rcosθ + 1) \
#            + mu * rcosθ + 0.5 * mu * mu + psi0
#     # note this is actualy -ψ (has the same roots)


def _lagrange_objective(x, mu):
    """
    This equation is derived from that giving the force on a test mass in
    the roche potential within the orbital plane and along the line of
    centres.

    # TODO give equation here

    It is more amenable to numerical methods than the full binary
    potential equation. The 3 unique roots of this equation are the
    L3 < L1 < L2 Lagrangian points given in units of the binary separation `a`.
    """

    assert mu >= 0

    x1 = (x - mu)  # distance from primary
    x2 = (x - mu + 1)  # distance from secondary
    x1sq = x1 ** 2
    x2sq = x2 ** 2
    return (+ x * x1sq * x2sq
            - mu * np.sign(x2) * x1sq
            - (1 - mu) * np.sign(x1) * x2sq)


class UnderSpecifiedParameters(Exception):
    pass


class RochePotential(object):
    """
    Gravitational potential of two point masses, m1, m2, expressed in
    co-rotating (Cartesian) coordinate frame with origin at the center of mass
    (CoM) of the system.

    Distances and times are expressed internally in units of ``a = |r1| + |r2|'',
    the semi-major axis of the orbit, and ``P'' the orbital period.  The mass
    ratio ``q = m1 / m2'' uniquely defines the Roche potential in this frame.

    Note that some authors use the definition q = m2/m1.  If you prefer this
    definition, use the class method ``RochePotential.q_is_m2_over_m1()'' to
    switch to your preferred definition before you do any computation.


    """

    # TODO xyz, rθφ.  potential expressed in different coordinate systems

    # default tolerance on Lagrange point solutions
    xtol = 1.e-9

    # Mass ratio definition. This tells the class how to interpret init
    # parameter q:      q = m1 / m2 if True;        q = m2 / m1 if False
    _q_is_m1_over_m2 = False

    @classmethod
    def q_is_m1_over_m2(cls):
        cls._q_is_m1_over_m2 = False

    @classmethod
    def q_is_m2_over_m1(cls):
        cls._q_is_m1_over_m2 = True

    # alternate constructor
    @classmethod
    def from_masses(cls, m1, m2):
        """

        Parameters
        ----------
        m1: float
            The primary mass
        m2: float
            The secondary mass


        Returns
        -------

        """
        if cls._q_is_m1_over_m2:
            return cls(m1 / m2)
        else:
            return cls(m2 / m1)

    # def set_scale(self, a, P):
    #     """
    #     Set the distance and time scales from the value of the
    #     semi-major axis and the orbital period.  All computations are done
    #     scale free internally, but the returned results will be scaled to the
    #     values given here
    #
    #     Parameters
    #     ----------
    #     a
    #     P
    #
    #     Returns
    #     -------
    #
    #     """

    def __init__(self, q=None, m1=None, m2=None, a=None, P=None):
        """
        Create a Roche potential from various parameterisations. The class
        can be initialized by specifying any minimal subset of the the
        following:

        Parameters
        ----------
        q: float
            The mass ratio
        m1: float
            Primary mass
        m2: float
            Secondary mass
        a: float
            Orbital semi-major axis
        P: float
            The orbital Period
        """

        # if one of the masses are not given, we need either the orbital period
        # and the semi-major axis
        if None in (m1, m2):
            if q is None:
                if None in (a, P):
                    raise UnderSpecifiedParameters()
                    #
                elif not ((m1 is None) and (m2 is None)):  # not both None
                    # use a and P along with the given mass to compute the other
                    # mass using Kelper's third law. Also set the scale for
                    # results.
                    # total mass
                    m = (4 * np.pi * np.pi) * (a ** 3) / (G * P * P)
                    if m1 is None:
                        m1 = m - m2
                    else:
                        m2 = m - m1
                else:
                    raise UnderSpecifiedParameters()

                q = self.get_q(m1, m2)

        else:  # both m1 and m2 given
            q = self.get_q(m1, m2)

        if self._q_is_m1_over_m2:
            # internal definition used is `q = m2 / m1`
            self.q = 1. / q  # this will call `q.setter'
        else:
            self.q = q  # this will call `q.setter'

        # Potential at L1
        self.psi0 = self(*self.l1, 0)

        # self.omega        # Orbital angular velocity

    def __call__(self, x, y, z):
        """
        Evaluate potential at Cartesian (x, y, z) coordinate.
        """

        # TODO: units??
        return self.psi(x, y, z)

    @property
    def q(self):
        """mass ratio primary to secondary"""
        if self._q_is_m1_over_m2:
            return self._q
        return 1. / self._q

    @q.setter
    def q(self, q):
        if q <= 0:
            raise ValueError("Mass ratio 'q' should be a positive integer.")

        q = float(q)
        if self._q_is_m1_over_m2:
            self._q = q
        else:
            # internal definition used is `q = m2 / m1`
            self._q = 1 / q

        # reset all the lazy properties that depend on q
        del (self.mu, self.l1, self.l2, self.l3, self.l4, self.l5, self.psi0)

    def q_is(self):
        return 'm1/m2' if self._q_is_m1_over_m2 else 'm2/m1'

    def get_q(self, m1, m2):
        """Compute the mass ratio from the masses"""
        return (m1 / m2) if self._q_is_m1_over_m2 else (m2 / m1)

    @lazyproperty
    def mu(self):
        """position of the secondary wrt CoM in units of a"""
        return 1. / (self._q + 1.)

    @property
    def r1(self):
        """position of the primary wrt CoM in units of a"""
        return self.mu - 1

    r2 = mu

    @lazyproperty
    def psi0(self):
        """Gravitational potential at L1"""
        return self(*self.l1, 0)

    # def centrifugal(self, x, y, z):  #
    #     """
    #     Potential due to centrifugal force at Cartesian position (x, y, z).
    #     """
    #     # self.omega ** 2 *
    #     return 0.5 * (x * x + y * y)
    #
    # def primary(self, x, y, z):  # psi_m1  # of_primary  # of_m1
    #     """
    #     Potential due to primary at Cartesian position (x, y, z).
    #     """
    #     mu = self.mu
    #     return -mu / np.sqrt((x + 1 - mu) ** 2 + y * y + z * z)
    #
    # def secondary(self, x, y, z):
    #     """
    #     Potential due to secondary at Cartesian position (x, y, z).
    #     """
    #     mu = self.mu
    #     return (mu - 1.) / np.sqrt((x - mu) ** 2 + y * y + z * z)

    def total(self, x, y, z):
        """
        Gravitational + centrifugal potential of binary system in rotating
        coordinate frame (Cartesian)
        Takes xyz grid in units of the semi-major axis (a = r1 + r2)
        """

        # x, y, z = map(np.asarray, (x, y, z))
        _phi = binary_potential_com(self._q, x, y, z)
        # k = -G*(M1+M2)/(2*(r1+r2))
        # k = -1
        return _phi

    psi = total  # fixme: will not inherit

    def _solve_lagrange123(self, interval, xtol=xtol):
        """
        Solve for L1, L2, or L3 (depending on given interval)

        Parameters
        ----------
        interval

        Returns
        -------

        """
        lx = brentq(_lagrange_objective, *interval, (self.mu,), xtol)
        return LagrangePoint(lx, 0)

    @lazyproperty
    def l1(self):
        δ = 1e-6
        interval = (self.mu - 1 + δ, self.mu - δ)
        return self._solve_lagrange123(interval)  # TODO scale

    @lazyproperty
    def l2(self):
        δ = 1e-6
        interval = self.mu + δ, 2
        return self._solve_lagrange123(interval)  # TODO scale

    @lazyproperty
    def l3(self):
        δ = 1e-6
        interval = -2, self.r1 - δ
        return self._solve_lagrange123(interval)  # TODO scale

    @lazyproperty
    def l4(self):
        # L4 has analytic solution  # TODO scale
        return LagrangePoint(self.mu - 0.5, np.sqrt(3) / 2)

    @lazyproperty
    def l5(self):
        # L5 has analytic solution  # TODO scale
        return LagrangePoint(self.mu - 0.5, -np.sqrt(3) / 2)

    @lazyproperty
    def lagrangians2D(self):
        return np.c_[[getattr(self, 'l%i' % i)  # TODO scale, performance
                      for i in range(1, 6)]]

    @lazyproperty
    def lagrangians3D(self):
        x, y = self.lagrangians2D.T
        z = self.psi(x, y, 0)
        return np.c_[x, y, z]  # TODO scale, performance


# class _PlottingMixin(object):
# res_default = rres, _ = RocheSolver1.res_default


class EquipotentialSolver(RochePotential):
    pass  # TODO


class RocheSolver(RochePotential):  #
    """
    Solver for the Roche equipotential surfaces.
    """

    def _solver(self, theta, level=None, primary=False):
        """
        Solve for the shape of the Roche lobe in polar coordinates centred
        on the star.

        Parameters
        ----------
        theta
        level:  float, optional
            Value of the potential at which to draw the contours.  Default is
            to use the value of the potential at the inner Lagrange point L1.

        Returns
        -------

        """

        #
        if level is None:
            level = -self.psi0

        # if level > self.psi0:
        #     # ignore `primary' parameter
        #     'use binary potential CoM'

        # use symmetry to solve for both lobes with primary equation
        if primary:  # primary
            mu = self.mu
            invert = 1
        else:  # secondary
            mu = 1 - self.mu
            invert = -1

        # get interval on radius containing root
        rmin = 1e-6  # will not work if Roche lobe is smaller than this
        # (that is, for a very extreme mass ratio)
        rmax = invert * self.l1.x - mu + 1  # distance from point mass to L1

        # Since the solver uses a root finding algorithm, it cannot be used to
        # solve for the equipotential point at L1 (which is a saddle point)
        l1 = (theta == 0)

        r = np.ma.empty_like(theta)
        for i in np.where(~l1)[0]:
            # noinspection PyTypeChecker
            r[i] = brentq(
                    _binary_potential_polar1, rmin, rmax, (theta[i], mu, level))

        # sub exact L1 point
        # radial distance from point mass location (center of star) to L1 point
        r[l1] = rmax

        # coordinate conversion
        x, y = pol2cart(r, theta)
        x += mu - 1  # shift to CoM
        x *= invert

        return x, y

    def solve(self, res=RESOLUTION.alt, theta=None, reflect=True, primary=False,
              scale=1.):
        """ """

        if theta is None:
            theta = np.linspace(0, np.pi, res)
        else:
            # if `theta' array given don't reflect unless explicitly asked to
            # do so by `reflect=True`
            theta = np.atleast_1d(theta)

        # solve contour
        x, y = self._solver(theta, -self.psi0, primary)
        x *= scale
        y *= scale

        # use reflection symmetry to construct the full contour
        if reflect:
            return np.r_[x, x[::-1]], np.r_[y, -y[::-1]]

        return x, y

    def solve_contour(self, mu, theta, level, primary=False):
        # FIXME get this working
        if level > self.psi0:
            fun = _binary_potential_polar_com
        else:
            pass

        r = np.ma.empty_like(theta)
        for i in enumerate(theta):
            # noinspection PyTypeChecker
            r[i] = brentq(
                    _binary_potential_polar1, rmin, rmax, (theta[i], mu, level))

    def make3d(self, res, primary, scale=1., phi=None):
        """
        Solve 1D roche lobe, and use rotational symmetry (along the line of
        centres) to generate 3D Roche lobe surface.
        """

        resr, resaz = duplicate_if_scalar(res)
        if phi is None:
            phi = np.linspace(0, 2 * np.pi, resaz)
        else:
            phi = np.asarray(phi)
            resaz = len(phi)

        x, y = self.solve(resr, reflect=False, primary=primary, scale=scale)
        z = np.zeros_like(x)

        # make 3D
        X = np.tile(x, (resaz, 1))

        # Make use of the rotational symmetry around the x-axis to construct
        # the Roche lobe in 3D from the 2D solution

        rm = rotation_matrix_2d(phi)
        Y, Z = np.einsum('ijk,jl->ikl', rm, [y, z])
        return X, Y, Z

    # def solve(self, res=30, full=True):
    #     raise NotImplementedError

    # def _solver(self, func, interval, Theta, level=None):
    #     """
    #     Solve for the shape of the Roche lobes in polar coordinates centred
    #     on the star.
    #
    #     Parameters
    #     ----------
    #     func
    #     interval
    #     Theta
    #     level:  float, optional
    #         Value of the potential at which to draw the contours.  Default is
    #         to use the value of the potential at the inner Lagrange point L1.
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     #
    #     if level is None:
    #         level = self.psi0
    #
    #     r = np.ma.empty_like(Theta)
    #     for i, theta in enumerate(Theta):
    #         r[i] = brentq(func, *interval, (theta, mu, level))
    #
    #     return r
    #
    #     # # TODO: do this better. first transform based on q, then deal with
    #     # # special cases of theta, then solve for remaining
    #     #
    #     # # Theta = np.linspace(0, np.pi, res)
    #     # A = np.ma.empty_like(Theta)
    #     # for i, theta in enumerate(Theta):
    #     #     try:
    #     #         # WARNING: problems at θ = 0: inaccurate
    #     #         # and θ = π: ValueError: f(a) and f(b) must have different signs
    #     #         # fixme: maybe deal explicitly with these instead of trying
    #     #         # to catch at every computation.  Then you can return array
    #     #         # instead of masked array which seems unnecessary
    #     #         A[i] = brentq(func, *interval, (self.mu, theta, self.psi0))
    #     #     except ValueError as err:
    #     #         warnings.warn(
    #     #                 'ValueError %s\nMasking element %i (theta = %.3f)' %
    #     #                 (str(err), i, theta))
    #     #         A[i] = np.ma.masked  # exceptions are masked for robustness
    #     # return Theta, A


# class RocheSolver1(RocheSolver):
#     rres, _ = RocheSolver.res_default

# def _vec_solver(self, func, interval, theta, mu, level):
#     for i in np.where(~l1)[0]:
#         # noinspection PyTypeChecker
#         r[i] = brentq(
#                 _binary_potential_polar1, rmin, rmax, (theta[i], mu, level))


# class RocheSolver2(RocheSolver):
#     rres, _ = RocheSolver.res_default
#
#     def solve(self, res=30, full=True):
#         pass


# def solve(self, res=rres, reflect=None, theta=None):
#     """Returns the secondary Roche Lobe in 2D"""
#     #
#
#     if theta is None:
#         theta = np.linspace(0, np.pi, res)
#         reflect = True
#     else:
#         # if `theta' array given don't reflect unless explicitly asked to
#         #  do so by `reflect=True`
#         theta = np.atleast_1d(theta)
#         reflect = reflect or False
#
#     # Since the solver uses a root finding algorithm, it cannot be used to
#     # solve for the equipotential point at L1 (which is a saddle point)
#     l1 = (theta == np.pi)
#
#     # solve contour
#     rmax = self.mu - self.l1.x
#     r = np.empty_like(theta)
#     r[~l1] = self._solver(
#             _binary_potential_polar2, (1e-6, rmax), theta[~l1])
#     # sub exact L1 point
#     r[theta == np.pi] = self.mu - self.l1.x
#
#     # coordinate conversion
#     x, y = pol2cart(r, theta)
#     x += self.mu  # shift to CoM
#
#     if full:
#         # use reflection symmetry
#         return np.r_[x[::-1], x], np.r_[y[::-1], -y]
#
#     return x[::-1], y[::-1]


class RocheLobe(RocheSolver):

    def __init__(self, q, primary, scale=1):
        # if isinstance(q_or_solver, numbers.Real):
        #     # noinspection PyTypeChecker
        #     self.solver = RocheSolver(q_or_solver)
        # elif isinstance(q_or_solver, RocheSolver):
        #     self.solver = q_or_solver
        # else:
        #     raise ValueError('Bad mass ratio value or solver')

        RocheSolver.__init__(self, q)

        self.primary = bool(primary)
        self.scale = float(scale)

    def plot2d(self, ax=None, res=RESOLUTION.azim, **kws):
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

        x1, y1 = self.solve(res, primary=self.primary, scale=self.scale)
        # todo memoize solver for efficiency ??
        lobe, = ax.plot(x1, y1, **kws)

        return ax, lobe

    plot2D = plot2d  # FIXME not inherited if overwritten

    def plot_wireframe(self, ax=None, res=RESOLUTION, **kw):
        """

        Parameters
        ----------
        ax
        res : {int, tuple}
                resolution in r and/or theta
        kw

        Returns
        -------

        """

        poly3 = ax.plot_wireframe(
                *self.make3d(res, self.primary, self.scale),
                **kw)
        return ax, poly3

    def plot_surface(self, ax=None, res=RESOLUTION, **kw):
        """
        res : {int, tuple}
                resolution in r and/or theta
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        x, y, z = self.make3d(res, self.primary, self.scale)
        poly3 = ax.plot_surface(x, y, z, **kw)

        return ax, poly3


class RocheLobes(object):
    # res_default = rres, _ = RocheSolver1.res_default

    def __init__(self, q, scale=1):
        """
        """

        # solver = RocheSolver(q)
        self.primary = RocheLobe(q, True, scale)  # accretor
        self.secondary = RocheLobe(q, False, scale)  # donor
        self.scale = float(scale)
        self._axes = None

    @property
    def axes(self):
        if self._axes is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            self._axes = ax
        return self._axes

    def plot2d(self, ax=None, res=RESOLUTION.azim, **kws):

        # plot both lobes on same axes
        if ax is None:
            fig, ax = plt.subplots()

        _, pri = self.primary.plot2d(ax, res,
                                     label='Primary',
                                     color='g')
        _, sec = self.secondary.plot2d(ax, res,
                                       label='Secondary',
                                       color='orange')

        mu = self.primary.mu
        lagrangeMarks, = ax.plot(*self.primary.lagrangians2D.T, 'm.')
        centre = np.multiply([mu, mu - 1], self.scale)
        centreMarks, = ax.plot(centre, [0, 0], 'g+', label='centres')
        comMarks, = ax.plot(0, 0, 'rx', label='CoM')
        artists = pri, sec, lagrangeMarks, comMarks, centreMarks

        ax.grid()

        return ax, artists

    plot2D = plot2d

    def label_lagrangians(self, ax, txt_offset=(0.02, 0.02), **kws):

        # Label lagrangian points
        texts = []
        for i, xy in enumerate(self.primary.lagrangians2D):
            text = ax.annotate('L%i' % (i + 1), xy, xy + txt_offset, **kws)
            texts.append(text)

        return texts

    def plot_wireframe(self, ax=None, res=RESOLUTION, **kws):
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

        poly1 = self.primary.plot_wireframe(ax, res, **kws)
        poly2 = self.secondary.plot_wireframe(ax, res, **kws)

        ax.set_aspect('equal')
        ax.figure.tight_layout()

        return ax, (poly1, poly2)

    def plot_surface(self, ax=None, res=RESOLUTION, **kws):
        """
        res : {int, tuple}
                resolution in r and/or theta
        """
        if ax is None:
            ax = self.axes

        poly1 = self.primary.plot_surface(ax, res, **kws)
        poly2 = self.secondary.plot_surface(ax, res, **kws)

        ax.set_aspect('equal')
        ax.figure.tight_layout()

        return ax, (poly1, poly2)

    plot3d = plot3D = plot_surface


if __name__ == '__main__':
    # test primary solver
    fig, ax = plt.subplots()
    for q in np.linspace(0, 1, 11):
        try:
            roche = RocheLobes(q)
            x1, y1 = roche.primary()
            pri, = ax.plot(x1, y1, label='q = %.1f' % q)
            print(q)
        except Exception as e:
            print(q, e)
        ax.legend()
