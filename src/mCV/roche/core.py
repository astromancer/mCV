"""
Methods for solving and plotting Roche lobe surfaces in 2d and 3d.
"""

# TODO: note the basic equations + references here

# Both stars treated as point masses.

# TODO: I'm sure others have published more recent work on solving these
# equations.  Do lit review and harvest best methods
# TODO: improve documentation


# todo: Seidov 20xx
# http://iopscience.iop.org.ezproxy.uct.ac.za/article/10.1086/381315/pdf
# def q_of l1(q):
# def q_of_l2(q):
# def q_of_l3(q):

# std
import inspect
import numbers
from collections import namedtuple

# third-party
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from astropy import units as u
from astropy.constants import G
from astropy.utils import lazyproperty
from astropy.units.quantity import Quantity
from astropy.coordinates import CartesianRepresentation

# local
from recipes import pprint as pp
from recipes.decorators import Decorator
from recipes.logging import LoggingMixin
from recipes.misc import duplicate_if_scalar
from recipes.transforms import pol2cart, rotation_matrix_2d


π = np.pi
_4π2 = 4 * π * π

Mo = u.M_sun
Ro = u.R_sun

# Verify units
verify_mass_unit = u.quantity_input(mass=['mass', 'dimensionless'])

# Default units for orbital Parameters
PARAM_DEFAULT_UNITS = dict(m=Mo, m1=Mo, m2=Mo, a=u.AU, p=u.hour)

# Plotting defaults
WIREFRAME_DEFAULTS = dict(color='c', lw=0.75, alpha=0.5)
LAGRANGE_3D_DEFAULTS = dict(color='darkgreen', marker='o', ms=5, ls='none',)

# Default altitudinal and azimuthal resolution for RocheLobes
RESOLUTION = namedtuple('Resolution', ('alt', 'azim'))(50, 35)


class apply_default_units(Decorator):
    def __init__(self, default_units=(), **kws):
        for _, unit in dict(default_units, **kws).items():
            assert isinstance(unit, (u.Unit, u.IrreducibleUnit))

        self.default_units = default_units

    def __call__(self, func):
        self.sig = inspect.signature(func)
        return super().__call__(func)

    def __wrapper__(self, func, *args, **kws):
        return func(**{
            name: (val * self.default_units[name]
                   if isinstance(val, (numbers.Real, np.ndarray)) else
                   val)
            for name, val in self.sig.bind(*args, **kws).arguments.items()
        })


def _check_units(mapping):
    for obj, kind in mapping.items():
        if isinstance(obj, Quantity):
            assert obj.unit.physical_type == kind

# def semi_major_axis(p, m1, m2):

#     params = dict(locals())
#     nrs = {key: isinstance(val, numbers.Real) for key, val in params.items()}
#     if all(nrs.values()):
#         # Assume p [yr], m [Mo] -> a [AU]
#         return np.cbrt(p * p * (m1 + m2)) * u.AU

#     # at least one of the input parameters is Quantity
#     return _semi_major_axis(**params)


@apply_default_units(p=u.yr, m1=Mo, m2=Mo)
@u.quantity_input(p='time', m1='mass', m2='mass')
def semi_major_axis(p, m1, m2) -> u.AU:
    """
    Calculate the semi major axis from the orbital period and masses.

    Parameters
    ----------
    p: float
        The orbital period
    m1
    m2

    Returns
    -------

    """
    return np.cbrt(p * p * G * (m1 + m2) / _4π2)


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
    # for efficiency, only calculate these once
    mu = 1. / (q + 1.)
    x_mu = x - mu
    y2 = y * y
    yz2 = y2 + z * z
    return -(mu / np.sqrt((x_mu + 1) ** 2 + yz2)
             + (1 - mu) / np.sqrt(x_mu ** 2 + yz2)
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

    x1 = (x - mu)        # distance from primary
    x2 = (x - mu + 1)    # distance from secondary
    x1sq = x1 ** 2
    x2sq = x2 ** 2
    return (+ x * x1sq * x2sq
            - mu * np.sign(x2) * x1sq
            - (1 - mu) * np.sign(x1) * x2sq)


class UnderSpecifiedParameters(Exception):
    """
    Exception raised when insufficient information provided to construct the
    BinaryParameters.
    """


class OverSpecifiedParameters(Exception):
    """
    Exception raised when conflicting parameters are provided to construct the
    BinaryParameters.
    """


class BinaryParameters(LoggingMixin):
    """
    Parameterization helper for binary star systems. Serves as a interpretation
    layer that checks validity of parameters and units of input combinations
    from the following set of parameters:
        q: float
            The mass ratio. The default definition is `q = m2 / m1`. This can be
            changed via the classmethod meth::q_is_m1_over_m2 prior to
            initialization.
        m: float, Quantity, optional
            Total mass of the binary.
        m1: float, Quantity, optional
            Primary mass.
        m2: float, Quantity, optional
            Secondary mass.
        a: float, Quantity, optional
            Orbital semi-major axis.
        P: float, Quantity, optional
            The orbital Period

    The minimal parameter set is
    >>> BinaryParameters(q=1)
    In this case, the internal length scale is *1a*, where *a* is the orbital
    semi-major axis and the internal mass scale in *1m* where *m* is the total
    mass of the binary. All results are returned in the centre of mass coordinates
    with spatial scale a.
    Initializing with mass scale:
    >>> BinaryParameters(q=0.5, m1=0.4) # default unit is assumed solar masses

    """
    # Mass ratio definition. This tells the class how to interpret init
    # parameter q:      q = m1 / m2 if True;        q = m2 / m1 if False
    _q_is_m1_over_m2 = False

    @classmethod
    def q_is_m1_over_m2(cls):
        """
        This classmethod defines the mass ratio as q = m1 / m2. Should be
        called before instantiating objects.
        """
        cls._q_is_m1_over_m2 = False

    @classmethod
    def q_is_m2_over_m1(cls):
        """
        This classmethod defines the mass ratio as q = m2 / m1. Should be
        called before instantiating objects.
        """
        cls._q_is_m1_over_m2 = True

    # alternate constructor
    # @classmethod
    # def from_masses(cls, m1, m2, a=1, P=None):
    #     """
    #     Construct RochePotential object from constituent object masses.

    #     Parameters
    #     ----------
    #     m1: float
    #         The primary mass
    #     m2: float
    #         The secondary mass

    #     Returns
    #     -------

    #     """
    #     if cls._q_is_m1_over_m2:
    #         return cls(m1 / m2, a=a, P=P)
    #     return cls(m2 / m1, a=a, P=P)

    @staticmethod
    def _resolve_q_m(q, m, m1, m2, a, P):
        if q:
            return q, m or 1

        if m1 and m2:
            if m and (m1 != (m1x := m - m2)):
                raise OverSpecifiedParameters(
                    f'Conflicting parameters m1, m2, m. '
                    f'm1 expected {m1x}, received {m1}')

            return m2 / m1, m1 + m2

        # q is None
        if m is None:
            if None in (a, P):
                raise UnderSpecifiedParameters()

            # use a and P along with the given mass to compute the other
            # mass using Kelper's third law. Also set the scale for
            # results.
            # total mass
            m = _4π2 * (a ** 3) / (G * P * P)

        # q is None, m is not None
        if m2:
            m1 = m - m2

        return m / m1 - 1, m

    def _resolve_a_P(self, a, P):
        if a and P:
            return a, P

        if a:
            P = 2 * π * a ** (3/2) / (self.m * G)
            return a, P

        if P:
            a = np.cbrt(self.m * G * P * P / _4π2)
            return a, P

        return 1, None

    def __init__(self, q=None, m1=None, m2=None, m=None, a=None, P=None):
        """
        Resolve the binary parameters from various input. The class
        can be initialized by specifying any minimal subset of the
        following parameters:

        Parameters
        ----------
        q: float
            The mass ratio
        m: float, Quantity
            Total mass in the system
        m1: float
            Primary mass
        m2: float
            Secondary mass
        a: float
            Orbital semi-major axis
        P: float
            The orbital period

        Examples
        --------
        BinaryParameters(1)              # q = 1
        BinaryParameters(0.5, m1=0.3)    # m1 = 0.3
        """

        # if one of the masses are not given, we need either the orbital period
        # and the semi-major axis

        _check_units({m: 'mass', m1: 'mass', m2: 'mass',
                      a: 'length', P: 'time'})
        self._q, self._m = self._resolve_q_m(q, m, m1, m2, a, P)
        # compute scale (semi-major axis)
        self._a, self._P = self._resolve_a_P(a, P)

    def pprint(self):
        a, P = self.a, self.P
        astr = (f"""{a.to('AU'):.5f}
                    \t\t= {pp.nr(a.value)} {a.unit}
                    \t\t= {a.to('lyr').value * u.a.to('s'):.5f} lightseconds
                """.expandtabs()
                if isinstance(a, Quantity) else
                f'a = {a}')

        Pstr = (f'{(P := self.P.to(u.h)).value:.3f} {P.unit}'
                if isinstance(P, Quantity) else
                f'P  = {P}')

        print(f'''\
            Mass ratio {self.q_is.upper()}: q  = {self.q:.3f}
            Semi-major axis:    a = {astr}

            Orbital Period:       P  = {Pstr}
            ''')

    @property
    def q(self):
        """Mass ratio. See """
        if self._q_is_m1_over_m2:
            return 1. / self._q
        return self._q

    @q.setter
    def q(self, q):
        self._set_q(q)
        self.logger.debug('Mass ratio updated: q = {} = {}', self.q_is, self.q)

    def _set_q(self, q):
        if q <= 0:
            raise ValueError("Mass ratio 'q' should be a positive integer.")

        q = float(q)
        # internal definition used is `q = m2 / m1 < 1`
        self._q = 1 / q if self._q_is_m1_over_m2 else q

        # reset all the lazy properties that depend on q
        del self.mu

    @property
    def q_is(self):
        """String giving definition of mass ratio q"""
        return 'm1/m2' if self._q_is_m1_over_m2 else 'm2/m1'

    @property
    def m1(self):
        """Primary / Accretor mass"""
        return self.m / (self._q + 1)

    @m1.setter
    @verify_mass_unit
    def m1(self, mass):
        self.q = (mass / self.m2) if self._q_is_m1_over_m2 else (self.m2 / mass)

    @property
    def m2(self):
        """Secondary / Donor mass"""
        return self.m - self.m1

    @m2.setter
    @verify_mass_unit
    def m2(self, mass):
        self.q = (self.m1 / mass) if self._q_is_m1_over_m2 else (mass / self. m1)

    @property
    def m(self):
        """Total binary mass"""
        return self._m

    @m.setter
    @verify_mass_unit
    def m(self, mass):
        self._m = mass

    @lazyproperty
    def mu(self):
        """
        X-coordinate position of the secondary wrt CoM coordinates in units of
        orbital semi-major axis *a*.
        """
        return 1. / (self._q + 1.)

    @mu.setter
    def mu(self, mu):
        q = 1 / float(mu) - 1
        self.q = 1 / q if self._q_is_m1_over_m2 else q

    @property
    def r2(self):
        """
        Euclidean xyz position of the secondary wrt CoM coordinates in units of
        orbital semi-major axis *a*.
        """
        return CartesianRepresentation(((self.mu - 1) * self.a, 0, 0))

    @property
    def r1(self):
        """
        Euclidean xyz position of the primary wrt CoM coordinates in units of
        orbital semi-major axis *a*.
        """
        return CartesianRepresentation((self.mu * self.a, 0, 0))

    # self.omega        # Orbital angular velocity

    @property
    def a(self):
        """Orbital semi-major axis"""
        return self._a

    @a.setter
    @u.quantity_input(a=['length', 'dimensionless'])
    def a(self, a):
        # update period *P*
        self._a, self._P = self._resolve_a_P(a, None)
        self.logger.debug('Semi-major axis updated: a = {}', a)

    @property
    def P(self):
        """Orbital Period"""
        return self._P

    @P.setter
    @u.quantity_input(P=['time', 'dimensionless'])
    def P(self, P):
        # update period *P*
        self._a, self._P = self._resolve_a_P(None, P)
        self.logger.debug('Period updated: P = {}', P)


class Axes3DHelper:
    _axes = None

    @property
    def axes(self):
        if self._axes is None:
            _, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            self._axes = ax
        return self._axes


class RochePotential(BinaryParameters, Axes3DHelper):
    """
    Gravitational potential of two point masses, m1, m2, expressed in
    co-rotating (Cartesian) coordinate frame with origin at the center of mass
    (CoM) of the system.

    Distances and times are expressed internally in units of ``a = |r1| + |r2|'',
    the semi-major axis of the orbit, and ``P'' the orbital period.  The mass
    ratio ``q = m1 / m2'' uniquely defines the Roche potential in this frame.

    """

    # TODO xyz, rθφ.  potential expressed in different coordinate systems

    # default tolerance on Lagrange point solutions
    xtol = 1.e-9

    def _set_q(self, q):
        super()._set_q(q)

        # reset all the lazy properties that depend on q
        del (self.l1, self.l2, self.l3, self.l4, self.l5, self.psi0)

    def __call__(self, xyz):
        """
        Evaluate potential at Cartesian (x, y, z) coordinate.
        """
        if isinstance(xyz, Quantity):
            xyz = (xyz / self.a)

        return binary_potential_com(self._q, *np.array(xyz))

    @lazyproperty
    def psi0(self):
        """Gravitational potential at L1"""
        return self(self.l1, 0)

    # def centrifugal(self, x, y, z):  #
    #     """
    #     Potential due to centrifugal force at Cartesian position (x, y, z).
    #     """
    #     # self.omega ** 2 *
    #     return 0.5 * (x * x + y * y)

    # def primary(self, x, y, z):  # psi_m1  # of_primary  # of_m1
    #     """
    #     Potential due to primary at Cartesian position (x, y, z).
    #     """
    #     mu = self.mu
    #     return -mu / np.sqrt((x + 1 - mu) ** 2 + y * y + z * z)

    # def secondary(self, x, y, z):
    #     """
    #     Potential due to secondary at Cartesian position (x, y, z).
    #     """
    #     mu = self.mu
    #     return (mu - 1.) / np.sqrt((x - mu) ** 2 + y * y + z * z)

    def total(self, xyz):
        """
        Gravitational + centrifugal potential of binary system in rotating
        coordinate frame (Cartesian)
        Takes xyz grid in units of the semi-major axis (a = r1 + r2)
        """

        # x, y, z = map(np.asarray, (x, y, z))
        # _phi = binary_potential_com(self._q, x, y, z)
        # k = -G * self.m / (2 * self.a)
        # k = -1

        u = self(self._q, xyz)

        return

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
        return CartesianRepresentation((lx * self.a, 0, 0))

    @lazyproperty
    def l1(self):
        δ = 1e-6
        interval = (self.mu - 1 + δ, self.mu - δ)
        return self._solve_lagrange123(interval)

    @lazyproperty
    def l2(self):
        δ = 1e-6
        interval = self.mu + δ, 2
        return self._solve_lagrange123(interval)

    @lazyproperty
    def l3(self):
        δ = 1e-6
        interval = -2, self.mu - 1 - δ
        return self._solve_lagrange123(interval)

    @lazyproperty
    def l4(self):
        # L4 has analytic solution
        return CartesianRepresentation(
            np.array([self.mu - 0.5, np.sqrt(3) / 2, 0]) * self.a)

    @lazyproperty
    def l5(self):
        # L5 has analytic solution
        return CartesianRepresentation((self.l4.x, -self.l4.y, self.l4.z))

    @lazyproperty
    def lagrangians(self):
        return CartesianRepresentation(
            np.transpose([getattr(self, f'l{i}').xyz for i in range(1, 6)])
        )

    # aliases
    L1 = l1
    L2 = l2
    L3 = l3
    L4 = l4
    L5 = l5

    def make_surface(self, res=100, extent=2, zmin=-12):
        #
        if isinstance(extent, numbers.Real):
            x0, x1 = y0, y1 = np.array([-1, 1]) * extent
        elif np.shape(extent) == (2,):
            x0, x1 = y0, y1 = extent
        else:
            (x0, x1), (y0, y1) = extent

        # get potential surface data
        ires, jres = duplicate_if_scalar(res)
        Y, X = np.mgrid[x0:x1:complex(jres),
                        y0:y1:complex(ires)]

        g = self([X, Y, np.zeros_like(X)])
        g[g < zmin] = zmin
        return X, Y, g

    def plot_wireframe(self, ax=None, xyz=None, res=100, extent=2, every=4, **kws):

        if xyz is None:
            xyz = self.make_surface(res, extent)

        ax = ax or self.axes
        return ax.plot_wireframe(*xyz,
                                 **{'rstride': every, 'cstride': every,
                                    **WIREFRAME_DEFAULTS,
                                    **kws})

    def plot_contour3d(self, ax=None, xyz=None, levels=(), cmap='jet', **kws):

        if levels in ((), None):
            # default contours #NOTE: these contours are not evenly spaced in z
            ψ = ψ1, ψ2, *_, ψ5 = self(self.lagrangians.xyz)
            levels = np.sort(np.unique(np.r_[
                np.linspace(-12, ψ1, 25),
                #np.linspace(roche.psi0, g.max(), 5),
                ψ,
                np.linspace(ψ2, ψ5, 5)[1:],
                ψ5 * 1.005]
            ))

        if xyz is None:
            xyz = self.make_surface()

        cmap = plt.get_cmap(cmap)
        colors = cmap((levels - levels[0]) / levels.ptp())
        cons = ax.contour3D(*xyz,
                            levels=levels,
                            colors=colors,
                            **kws)

        ax.set_zlim(levels.min(), levels.max())
        return cons

    # def get_contour_levels(self):

    def plot3d(self, ax=None, wireframe=(), lagrangians=LAGRANGE_3D_DEFAULTS,
               contours=()):

        ax = ax or self.axes

        # Potential well
        xyz = self.make_surface()
        wf = self.plot_wireframe(ax, xyz, **dict(wireframe))

        # Lagrangian points
        Lpoints, = ax.plot3D(*(xyz_ := self.lagrangians.xyz)[:2], self(xyz_),
                             **dict(lagrangians))

        # Equipotentials
        contours = self.plot_contour3d(ax, xyz, **dict(contours))

        ax.set_box_aspect((1, 1, 0.5))
        #ax.azim, ax.elev = -90, 90
        xyunit = getattr(self.a, 'unit', 'a')
        zunit = getattr(xyz[-1], 'unit', r'\left[\frac{GM}{2a}\right]')

        ax.set_xlabel(f'$x\ [{xyunit}]$', usetex=True)
        ax.set_ylabel(f'$y\ [{xyunit}]$',  usetex=True)
        ax.set_zlabel(fr'$\Psi(x,y,0)\ {zunit}$', usetex=True)

        return wf, Lpoints, contours

    plot3D = plot3d

# class EquipotentialSolver(RochePotential):
#     pass  # TODO


class RocheSolver(RochePotential):  #
    """
    Solver for the Roche equipotential surfaces.
    """

    def _solve_radius(self, theta, level=None, primary=False):
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
            mu = 1 - self.mu
            invert = -1
        else:  # secondary
            mu = self.mu
            invert = 1

        # get interval on radius containing root
        rmin = 1e-6  # will not work if Roche lobe is smaller than this
        # (that is, for a very extreme mass ratio)
        rmax = invert * self.l1.x - mu + 1  # distance from point mass to L1

        # Since the solver uses a root finding algorithm, it cannot be used to
        # solve for the equipotential point at L1 (which is a saddle point)
        l1 = (theta == 0)

        # radial distance from point mass location (center of star) to L1 point
        r = np.ma.empty_like(theta)
        # substitute exact L1 point
        r[l1] = rmax

        for i in np.where(~l1)[0]:
            # todo memoize solver for efficiency !
            r[i] = brentq(_binary_potential_polar1,
                          rmin, rmax, (theta[i], mu, level))

        return r, mu - 1, invert

    def _solve(self, theta, level=None, primary=False):
        # solve radius
        r, xoff, xflip = self._solve_radius(theta, level, primary)

        # coordinate conversion
        x, y = pol2cart(r, theta)
        # shift to CoM & flip left-right
        return (x + xoff) * xflip, y

    def solve(self, res=RESOLUTION.alt, theta=None, reflect=None, primary=False,
              scale=1.):
        """
        Solve for the roche lobe surface. Return x, y points on the 2d surface
        in the CoM coordinate frame.

        Parameters
        ----------
        res : [type], optional
            [description], by default RESOLUTION.alt
        theta : [type], optional
            [description], by default None
        reflect : [type], optional
            [description], by default None
        primary : bool, optional
            [description], by default False
        scale : [type], optional
            [description], by default 1.

        Examples
        --------
        >>>

        Returns
        -------
        [type]
            [description]
        """

        # if `theta' array given don't reflect unless explicitly asked to
        # do so by `reflect=True`
        auto_range = theta is None
        reflect = auto_range if reflect is None else reflect
        theta = (np.linspace(0, π, res) if auto_range else
                 np.atleast_1d(theta))

        # solve contour
        x, y = self._solve(theta, -self.psi0, primary) * scale

        # use reflection symmetry to construct the full contour
        if reflect:
            return np.r_[x, x[::-1]], np.r_[y, -y[::-1]]

        return x, y

    # def solve_contour(self, mu, theta, level, primary=False):
    #     # FIXME get this working
    #     if level > self.psi0:
    #         fun = _binary_potential_polar_com
    #     else:
    #         pass

    #     r = np.ma.empty_like(theta)
    #     for i in enumerate(theta):
    #         # noinspection PyTypeChecker
    #         r[i] = brentq(
    #             _binary_potential_polar1, rmin, rmax, (theta[i], mu, level))

    def make_surface(self, res, primary, scale=1., phi=None):
        """
        Solve 1D roche lobe, and use rotational symmetry (along the line of
        centres) to generate 3D Roche lobe surface.
        """

        resr, resaz = duplicate_if_scalar(res)
        if phi is None:
            phi = np.linspace(0, 2 * π, resaz)
        else:
            phi = np.asarray(phi)
            resaz = len(phi)

        x, y = self.solve(resr, reflect=False, primary=primary, scale=scale)
        z = np.zeros_like(x)

        # make 3D
        X = np.tile(x, (resaz, 1))

        # Use the rotational symmetry around the x-axis to construct
        # the Roche lobe in 3D from the 2D numerical solution
        Y, Z = np.einsum('ijk,jl->ikl', rotation_matrix_2d(phi), [y, z])
        return X, Y, Z

    # def solve(self, res=30, full=True):
    #     raise NotImplementedError

    # def _solve(self, func, interval, Theta, level=None):
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
    #     # # Theta = np.linspace(0, π, res)
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

# def _vec_solve(self, func, interval, theta, mu, level):
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
#         theta = np.linspace(0, π, res)
#         reflect = True
#     else:
#         # if `theta' array given don't reflect unless explicitly asked to
#         #  do so by `reflect=True`
#         theta = np.atleast_1d(theta)
#         reflect = reflect or False
#
#     # Since the solver uses a root finding algorithm, it cannot be used to
#     # solve for the equipotential point at L1 (which is a saddle point)
#     l1 = (theta == π)
#
#     # solve contour
#     rmax = self.mu - self.l1.x
#     r = np.empty_like(theta)
#     r[~l1] = self._solve(
#             _binary_potential_polar2, (1e-6, rmax), theta[~l1])
#     # sub exact L1 point
#     r[theta == π] = self.mu - self.l1.x
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


class RocheLobe(RocheSolver, Axes3DHelper):

    def __init__(self, q=None, m1=None, m2=None, a=None, P=None, primary=False):

        RocheSolver.__init__(self, q, m1, m2, a, P)
        self.primary = bool(primary)

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
            _, ax = plt.subplots()

        x1, y1 = self.solve(res, primary=self.primary, scale=self.a)

        lobe, = ax.plot(x1, y1, **kws)

        return ax, lobe

    plot2D = plot2d  # FIXME not inherited if overwritten

    def plot_wireframe(self, ax=None, res=RESOLUTION, **kws):
        """

        Parameters
        ----------
        ax
        res : {int, tuple}
                resolution in r and/or theta
        kws

        Returns
        -------

        """

        poly3 = ax.plot_wireframe(
            *self.make_surface(res, self.primary, self.a),
            **kws)
        return ax, poly3

    def plot_surface(self, ax=None, res=RESOLUTION, **kws):
        """
        res : {int, tuple}
                resolution in r and/or theta
        """
        ax = ax or self.axes
        x, y, z = self.make_surface(res, self.primary, self.a)
        poly3 = ax.plot_surface(x, y, z, **kws)
        return ax, poly3


class RocheLobes(RocheLobe):
    # res_default = rres, _ = RocheSolver1.res_default

    def __init__(self,  q=None, m1=None, m2=None, a=None, P=None):
        """
        """
        # accretor
        super().__init__(q, m1, m2, a, P)
        self.primary = self.accretor = self
        # donor
        self.secondary = self.donor = RocheLobe(q, m1, m2, a, P, False)

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
        centreMarks, = ax.plot(np.multiply([mu, mu - 1], self.primary.a),
                               [0, 0],
                               'g+', label='centres')
        comMarks, = ax.plot(0, 0, 'rx', label='CoM')
        artists = pri, sec, lagrangeMarks, comMarks, centreMarks

        ax.grid()
        ax.set(xlabel='x (a)', ylabel='y (a)')

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
        kws

        Returns
        -------

        """
        if ax is None:
            ax = self.axes

        poly1 = self.primary.plot_wireframe(ax, res, **kws)
        poly2 = self.secondary.plot_wireframe(ax, res, **kws)

        return self._extracted_from_plot_surface_9(ax, 'equal', poly1, poly2)

    def plot_surface(self, ax=None, res=RESOLUTION, **kws):
        """
        res : {int, tuple}
                resolution in r and/or theta
        """
        if ax is None:
            ax = self.axes

        poly1 = self.primary.plot_surface(ax, res, **kws)
        poly2 = self.secondary.plot_surface(ax, res, **kws)

        return self._extracted_from_plot_surface_9(ax, 'auto', poly1, poly2)

    # TODO Rename this here and in `plot_wireframe` and `plot_surface`
    def _extracted_from_plot_surface_9(self, ax, arg1, poly1, poly2):
        ax.set_aspect(arg1)
        ax.figure.tight_layout()
        return ax, (poly1, poly2)

    plot3d = plot3D = plot_surface
