"""
Methods for solving and plotting Roche lobe surfaces in 2d and 3d.
"""


# Kopal: 1972AdA&A...9....1K

# TODO: note the basic equations + references here


# std
import inspect
import numbers
import textwrap as txw
import functools as ftl
from collections import namedtuple

# third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import brentq
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from astropy import units as u
from astropy.constants import G
from astropy.utils import lazyproperty
from astropy.units import UnitTypeError
from astropy.units.quantity import Quantity
from astropy.units.physical import PhysicalType
from astropy.coordinates import CartesianRepresentation

# local
from recipes.array import fold
from recipes import pprint as pp
from recipes.dicts import AttrReadItem
from recipes.decorators import Decorator
from recipes.logging import LoggingMixin
from recipes.misc import duplicate_if_scalar
from recipes.transforms import pol2cart, rotation_matrix_2d


# Module constants
# ---------------------------------------------------------------------------- #
π = np.pi
_4π2 = 4 * π * π

Mo = u.M_sun
Ro = u.R_sun
lightseconds = u.def_unit('lightseconds', (u.lyr / float(u.a.to('s'))))

# ---------------------------------------------------------------------------- #

# Default units for orbital Parameters
PARAM_DEFAULT_UNITS = dict(m=Mo, m1=Mo, m2=Mo, a=u.AU, p=u.hour)
PARAM_PHYSICAL_TYPE = dict(m='mass', m1='mass', m2='mass',
                           a='length',
                           P='time')

# Plotting defaults
ARTIST_PROPS_2D = AttrReadItem(
    lagrange=dict(color='m', marker='.', ls='none'),
    accretor=dict(label='Accretor', color='g'),
    donor=dict(label='Donor', color='orangered'),
    centres=dict(label='centres', color='g', marker='+', ls='none'),
    CoM=dict(label='CoM', color='r', marker='x')
)
ARTIST_PROPS_2D['primary'] = ARTIST_PROPS_2D['accretor']
ARTIST_PROPS_2D['secondary'] = ARTIST_PROPS_2D['donor']
#
ARTIST_PROPS_3D = AttrReadItem(
    wireframe=dict(color='c', lw=0.75, alpha=0.5),
    lagrange=dict(color='darkgreen', marker='o', ms=3, ls='none')
)

# Default altitudinal and azimuthal resolution for RocheLobes
RESOLUTION = namedtuple('Resolution', ('alt', 'azim'))(50, 35)


# Helper functions
# ---------------------------------------------------------------------------- #

# Verify units
verify_mass_unit = u.quantity_input(mass=['mass', 'dimensionless'])


def get_value(x):
    """Get numeric value from quantity or return object itself."""
    return getattr(x, 'value', x)


class apply_default_units(Decorator):
    """
    Decorator for applying default units to function input parameters.
    """

    def __init__(self, default_units=(), **kws):
        self.sig = None     # placeholder
        for unit in dict(default_units, **kws).values():
            assert isinstance(unit, u.UnitBase)

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


def _check_units(namespace, allowed_physical_types):
    for name, kind in allowed_physical_types.items():
        if name not in namespace:
            continue

        obj = namespace[name]
        if isinstance(kind, (str, PhysicalType)):
            kind = (kind, )

        if isinstance(obj, Quantity) and (obj.unit.physical_type not in kind):
            raise UnitTypeError(f'Parameter {name} should have physical '
                                f'type(s) {kind}, not '
                                f'{obj.unit.physical_type}.')


def _resolve_a_P(m, a, P):  # sourcery skip: de-morgan
    if (a is None) and (P is None):
        return 1, P

    if P is None:
        P = 2 * π * a ** (3/2) / (m * G)
        return a, P

    if a is None:
        a = np.cbrt(m * G * P * P / _4π2)
        return a, P

    return a, P

# ---------------------------------------------------------------------------- #


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

# @


def q_of_l1(x1):
    #  Seidov 2004 eq 3
    # http://dx.doi.org/10.1086/381315
    x12 = x1 * x1
    return (1 - x1) ** 3 * (1 + x1 + x12) / (x1 ** 3 * (3 - 3 * x1 + x12))


def psi1(x1):
    #  Seidov 2004 eq 4
    t = x1 * (1 - x1)
    return np.polyval([-4, -10, 15, -12, 3], t) / np.polyval([1, 2, -1], t)


def q_of_l2(x2):
    #  Seidov 2004 Eq 6:
    x22 = x2 * x2
    return (x2 - 1) ** 3 * (1 + x2 + x22) / ((x22 * (2 - x2) * (1 - x2 + x22)))


def psi2(x2):
    #  Seidov 2004 eq 9
    return np.divide(np.polyval([4, -14, 18, 9, -36, 27, -4, -1], x2),
                     np.polyval([1, -2, 1, 2, -1], x2))


def q_of_l3(x):
    return 1 / q_of_l2(1 - x)

# def q_of_l3(x3):
#     #  Seidov 2004 eq 8: WARNING: This equation in the paper is wrong!!
#     x32 = x3 * x3
#     return (2 - x3) * x32 * (1 - x3 + x32) / ((x3 - 1) ** 3 * (1 + x3 + x32))


def _lagrange_objective_seidov(x, f, q):
    return f(x) - q


def _solve_lagrange123_seidov(i, q, xtol=1e-9):
    # faster solver for Lagrange points: Typically 3-5x faster than the
    # _solve_lagrange function
    assert i in (1, 2, 3), f'Invalid identifier for Lagrange point: {i}.'

    objective, interval = ((q_of_l1, (0, 1)) if i == 1 else
                           (q_of_l2, (1, 2)) if i == 2 else
                           (q_of_l3, (-1, 0)))
    interval += np.array([1, -1]) * xtol
    r1 = brentq(_lagrange_objective_seidov, *interval, (objective, q), xtol)
    return r1 + 1 / (q + 1) - 1


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


def _solve_lagrange123(i, q, xtol=1e-6):
    """
    Solve for L1, L2, or L3 x position in the CoM frame in units of semi-major
    axis *a*.

    Parameters
    ----------
    i : {1,2,3}
        L point to solve for
    q : float
        Mass ratio m2/m1
    xtol : float
        Tolarance value for solver.

    Returns
    -------
    float
        Distance to L1 in CoM coordinates in units of semi-major axis *a*.

    """
    assert i in (1, 2, 3), f'Invalid identifier for Lagrange point: {i}.'

    δ = 1e-6
    mu = 1. / (q + 1.)

    i -= 2
    interval = [-2, 2]
    δs = (δ, -δ)
    o = (-2, i)[i <= 0]
    for j in ((0, 1) if i == -1 else [i]):
        interval[j] = mu + δs[j] + o + j

    # if i == 1:    (mu - 1 + δ,    mu - δ)
    # elif i == 2:  (mu + δ,        2)
    # else:         (-2,            mu - 1 - δ)

    # {(-1, 0): -1,  # i + j
    #  (-1, 1):  0,  # i + j
    #  (0,  0):  0,  # i + j
    #  (1,  1): -1}  # (-2, i)[i <= 0] + j
    # print(interval)
    return brentq(_lagrange_objective, *interval, (mu,), xtol)


# def l1(q):  # xtol=1.e-9
#     """
#     Inner Lagrange points in units of binary separation a from origin CoM

#     Parameters
#     ----------
#     q
#     xtol

#     Returns
#     -------

#     """
#     return RochePotential(q).l1.x


def l1(q):
    """
    Inner Lagrange point (L1): The critical point between the two masses where 
    the gravitational attraction of M1 and that of M2 are equal and opposite.

    Parameters
    ----------
    q : numbers.Real
        Mass ratio m2/m1

    Returns
    -------
    float
        x-coordinate distance from centre-of-mass in units of *a*.
    """
    return _solve_lagrange123_seidov(1, q)


def l2(q):
    """
    Outer Lagrange point (L2): The point in the frame nearest the less massive
    body, where the combined gravitational force of both masses are equal and
    opposite the centrifugal force in the co-rotating frame. 


    Parameters
    ----------
    q : numbers.Real
        Mass ratio m2/m1

    Returns
    -------
    float
        x-coordinate distance from centre-of-mass in units of *a*.
    """
    return _solve_lagrange123_seidov(2, q)


def l3(q):
    """
    Outer Lagrange point (L3): The point nearest the more massive body, where
    the combined gravitational force of both masses are equal and opposite the
    centrifugal force in the co-rotating frame.

    Parameters
    ----------
    q : numbers.Real
        Mass ratio m2/m1.

    Returns
    -------
    float
        x-coordinate distance from centre-of-mass in units of *a*.
    """
    return _solve_lagrange123_seidov(3, q)


def l4(q):
    # L4 has analytic solution
    return np.array([1 / (q + 1) - 0.5, np.sqrt(3) / 2, 0])


def l5(q):
    # L5 has analytic solution
    return l4 * [1, -1, 1]


def L(i, q):
    return LPOINT_FUNCS[int(i)](q)


# aliases
L1 = l1
L2 = l2
L3 = l3
L4 = l4
L5 = l5

# Lagrange point function map
LPOINT_FUNCS = dict(enumerate([l1, l2, l3, l4, l5], 1))


# ---------------------------------------------------------------------------- #


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

# ---------------------------------------------------------------------------- #


class BinaryParameters(LoggingMixin):
    """
    Parameterization helper for binary star systems. Serves as a interpretation
    layer that checks validity of parameters and units of input combinations
    from the following set of parameters describing the binary orbit:
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
    mass of the binary. All results are returned in the centre of mass
    coordinates in units of a.
    Initializing with mass scale:
    >>> b = BinaryParameters(q=0.5, m1=0.4) # default unit is assumed solar masses
    ... b.m2


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
            return q, (1 if m is None else m)

        if (m1 is not None) and (m2 is not None):
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
        if m2 is not None:
            m1 = m - m2

        return m / m1 - 1, m

    def _resolve_a_P(self, a, P):  # sourcery skip: de-morgan
        a, P = _resolve_a_P(self.m, a, P)
        if hasattr(a, 'unit') and isinstance(a.unit, u.CompositeUnit):
            a = a.to('AU')
        return a, P

    def __init__(self, q=None, m=None, m1=None, m2=None,  a=None, P=None):
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

        _check_units(locals(), PARAM_PHYSICAL_TYPE)
        self._q, self._m = self._resolve_q_m(q, m, m1, m2, a, P)

        # compute length scale (semi-major axis)
        self._a, self._P = self._resolve_a_P(a, P)

    def __repr__(self):
        return f'{self.__class__.__name__}(q={self.q:.3f})'

    def __str__(self):
        info = (f'{p} = {v}' for p in 'qmP' if (v := getattr(self, p)))
        return f'{self.__class__.__name__}({", ".join(info)})'

    def pformat(self):
        """Format binary parameters as string."""
        a, P = self.a, self.P
        astr = (f"""{a.to('AU'):.5f}
                \t\t   = {pp.nr(a.to('km').value)} km
                \t\t   = {a.to('lyr').value * u.a.to('s'):.5f} lightseconds\
                """.expandtabs().strip('\n')
                if isinstance(a, Quantity) else
                f'a = {a}')

        Pstr = (f'{(P := self.P.to(u.h)).value:.3f} {P.unit}'
                if isinstance(P, Quantity) else
                f'P  = {P}')

        return txw.dedent(
            f'''\
            Binary Parameters:
            Mass ratio ({self.q_is}) : q = {self.q:.3f}
            Mass (m1+m2)       : m = {self.m.to(Mo).value} M⊙
            Orbital Period     : P = {Pstr}
            Semi-major axis    : a = {astr}\
            ''')

    def pprint(self):
        """Pretty print binary parameters"""
        print(self.pformat())

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
        return CartesianRepresentation((self.mu, 0, 0)) * self.a

    @property
    def r1(self):
        """
        Euclidean xyz position of the primary wrt CoM coordinates in units of
        orbital semi-major axis *a*.
        """
        return CartesianRepresentation((self.mu - 1, 0, 0)) * self.a

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

    @lazyproperty
    def omega(self):
        """Angular velocity"""
        return np.sqrt(G * self.m / self.a ** 3)

    # alias
    Ω = omega


class Axes3DHelper:
    """Launch a figure with 3D axes on demand."""

    _cid = None

    _subplot_kws = {'subplot_kw': dict(projection='3d')}

    @lazyproperty
    def axes(self):
        return self.get_axes()

    def get_axes(self):
        fig, ax = plt.subplots(**self._subplot_kws)
        self._cid = fig.canvas.mpl_connect('draw_event', self._on_first_draw)
        return ax

    def _on_first_draw(self, event):
        # have to draw rectangle inset lines after first draw else they point
        # to wrong locations on the edges of the lower axes
        fix_axes_offset_text(self.axes)

        # disconnect so this only runs once
        self.axes.figure.canvas.mpl_disconnect(self._cid)


def fix_axes_offset_text(ax):
    # return
    zax = ax.zaxis
    offsetText = zax.offsetText
    if offset := offsetText.get_text():
        # mute offset text
        offsetText.set_visible(False)

        
        # pp.METRIC_PREFIXES[zax.major.formatter.orderOfMagnitude]
        # .k.unit.to_string()
        # place with units in label
        ax.set_zlabel(zax.label.get_text().replace(
            r'\right]$',
            fr'\times 10^{{{zax.major.formatter.orderOfMagnitude}}}\right]$'))


# class Zbar:


def zaxis_cmap(ax, zrange=(), nseg=50, cmap=None):
    xyz = np.empty((3, nseg))
    _, x1 = ax.get_xlim()
    _, y1 = ax.get_ylim()
    xyz[:2] = np.array((x1, y1), ndmin=2).T
    z = xyz[2] = np.linspace(*(zrange or ax.get_zlim()), nseg)
    l = Line3DCollection(fold.fold(xyz.T, 2, 1, pad=False),
                         cmap=plt.get_cmap(cmap),
                         array=xyz[2],
                         zorder=10,
                         lw=3)
    ax.add_collection(l, autolim=False)
    # print(xyz[[0, 1], 0])
    l._z = z

    ax.zaxis.line.set_visible(False)
    return l


class RochePotential(BinaryParameters, Axes3DHelper):
    """
    Gravitational potential of two point masses, m1, m2, expressed in
    co-rotating (Cartesian) coordinate frame with origin at the center of mass
    (CoM) of the system.

    Distances and times are expressed internally in units of ``a = |r1| + |r2|'',
    the semi-major axis of the orbit, and ``P'' the orbital period.  The mass
    ratio ``q = m2 / m1'' uniquely defines the shape of the binary potential in
    this frame. The mass, period, and semi-major axis are related through
    Keplers 3rd law and together define the physical length scale.

    Providing the
    If they are


    """

    # TODO xyz, rθφ.  potential expressed in different coordinate systems

    # default tolerance on Lagrange point solutions
    xtol = 1.e-9

    # ------------------------------------------------------------------------ #

    def __call__(self, xyz):
        """
        Evaluate potential at Cartesian (x, y, z) coordinate.
        """
        if isinstance(xyz, Quantity):
            xyz = (xyz / self.a)

        return self.k * binary_potential_com(self._q, *np.array(xyz))

    @lazyproperty
    def k(self):
        """
        Scaling constant for potential.
        """
        if isinstance(self.m, Quantity) and isinstance(self.a, Quantity):
            return (G * self.m / (2 * self.a)).to('J/kg')
        return 1

    @BinaryParameters.m.setter
    def m(self, mass):
        BinaryParameters.m.fset(self, mass)
        del self.k

    @BinaryParameters.a.setter
    def a(self, a):
        BinaryParameters.a.fset(self, a)
        del self.k

    def _set_q(self, q):
        super()._set_q(q)

        # reset all the lazy properties that depend on q
        del (self.l1, self.l2, self.l3, self.l4, self.l5)
        self._psix.cache_clear()  # pylint: disable=no-member

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

    # def total(self, xyz):
    #     """
    #     Gravitational + centrifugal potential of binary system in rotating
    #     coordinate frame (Cartesian)
    #     Takes xyz grid in units of the semi-major axis (a = r1 + r2)
    #     """

    #     # x, y, z = map(np.asarray, (x, y, z))
    #     # _phi = binary_potential_com(self._q, x, y, z)
    #     # k = -G * self.m / (2 * self.a)
    #     # k = -1

    #     u = self(self._q, xyz)

    #     return

    # Lagrange Points
    # ------------------------------------------------------------------------ #

    def _lagrange123(self, i, xtol=xtol):
        lx = _solve_lagrange123_seidov(i, self._q, xtol)
        return CartesianRepresentation((lx, 0, 0)) * self.a

    @lazyproperty
    def l1(self):
        """
        Inner Lagrange point (L1) in the : The point where the gravitational
        attraction of M1 and that of M2 are equal and opposite.
        """
        return self._solve_lagrange123_seidov(1)

    @lazyproperty
    def l2(self):
        """
        Outer Lagrange point (L2): The point in the co-rotating frame nearest
        the less massive body, where the combined gravitational force of both
        masses are equal and opposite the centrifugal force.
        """
        return self._solve_lagrange123_seidov(2)

    @lazyproperty
    def l3(self):
        """
        Outer Lagrange point (L2): The point in the co-rotating frame nearest
        the most massive body, where the combined gravitational force of both
        masses are equal and opposite the centrifugal force.
        """
        return self._solve_lagrange123_seidov(3)

    @lazyproperty
    def l4(self):
        # L4 has analytic solution
        return CartesianRepresentation(l4(self._q) * self.a)

    @lazyproperty
    def l5(self):
        # L5 has analytic solution
        return CartesianRepresentation((self.l4.x, -self.l4.y, self.l4.z))

    @lazyproperty
    def lagrangians(self):
        """Coordinates of all Lagrange points L1-L5"""
        return CartesianRepresentation(
            np.transpose([getattr(self, f'l{i}').xyz for i in range(1, 6)])
        )

    # aliases
    L1 = l1
    L2 = l2
    L3 = l3
    L4 = l4
    L5 = l5

    # Lagrange Potentials
    # ------------------------------------------------------------------------ #
    @property
    def psi1(self):
        """Gravitational potential at L1"""
        return self._psix(1)

    @property
    def psi2(self):
        """Gravitational potential at L2"""
        return self._psix(2)

    @property
    def psi3(self):
        """Gravitational potential at L3"""
        return self._psix(3)

    @property
    def psi4(self):
        """Gravitational potential at L4"""
        return self._psix(4)

    @property
    def psi5(self):
        """Gravitational potential at L5"""
        return self._psix(5)

    @ftl.lru_cache(5)
    def _psix(self, i):
        return self(getattr(self, f'l{i}').xyz)

    # aliases
    ψ1 = psi1
    ψ2 = psi2
    ψ3 = psi3
    ψ4 = psi4
    ψ5 = psi5

    # Plotting
    # ------------------------------------------------------------------------ #
    _subplot_kws = dict(
        figsize=(10, 7),
        facecolor='none',
        subplot_kw=dict(projection='3d',
                        facecolor='none',
                        computed_zorder=False,
                        azim=-135),
        gridspec_kw=dict(top=1.05,
                         left=-0.125,
                         right=1.08,
                         bottom=-0.05)
    )
    _zbar = _contours = None

    def get_axes(self):
        ax = super().get_axes()
        self._label_axes(ax)
        ax.set_box_aspect((1, 1, 0.5))
        return ax

    def _label_axes(self, ax):
        units = {}
        for key, default_unit in dict(a='a', k=r'\frac{GM}{2a}').items():
            val = getattr(self, key)
            units[key] = (val.unit.to_string('latex').strip('$')
                          if hasattr(val, 'unit') else
                          default_unit)

        kws = dict(usetex=True, labelpad=10)
        ax.set_xlabel(fr'$x\ [{units["a"]}]$', **kws)
        ax.set_ylabel(fr'$y\ [{units["a"]}]$', **kws)
        ax.set_zlabel(fr'$\Psi(x,y,0)\ \left[{units["k"]}\right]$', **kws)

    def make_surface(self, res=100, extent=2, zmin=-12):
        """
        Create surface points for plotting.

        Parameters
        ----------
        res : int, tuple, optional
            Resolution, by default 100
        extent : int, tuple, optional
            Physical intervals for x and/or y, by default 2
        zmin : int, optional
            Lower bound for potential, by default -12. Values smaller than this
            will be clipped

        Examples
        --------
        >>>

        Returns
        -------
        [type]
            [description]
        """
        if hasattr(extent, 'unit'):
            extent = extent / self.a

        shape = np.shape(extent)
        if shape in ((), (1,)):
            (x0, x1) = (y0, y1) = np.array([-1, 1]) * extent
        elif np.shape(extent) == (2,):
            (x0, x1) = (y0, y1) = extent
        else:
            (x0, x1), (y0, y1) = extent

        # get potential surface data
        ires, jres = np.array(duplicate_if_scalar(res)) * 1j
        Y, X = np.mgrid[x0:x1:jres,
                        y0:y1:ires]
        g = self([X, Y, np.zeros_like(X)])

        if not hasattr(zmin, 'unit'):
            zmin = (zmin * self.k)

        g[g < zmin] = zmin
        return X * self.a, Y * self.a, g

    def plot_wireframe(self, ax=None, xyz=None, res=100, extent=2, every=4,
                       **kws):

        if xyz is None:
            xyz = self.make_surface(res, extent)

        ax = ax or self.axes
        return ax.plot_wireframe(*xyz,
                                 **{'rstride': every, 'cstride': every,
                                    **ARTIST_PROPS_3D.wireframe,
                                    **kws})

    def plot_contour3d(self, ax=None, xyz=None, levels=(), cmap='jet', zbar=True,
                       **kws):

        if levels in ((), None):
            levels = self._get_contour_levels()

        if xyz is None:
            xyz = self.make_surface()
            
        cmap = plt.get_cmap(cmap)
        colors = cmap((levels - levels[0]) / levels.ptp())
        self._contours = ax.contour3D(*map(get_value, xyz),
                                      levels=levels,
                                      colors=colors,
                                      **kws)

        zrange = levels.min(), levels.max()
        ax.set_zlim(*zrange)
        self._label_axes(ax)

        # Line along z-axis as a cmap
        if zbar:
            # ((z := xyz[-1]).min(), z.max())
            self._zbar = zaxis_cmap(ax, zrange,
                                    cmap=self._contours.get_cmap())
            ax.figure.canvas.mpl_connect('motion_notify_event', self._on_rotate)

        return self._contours, self._zbar

    def _get_contour_levels(self, n01=25, n25=5, zmin=-12, zmax=None):
        # default contours #NOTE: these contours are not evenly spaced in z
        ψl = ψ1, ψ2, *_, ψ5 = self(self.lagrangians.xyz)

        return np.sort(np.unique(np.r_[
            np.linspace(*map(get_value, [zmin * self.k, ψ1]), n01, endpoint=False),
            get_value(ψl),
            np.linspace(*map(get_value, (ψ2, ψ5)), n25, endpoint=False)[1:],
            zmax or get_value(ψ5) * 1.005]
        ))

    def _on_rotate(self, event):  # sourcery skip: de-morgan

        if (event.inaxes is not (ax := self.axes) or
                (ax.button_pressed not in ax._rotate_btn)):
            return

        if None in (self._zbar, self._contours):
            return

        self._update_zbar()

    def _update_zbar(self):
        ax = self.axes
        zax = ax.zaxis
        nseg = 50
        xyz = np.empty((3, nseg))
        mins, maxs, *_, highs = zax._get_coord_info(ax.figure._cachedRenderer)
        (x1, y1, _), _ = zax._get_axis_line_edge_points(
            np.where(highs, maxs, mins),
            np.where(~highs, maxs, mins))
        xyz[:2] = np.array((x1, y1), ndmin=2).T
        xyz[2] = self._zbar._z
        self._zbar.set_segments(fold.fold(xyz.T, 2, 1, pad=False))

    def _on_first_draw(self, event):
        if self._zbar:
            self._update_zbar()
        return super()._on_first_draw(event)

    def plot3d(self, ax=None, xyz=None,
               wireframe=(),
               lagrangians=ARTIST_PROPS_3D.lagrange,
               contours=()):

        ax = ax or self.axes

        # Potential well
        if xyz is None:
            xyz = self.make_surface()
                    
        wf = None
        zomx = None
        if wireframe is not False:
            wf = self.plot_wireframe(ax, xyz, **dict(wireframe))
            zomx = wf.get_zorder()

        # Equipotentials
        if contours is not False:
            contours, zline = self.plot_contour3d(ax, get_value(xyz),
                                                  **dict(contours))
            zorders = [_.zorder for _ in contours.collections]
            wf.set_zorder(min(zorders) - 0.1)
            zomx = max(zorders)
            # pl._sort_zpos = o
        else:
            contours = zline = None

        # Lagrangian points
        Lpoints = None
        if lagrangians is not False:
            Lpoints, = ax.plot3D(*(ρ := self.lagrangians.xyz)[:2], self(ρ),
                                 **dict(lagrangians), zorder=zomx)

        return AttrReadItem(wf=wf, l=Lpoints, contours=contours, zline=zline)
        # return wf, Lpoints, contours, zbarline

    plot3D = plot3d


# aliases
TwoBodyPotential = BinaryPotential = RochePotential


@ftl.lru_cache()
def solve_roche_radius(mu, level, theta, rmin, rmax):
    return brentq(_binary_potential_polar1, rmin, rmax, (theta, mu, level))


class EquipotentialSolver(LoggingMixin):
    """
    Solver for the Roche equipotential surfaces.
    """

    def __init__(self, q=None, m=None, m1=None, m2=None, a=None, P=None):
        #
        self.u = self.Ψ = RochePotential(q, m, m1, m2, a, P)

    def __call__(self, res=RESOLUTION.alt, theta=None, reflect=None,
                 primary=False):
        return self.solve(res, theta, reflect, primary)

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
            level = -self.u.ψ1 / self.u.k

        # if level > self.psi1:
        #     # ignore `primary' parameter
        #     'use binary potential CoM'

        # use symmetry to solve for both lobes with primary equation
        invert = (-1, 1)[primary]
        mu = (not primary) + invert * self.u.mu

        # get interval on radius containing root
        rmin = 1e-6  # will not work if Roche lobe is smaller than this
        # (that is, for a very extreme mass ratio)

        # Since the solver uses a root finding algorithm, it cannot be used to
        # solve for the equipotential point at L1 (which is a saddle point)
        l1 = (theta == 0)

        # radial distance from point mass location (center of star) to L1 point
        r = np.ma.empty_like(theta)
        # distance  to L1 # substitute exact L1 point
        r[l1] = rmax = float(invert * (self.u.l1.x / self.u.a) - mu + 1)

        # todo multiprocess
        for i in np.where(~l1)[0]:
            r[i] = solve_roche_radius(mu, level, theta[i], rmin, rmax)

        # centre = float(self.u.mu - primary)
        return r, float(self.u.mu - primary), invert

    def _solve(self, theta, level=None, primary=False):
        # solve radius
        r, xoff, xflip = self._solve_radius(theta, level, primary)

        # coordinate conversion
        x, y = pol2cart(r, theta)
        # shift to CoM & flip left-right

        a = self.u.a
        return (x * xflip + xoff) * a, y * a

    def solve(self, res=RESOLUTION.alt, theta=None, reflect=None, primary=False):
        """
        Solve for the Roche surface equipotenial contour. Return x, y points on
        the 2d surface (contour) in the CoM coordinate frame.

        Parameters
        ----------
        res : float, optional
            Angular resolution of the radial profile in points per π radians, by
            default RESOLUTION.alt.
        theta : np.ndarray, optional
            Angle(s) (clockwise from the line of centres) for which the radius
            of the equipotential surface will be solved. The default is None,
            which means *res* number of equispaced angles between 0 and π.
        reflect : bool, optional
            Whether to reflect the solved points about the x-axis to get the
            bottom (-z) part of the roche lobe surface. Default is to reflect
            the solution if user provides *theta* angles for solution explicitly.
        primary : bool, optional
            Whether to solve for the primary (accretor) or secondary (donor),
            by default False, implying secondary.
        scale : float, Quantity, optional
            Length scale by which the solution will be multiplied, by default
            scale by the value of the semi-major axis *a*.

        Examples
        --------
        >>> self.solve(10, primary=True)

        Returns
        -------
        x, y : np.ndarray
            Roche surface equipotential points.
        """

        # if `theta' array given don't reflect unless explicitly asked to
        # do so by `reflect=True`
        auto_range = (theta is None)
        reflect = auto_range if reflect is None else reflect
        theta = (np.linspace(0, π, res) if auto_range else np.atleast_1d(theta))

        # solve contour
        x, y = np.array(self._solve(theta, -self.u.ψ1 / self.u.k, primary))

        # use reflection symmetry to construct the full contour
        if reflect:
            return np.r_[x, x[::-1]], np.r_[y, -y[::-1]]

        return x, y

    # def solve_contour(self, mu, theta, level, primary=False):
    #     # FIXME get this working
    #     if level > self.psi1:
    #         fun = _binary_potential_polar_com
    #     else:
    #         pass

    #     r = np.ma.empty_like(theta)
    #     for i in enumerate(theta):
    #         # noinspection PyTypeChecker
    #         r[i] = brentq(
    #             _binary_potential_polar1, rmin, rmax, (theta[i], mu, level))

    def make_surface(self, res, primary, scale=None, phi=None):
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


class RocheLobe(Axes3DHelper):

    def __init__(self, solver, primary=False):
        self.solver = solver
        self.u = self.solver.u
        self.primary = bool(primary)

    @lazyproperty
    def rmax(self):
        # distance from object center to L1
        invert = (-1, 1)[self.primary]
        mu = (not self.primary) + invert * self.u.mu
        return invert * (self.u.l1.x / self.u.a) - mu + 1

        # use symmetry to solve for both lobes with primary equation
        # if self.primary:  # primary
        #     mu = 1 - self.solver.u.mu
        #     invert = -1
        # else:  # secondary
        #     mu = self.solver.u.mu
        #     invert = 1

        # # distance  to L1
        # return invert * (self.u.l1.x / self.u.a) - mu + 1

    def solve_radius(self, theta):
        return self.solver._solve_radius(theta, -self.u.ψ1 / self.u.k,
                                         self.primary)

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

        x1, y1 = self.solver(res, primary=self.primary)
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
            *self.solver.make_surface(res, self.primary),
            **kws)
        return ax, poly3

    def plot_surface(self, ax=None, res=RESOLUTION, **kws):
        """
        res : {int, tuple}
                resolution in r and/or theta
        """
        ax = ax or self.axes
        x, y, z = self.solver.make_surface(res, self.primary)
        poly3 = ax.plot_surface(x, y, z, **kws)
        return ax, poly3


class RocheLobes(Axes3DHelper):
    # res_default = rres, _ = RocheSolver1.res_default

    def __init__(self,  q=None, m=None, m1=None, m2=None, a=None, P=None):
        """
        """

        self.solver = EquipotentialSolver(q, m, m1, m2, a, P)
        self.u = self.solver.u

        # accretor
        # super().__init__(q, m, m1, m2, a, P)
        self.primary = self.accretor = RocheLobe(self.solver, True)
        # donor
        self.secondary = self.donor = RocheLobe(self.solver, False)

    def plot2d(self, ax=None, res=RESOLUTION.azim, **kws):

        # plot both lobes on same axes
        if ax is None:
            _, ax = plt.subplots()

        _, pri = self.accretor.plot2d(ax, res, **ARTIST_PROPS_2D.accretor)
        _, sec = self.donor.plot2d(ax, res, **ARTIST_PROPS_2D.donor)

        Lpoints, = ax.plot(*get_value(self.u.lagrangians.xyz[:2]),
                           **ARTIST_PROPS_2D.lagrange)
        centreMarks, = ax.plot([self.u.r1.x.value, self.u.r2.x.value], [0, 0],
                               **ARTIST_PROPS_2D.centres)
        comMarks, = ax.plot(0, 0, **ARTIST_PROPS_2D.CoM)
        artists = pri, sec, Lpoints, comMarks, centreMarks

        ax.grid()
        unit = getattr(self.u.a, 'unit', 'a')
        ax.set(xlabel=f'x [{unit}]',
               ylabel=f'y [{unit}]')

        return ax, artists

    plot2D = plot2d

    def label_lagrangians(self, ax, txt_offset=(0.02, 0.02), **kws):

        # Label lagrangian points
        texts = []
        txt_offset = np.array(txt_offset) * get_value(self.u.a)
        for i, xy in enumerate(self.u.lagrangians.xyz[:2].T):
            text = ax.annotate(f'L{i + 1}', xy, xy + txt_offset, **kws)
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
