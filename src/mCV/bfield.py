"""
Magnetic field models for stars
"""

# std
import numbers
import warnings
import functools as ftl
from collections.abc import Collection

# third-party
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D
from matplotlib.collections import LineCollection
from matplotlib.projections.polar import PolarAxes
from loguru import logger
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from astropy import units as u
from astropy.constants import mu0
from astropy.coordinates import (
    BaseRepresentation, PhysicsSphericalRepresentation as SphericalCoords)
from scipy.integrate import quad
from scipy.misc import derivative
from scipy.optimize import brentq
from scipy.special import factorial, lpmn, lpmv

# local
from scrawl.dualaxes import DualAxes
from recipes import op
from recipes.array.fold import fold
from recipes.string import named_items
from recipes.oo.temp import temporarily
from recipes.functionals import echo0, raises
from recipes.dicts import pformat as pformat_dict
from recipes.transforms.rotation import EulerRodriguesMatrix
from recipes.oo.property import ForwardProperty, lazyproperty
from recipes.transforms import cart2pol, cart2sph, pol2cart, sph2cart
from recipes.dicts import AttrDict, invert

# relative
from .roche import ARTIST_PROPS_3D, Ro
from .utils import _check_optional_units, default_units, get_value, has_unit
from .axes_helpers import (AxesHelper, OriginLabelledAxes, SpatialAxes3D,
                           get_axis_label)
from .plotting_helpers import (pi_radian_formatter, theta_tickmarks,
                               x10_formatter, plot_line_cmap as colorline)


# ---------------------------------------------------------------------------- #
# Module variables
π = np.pi
_2π = 2 * π
_4π = 4 * π
π_2 = π / 2
μ0_2π = mu0 / _2π


JoulePerTesla = u.J / u. T
DIPOLE_MOMENT_DEFAULT = 1 * JoulePerTesla
ORIGIN_DEFAULT = (0, 0, 0) * Ro

MULTIPOLE_NAMES = {
    # 0: monopole
    1: 'dipole',
    2: 'quadrupole',
    3: 'sextupole',
    4: 'octupole',
    5: 'decapole',
    6: 'dodecapole',
    7: 'quadecapole',
    8: 'sedecapole',
    9: 'octadecapole',
    10: 'icosapole'

}
MULTIPOLE_DEGREES = {
    **invert(MULTIPOLE_NAMES),
    # alternate names
    'hexapole':     3,
    'sexadecapole': 8
}

# 'dotriacontapole':      16,
# 'triacontadipole':      16,
# 'tetrahexacontapole':   32,
# 'hexacontatetrapole':   32

# ---------------------------------------------------------------------------- #
# Default parameters configuration

CONFIG = AttrDict(
    points_per_loop=50,
    loops_per_azimuth=4,
    rshells=1
)


# "... toroidal moments and the corresponding fields of toroidal multipoles do
#   not make a contribution to the static magnetic field.""
#   - Agre 2011, https://doi.org/10.3367/UFNe.0181.201102d.0173


# ---------------------------------------------------------------------------- #
def _repr_helper(obj, *attrs, **kws):
    return pformat_dict({att: getattr(obj, att) for att in attrs},
                        type(obj).__name__, str, '=', brackets='()', **kws)


# ---------------------------------------------------------------------------- #
def odd(l):
    return (l % 2) == 1


def even(l):
    return (l % 2) == 0


def wrap(n, l):
    return n if n >= 0 else (n + l * ((-n // l) + 1))


# def reflect(array, pivot):

def read_only(array):
    # make read only
    array.flags.writeable = False
    return array

# ---------------------------------------------------------------------------- #


_int_to_slice = {0: slice(1),
                 1: slice(1, None)}


def _resolve_intersection_index(i):
    if isinstance(i, numbers.Integral):
        return _int_to_slice[i]

    if i in (..., slice(None)):
        return slice(None)

    if not isinstance(i, slice):
        raise ValueError(f'Invalid intersection specifier: {i}. Should one of '
                         f'{0, 1, ...}')
    # must be a slice if we are here
    return i

# ---------------------------------------------------------------------------- #


def legendre(l, m, cosθ):
    # Schmidt normalized Associated Lagendre Function (fnALF)

    if m == 0:  # sourcery skip: assign-if-exp
        # In the degenerate case m = 0, Eq. (16) yields a value of P(0, n, cosθ)
        # that is too large by a factor 2**1/2 (see Chapman and Bartels, 1940,
        # Chapter XVII, Eqs.19 and 20).
        k = 1
    else:
        k = (-1) ** m * np.sqrt(2 * factorial(l - m) / factorial(l + m))
    return k * lpmv(m, l, cosθ)


def legendre1(l, cosθ):
    return -np.sqrt(2 * factorial(l - 1) / factorial(l + 1)) * lpmv(1, l, cosθ)


fnalf = legendre
fnalf1 = legendre1


def r_multipole_fieldline(l, θ):
    return np.abs(np.sin(θ) * fnalf1(l, np.cos(θ))) ** (1 / l)


def dipole_fieldline(θ, φ):
    θ, φ = np.broadcast_arrays(θ, φ)
    return np.asanyarray(sph2cart(np.sin(θ) ** 2, θ, φ))


# class ALF:
#     def


def alf1_zero_intervals(l):
    # we know there is a zero at θ = π/2 if l is even
    if l <= 2:
        return []

    # Lacroix '84
    v = np.arccos(np.sqrt(1 - (1 / (l * (l + 1)))))
    intervals = np.cos(np.linspace(v, π - v, l))
    intervals = list(mit.pairwise(intervals))

    if (l % 2) == 0:  # l is even
        intervals.pop((l - 1) // 2)  # already have this zero at cosθ = π/2

    # print(f'l = {l} intervals\n', intervals)
    return np.sort(np.sort(intervals, axis=0), axis=1)


@ftl.lru_cache()
def _alf1_zeros(l):
    """
    Zeros of the associated Legendre function of degree l and order 1.
    """
    # we know there is a zero at 0 if l is even
    zeros = [1, *([0] * even(l))]
    if l <= 2:
        return np.sort(zeros)

    # solve for positive intervals only and use 0-symmetry
    intervals = alf1_zero_intervals(l)[((l+1) // 2 - 1):]
    for interval in intervals:
        x0 = brentq(objective_legendre_zeros, *interval, (l, ), xtol=1e-15)
        zeros.append(x0)

    return np.sort(zeros)


def _alf1_theta_zeros(l):
    """
    Zeros of the associated Legendre function of degree l and order 1.
    """
    # we know there is a zero at θ = π/2 if l is even
    return np.arccos(_alf1_zeros(l))[::-1]
    # if (l % 2):  # odd
    #     # add the zero from sinθ =  0 at θ = 0
    #     thc = np.hstack([0, thc])
    # return thc


# @ftl.lru_cache()
# def _alf1_zeros(l):
#     """
#     Zeros of the associated Legendre function of degree l and order 1.
#     """
#     # we know there is a zero at 0 if l is even
#     zeros = [1, *([0] * even(l))]
#     if l <= 2:
#         return np.sort(zeros)

#     # solve for positive intervals only and use 0-symmetry
#     intervals = alf1_zero_intervals(l)[(l+1)//2 - 1:]
#     for interval in intervals:
#         x0 = brentq(objective_legendre_zeros, *interval, (l, ), xtol=1e-15)
#         zeros.append(x0)

#     return np.sort(zeros)


def objective_legendre_zeros(x, l):
    return lpmv(1, l, x)


def _theta_max(l):
    if odd(l):
        yield π / 2

    for interval in mit.pairwise(_alf1_zeros(l)):
        yield np.arccos(brentq(objective_rmax, *interval, (l, )))


def objective_rmax(x, l):
    return lpmv(0, l, x)


@ftl.lru_cache()
def theta_max(l):
    return np.fromiter(_theta_max(l), 'f')[::-1]


def _solve_theta(l, rshell):
    thz = _alf1_theta_zeros(l)
    thm = theta_max(l)
    rmax = r_multipole_fieldline(l, thm)

    # touch point (single point intersection)
    w, = np.where(rshell == rmax)
    for θ in thm[w]:
        yield θ
        yield θ

    # check which loops are intersected
    i = np.digitize(rshell, rmax)
    if w.size:
        i = max(i, w.max())

    intervals = mit.pairwise(mit.interleave_longest(thz[i:], thm[i:]))
    for interval in intervals:
        yield brentq(_angle_solver_F, *interval, (l, rshell))


def solve_theta_r(l, rshell):
    # split into pairs, 2 points of intersection for each loop
    return list(mit.windowed(_solve_theta(l, rshell), 2, step=2))


def _angle_solver_F(θ, l, rl):
    return np.abs(np.sin(θ) * fnalf1(l, np.cos(θ))) - rl
    # return r_multipole_fieldline(l, θ) - shell


def _angle_solver(θ, l, shell):
    return r_multipole_fieldline(l, θ) - shell

# ---------------------------------------------------------------------------- #


def get_2d_axes_cart(fig):
    ax = DualAxes(fig, 1, 1, 1)
    fig.add_subplot(ax)
    ax.setup_ticks()

    ax.set(xlabel=r'$\theta$',
           xticks=(xticks := np.linspace(0, π, 7)),
           #            ylim=(-2.5, 2.5)
           )
    ax.set_ylabel(r'$r(\theta)$', rotation=0, labelpad=20)

    # ax.parasite.yaxis.offsetText.set_visible(False)
    ax.parasite.set(xlabel=r'$\theta$', xticks=xticks)
    ax.parasite.xaxis.set_major_formatter(pi_radian_formatter)
    #ax.set(yscale='log', ylim=(0.1, 100))
    ax.parasite.yaxis.set_ticklabels([])

    ax.grid()
    return ax


def setup_2d_axes_polar(ax):
    # radial ticks
    ax.yaxis.set_major_locator(MaxNLocator(5))

    # angular ticks
    ax.set_theta_zero_location('N')
    ax.xaxis.major.formatter = pi_radian_formatter

    theta_tickmarks(ax, direction='inout', length=0.02)
    # ax.set_rlabel_position(0)
    ax.tick_params('x', pad=0)
    return ax


def get_axes_polar():
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    return setup_2d_axes_polar(ax)


def plot_fieldline_cart(self, ax=None, res=CONFIG.points_per_loop):

    if ax is None:
        fig = plt.figure()
        ax = get_2d_axes_cart(fig)

    first = True
    zeros = self.zeros_0_π

    for loop in range(self.l):

        θ, r = self.get_theta_r_loop(loop)
        drdθ = derivative(self.r, θ, np.ptp(θ[:2]))

        ax.plot(θ, drdθ, 'x', label='numeric dr/dθ')
        # ax.plot(x,  lpmv(0, l, np.cos(x)), ':', label=r'$P_{\ell}^0$')
        ax.plot(θ, self.r(θ), 'o', mfc='none', label='r(θ, l)')
        ax.autoscale(False)

        ax.plot(θ, self.F(θ), '--', label='F(θ, l)')
        ax.plot(θ, self.dF(θ), ':', label='dF(θ, l)')

        ax.plot(θ, self.dr(θ), 'k-.', label='dr(θ, l)')
        ax.plot(θ, self.ds(θ), '-', label='ds(θ, l)')

        ax.set_prop_cycle(None)
        if first:
            first = False
            ax.legend(loc='upper left')

    ax.vlines(self.theta_max, -3, 3, '0.7')
    ax.plot(self.θmax, self.rmax, '*')  # ,  color=line.get_color())
    ax.plot(zeros, self.r(zeros), 'ks')


def plot_lines_polar(ax, segments, **kws):

    ax = ax or get_axes_polar()
    assert isinstance(ax, PolarAxes)

    # Have to set the transform explicitly for line collection in polar axes
    # transform=ax.transData._b <- doesn't account for theta offset rotation
    art = LineCollection(
        segments,
        transform=(Affine2D().rotate(ax._theta_offset.get_matrix()[0, 2]) +
                   ax.transProjectionAffine +
                   ax.transAxes),
        **kws
    )
    ax.add_collection(art)
    return art


def plot2d_multipoles(n, nrows=None, ncols=2, multipole=None, magnitude=1,
                      radius=1, **kws):

    multipole = multipole or IdealMultipole
    mkw = {}
    if issubclass(multipole, PhysicalMultipole):
        mkw = dict(magnitude=magnitude,
                   radius=radius)

    if nrows is None:
        nrows = n // ncols

    fig, axes = plt.subplots(nrows, n // nrows,
                             figsize=(6.5, 8.85),
                             sharex=True, sharey=True,
                             subplot_kw=dict(projection='polar'),
                             gridspec_kw=dict(top=0.96,
                                              bottom=0.035,
                                              left=0.035,
                                              right=0.945,
                                              hspace=0.25,
                                              wspace=0.25),
                             )
    for i in range(n):
        l = i + 1
        b = multipole(degree=l, **mkw)

        ax = axes[divmod(i, 2)]
        ax.set_theta_zero_location('N')

        # plot!
        b.plot2d(ax, **kws)

        # label degree l
        ax.text(-0.125, 1.035,
                fr'$\bf \ell = {l}$',
                size=14,
                transform=ax.transAxes)
        # ax.plot(theta, lpmv(1, l, np.cos(theta)))

    # radial ticks
    ax.set_yticks(np.linspace(0, 1, 5)[1:])

    # angular ticks
    ax.xaxis.major.formatter = pi_radian_formatter

    for ax in axes.ravel():
        # have to do this first for some godawful reason
        ax.set_autoscale_on(False)

    for ax in axes.ravel():
        # major
        theta_tickmarks(ax, direction='inout', length=0.02)
        # ax.set_rlabel_position(0)
        ax.tick_params('x', pad=0)
        # minor
        # theta_tickmarks(ax, 73, 0.01)

    # Make grid lines between axes for aesthetic
    axx = fig.add_axes([0, 0, 1, 1], facecolor='none', frame_on=False)
    axx.set(xlim=(0, 1), ylim=(0, 1), navigate=False, autoscale_on=False)
    axx.plot([0.5, 0.5], [0, 1], '0.5', lw=1)
    axx.hlines(np.linspace(0, 1, nrows + 1), 0, 1, '0.5', lw=1)
    axx.vlines(np.linspace(0, 1, ncols + 1), 0, 1, '0.5', lw=1)

    return fig


# ---------------------------------------------------------------------------- #
# class PPrint:
#     def __init__(self)


# class CartesianCoords(CartesianRepresentation):

#     @property
#     def rθφ(self):
#         """Spherical coordinate representation."""
#         return self.represent_as(SphericalCoords)


class RotationMixin:

    @lazyproperty
    def direction(self):
        """
        Directional unit vector of the magnetic axis. ie. The normalized dipole
        magnetic moment in Cartesian coordinates.
        """
        return sph2cart(1, self.theta, self.phi)

    @lazyproperty
    def theta(self):
        """Magnetic colatitude."""
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = float(theta)
        del self.direction

    # alias
    θ = theta

    @property
    def phi(self):
        """Magnetic longitude / azimuth."""
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = float(phi)
        del self.direction

    # alias
    φ = phi

    @lazyproperty
    def _rotation_matrix(self):
        return EulerRodriguesMatrix(
            (np.sin(self.phi), -np.cos(self.phi), 0), self.theta
        ).matrix


# ---------------------------------------------------------------------------- #
class DegreeProperty:

    def __repr__(self):
        return f'{self.__class__.__name__}({self.degree})'

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree):
        """
        Degree of the multipole, typically denoted l or n
        """
        self._set_degree(degree)

    def _set_degree(self, degree):
        if degree < 1:
            raise ValueError(f'Degree of magnetic multipole field must be '
                             f'greater than 0. For example:\n{MULTIPOLE_NAMES}')

        self._degree = int(degree)
        # reset lazyproperties
        del self.name, self.odd, self.even

    l = degree

    @lazyproperty
    def name(self):
        """Name of the field shape eg: dipole"""
        return MULTIPOLE_NAMES[self.degree]

    @lazyproperty
    def odd(self):
        return bool(self.degree % 2)

    @lazyproperty
    def even(self):
        return not self.odd


# class AssociatedLegendreFunction(DegreeeProperty):
#     def __call__(self):


class MultipoleFieldLines(DegreeProperty):
    # ------------------------------------------------------------------------ #
    def __init__(self, l=1):
        self.degree = l

    def __call__(self, θ, φ, rshells=CONFIG.rshells):
        return self.xyz(θ, φ, rshells)

    # ------------------------------------------------------------------------ #
    def F(self, θ):
        return np.sin(θ) * fnalf1(self.degree, np.cos(θ))

    def dF(self, θ):
        # cosθ = np.cos(θ)
        l = self.degree
        return np.sqrt(2 * l * (l + 1)) * np.sin(θ) * fnalf(l, 0, np.cos(θ))

    def r(self, θ):
        """
        Axisymmetric field line radial distance from origin parameterized by θ
        """
        return np.abs(self.F(θ)) ** (1 / self.degree)

    # def rr(self, θ):
    #     return np.abs(f := self.F(θ)) ** (1 / self.degree) * np.sign(f)

    def dr(self, θ):
        _l = 1 / self.degree
        return _l * abs(f := self.F(θ)) ** (_l - 1) * self.dF(θ) * np.sign(f)

    def ds(self, θ):
        return np.sqrt((self.dr(θ)) ** 2 + self.r(θ) ** 2)
        # return (f := np.abs(F(θ, l))) ** (1 / l) * np.sqrt(1 + (dF(θ, l) / (l * f)) ** 2)
        # return self.r(θ) * np.sqrt(1 + 2 * (1 + 1/l) * (fnalf(l, 0, cosθ) / fnalf1(l, cosθ)) ** 2)

    def xyz(self, θ, φ, rshells=CONFIG.rshells):
        """
        Cartesian distance from origin for axisymmetric field line at
        colatitiude θ and azimuth φ. 

        Note that the magnetic field for a pure multipole is independent of φ,
        so the φ vector is used here merely to broadcast the output array.
        """
        r = self.r(θ)  # field lines independent of φ, compute before broadcast
        θ, φ = np.broadcast_arrays(θ, φ)
        r *= np.array(rshells, ndmin=θ.ndim + 1).T

        # convert to Cartesian coordinates. spatial axis at index position 0
        return np.moveaxis(np.asanyarray(sph2cart(r, θ, φ)), 0, -1)

    # ------------------------------------------------------------------------ #
    def _set_degree(self, degree):
        super()._set_degree(degree)
        del self.theta0, self.theta_max

    @lazyproperty  # (depends_on=DegreeProperty.degree)
    def theta0(self):
        return read_only(_alf1_theta_zeros(self.degree))

    @lazyproperty(depends_on=theta0)
    def zeros_0_π(self):
        return read_only(np.hstack([self.theta0,
                                    π - self.theta0[(-1 - self.even)::-1]]))

    @lazyproperty(depends_on=zeros_0_π)
    def theta_intervals(self):
        return fold(self.zeros_0_π, 2, 1, pad=False)

    @lazyproperty  # (depends_on=DegreeProperty.degree)
    def theta_max(self):
        return theta_max(self.degree)

    # alias
    θc = θ0 = theta0
    θmax = theta_max

    @lazyproperty(depends_on=theta_max)
    def rmax(self):
        return self.r(self.θmax)

    # ------------------------------------------------------------------------ #
    def get_loop_index(self, theta, wrap=False):
        wind, remain = divmod(theta, π_2)
        i = np.digitize(remain, self.theta0) - 1
        loops, *_ = self._get_index_offset_sign(i)
        if wrap:
            return loops
        return (loops + wind * (self.l // 2)).astype(int)

    def _check_theta_in_loop(self, loop, theta, tol=1e-12):
        assert isinstance(loop, numbers.Integral)

        # (a, b), *_ = self._get_interval_offset_sign(loop)
        # (a <= theta)

        theta = np.asarray(theta)
        # loops = set(list(self.get_loop_index(theta)))

        # if len(loops) > 1:
        #     raise ValueError(f'Values span multiple loops: {loops}')

        a, b = self.get_theta_interval(loop)
        if any(l := (theta - a > tol) & (b - theta > tol)):
            raise ValueError(
                f'Point (θ={theta[l]}) is outside the interval of loop {loop}: '
                f'θ ∈ {(a, b)}'
            )

    def _get_index_offset_sign(self, loops):
        squeeze = np.ndarray.item if isinstance(loops, numbers.Integral) else echo0

        loops = np.array(loops, int, ndmin=1)
        w, loops = np.divmod(loops, self.l)
        offsets = w * π
        signs = np.ones_like(loops)

        #
        l2 = (self.l // 2)
        if self.odd and (b := (loops == l2)).any():
            loops[b] = l2
            offsets[b] = w[b] * π
            # return l2, w * π, 1

        # higher zeros are reflections of lower zeros around θ = π. Wrap
        # determines whether higher zeros, or their lower order reflection are
        # returned (wrap=True))
        if (b := (loops >= l2)).any():
            # print('fold')# print(f'{i=:}')
            loops[b] = (l2 - loops[b] - 1 - self.even) % (l2 + 1)
            offsets[b] += π
            signs[b] = -1

        return tuple(map(squeeze, (loops, offsets, signs)))

    # def _get_index_offset_sign(self, loop):
    #     # TODO: get this to work with arrays
    #     w, loop = divmod(loop, self.l)

    #     # single loop
    #     l2 = (self.l // 2)
    #     if self.odd and (loop == l2):
    #         return l2, w * π, 1

    #     # higher zeros are reflections of lower zeros around θ = π. Wrap
    #     # determines whether higher zeros, or their lower order reflection are
    #     # returned (wrap=True))
    #     if (b := (loop >= l2)):
    #         # print('fold')# print(f'{i=:}')
    #         loop = (l2 - loop - 1 - self.even) % (l2 + 1)

    #     d = (1, -1)[int(b)]
    #     return loop, (w + b) * π, d    # return loops, (w + b) * π, d

    def _get_interval_offset_sign(self, loops):
        loops, offsets, signs = self._get_index_offset_sign(loops)
        return self.theta_intervals[loops], offsets, signs

    # def get_loop_index(self, θ):
    #     """Get index and wind number"""
    #     w, θ = np.divmod(np.array(θ), π)
    #     index = np.digitize(θ, self.zeros_0_π) - 1
    #     return index, w.astype(int)

    # def wrap_loop_index(self, loop):
    #     # loop = self._resolve_loop_int(loop)
    #     loop = int(loop)
    #     l2 = (self.l // 2)
    #     if (δ := l2 - loop) <= 0:
    #         return (δ - 1 - self.even) % (l2 + 1)
    #     return loop

    def _resolve_loop_int(self, loop):
        loop = int(loop)
        l = self.l
        if loop >= l:
            self._raise_invalid_loop_int(loop)

        # wrap negative
        return loop % l

    def _resolve_loop(self, loop):
        # which zero in this loop (upper / lower / both)
        if isinstance(loop, numbers.Integral):
            return self._resolve_loop_int(loop)

        if isinstance(loop, Collection):
            loops = np.array(loop, int, ndmin=1)  #
            if any(b := (loops >= self.l)):
                self._raise_invalid_loop_int(set(loops[b]))
            return loops

        if loop is ...:
            return slice(None)

        if not isinstance(loop, slice):
            raise ValueError(f'Invalid loop specifier {loop} of type '
                             f'{type(loop)}. Should be a (collection of) '
                             f'integer(s) or slice or ellipsis.')
        return loop

    def _raise_invalid_loop_int(self, loop):
        raise ValueError(f'Invalid loop index: {loop}. Should be an integer '
                         f'i < {self.l}.')

    def get_zeros(self, loop=..., wrap=False):
        # higher zeros are reflections of lower zeros around θ = π. Fold
        # determines whether higher zeros, or their lower reflection are
        # returned (wrap=True))

        # multi loop
        if isinstance(loop, slice):
            return self.zeros_0_π[loop]

        w, loop = np.divmod(loop, self.l)
        if wrap:
            w = 0

        # single loop
        l2 = (self.l // 2)
        if self.odd and (loop == l2):
            return np.array([(θ := self.theta0[-1]), π - θ]) + w * π

        # higher zeros are reflections of lower zeros around θ = π. Wrap
        # determines whether higher zeros, or their lower order reflection are
        # returned (wrap=True))
        if (b := (loop >= l2)):
            # print('fold')# print(f'{i=:}')
            loop = (l2 - loop - 1 - self.even) % (l2 + 1)
            b = not wrap

        d = (1, -1)[int(b)]
        return (w + b) * π + d * self.theta0[loop:(loop+2 or None)][::d]

    def get_theta_interval(self, loop, wrap=False):
        assert isinstance(loop, numbers.Integral)

        interval, offset, sign = self._get_interval_offset_sign(loop)
        if wrap:
            return interval
        return sign * interval[::sign] + offset

    def _split_interval_by_loop(self, interval):
        nz = self.l // 2
        begin, end = interval
        # print(f'{end=:}')
        iw, i = divmod(begin, π_2)
        i = np.digitize(i, self.theta0) - 1
        loop = int(i + iw * nz)
        # d = -1
        while True:
            # get wrapped interval
            interval, offset, sign = self._get_interval_offset_sign(loop)
            _, stop = offset + sign * interval[::sign]

            # print(end, stop, stop - end, (stop - end) < 1e-9, '.'*8)

            if round((d := end - stop), 9) <= 0:
                # print(f'{end=:}, {interval=:}, {offset=:}, {sign=:}, {d=:}')
                if sign > 0:
                    interval = np.array([interval[0], end - offset])
                else:
                    interval = np.array([-d, interval[1]])
                # print(f'{interval=:}')

                yield loop, interval, offset, sign
                return
            else:
                yield loop, interval, offset, sign

            loop += 1

    # def _split_interval_by_loop(self, interval):

    #     nz = self.l // 2
    #     (iw, jw), (ir, jr) = _, remains = np.divmod(interval, π_2)
    #     i, j = np.digitize(remains, self.theta0) - 1

    #     # ends_on_zero = (jr in self.theta0[self.odd:])
    #     ends_on_π_2 = (j == 0 and jr == 0)
    #     # add1 = ((jr != 0) and self.odd) # (j == 0) and

    #     print(f'{interval=:}\n{remains=:}')
    #     print(f'{j=:}, {jw=:}, {nz=:}\n{ends_on_π_2=:}') # \n{ends_on_zero=:}

    #     start = int(i + iw * nz)
    #     # add1 = 1 if self.odd else -int(ends_on_π_2)
    #     if self.even:
    #         stop = int(j + jw * nz + 1 -int(ends_on_π_2))
    #     else:
    #         x = ((j == 0) and (jr != 0)) + (jw > 2)
    #         stop = int(j + jw * nz + 1 + x) # - int(ends_on_π_2)

    #     print(f'{start=:} {stop=:}')
    #     loops = np.arange(start, stop)

    #     intervals, offsets, signs = self._get_interval_offset_sign(loops)
    #     # intervals = np.array(intervals)
    #     intervals[0, 0] = ir
    #     # if jr:
    #     #     s = signs[-1]
    #     #     intervals[-1, int(s > 0)] = π_2 * (s < 0) + s * jr

    #     return loops, intervals, offsets, signs

    def split_interval_by_loop(self, interval, wrap=False):

        loops, _intervals, offsets, signs = zip(
            *self._split_interval_by_loop(interval))

        if wrap:
            return np.array(loops), np.array(_intervals)

        intervals = np.zeros((len(_intervals), 2))
        for i, (_, interval, offset, sign) in enumerate(
                zip(loops, _intervals, offsets, signs)):
            intervals[i] = sign * interval[::sign] + offset

        return np.array(loops), intervals

    # def split_interval_by_loop(self, interval):
    #     #
    #     start, stop = interval = sorted(interval)
    #     (i, j), (iw, jw) = self.get_loop_index(interval)

    #     if ends_on_zero := ((stop % π) in self.zeros_0_π):
    #         j -= 1

    #     intervals = np.vstack([np.tile(self.theta_intervals, (jw - iw, 1)),
    #                            self.theta_intervals[i:(j + 1)]])
    #     intervals[0, 0] = start % π
    #     intervals[-1, -1] = stop - (jw - ends_on_zero) * π

    #     l = self.l
    #     loops = np.arange(i + iw * l, j + jw * l + 1)
    #     offsets = np.tile(π * np.arange(jw - iw + 1), (l, 1)).T.ravel()
    #     return intervals, offsets, loops

    def get_max(self, loop):
        loop = self._resolve_loop_int(loop)
        l2 = (self.l // 2)
        if b := int((δ := l2 - loop + self.odd) <= 0):
            loop = (δ - 1) % l2
            # print(f'{δ=:}, {loop=:}')

        return (b * π + (1, -1)[b] * self.θmax[loop], self.rmax[loop])
        # return self.θmax[loop], self.rmax[loop]

    # ------------------------------------------------------------------------ #
    @lazyproperty
    def arc_lengths(self):
        """Estimate the arc length of all *l* loops"""
        s, _err = zip(*self._get_full_arc_lengths())
        # reflect for higher loops with identical shape
        return read_only(np.array(s + s[(-1 - self.odd)::-1]))

    def arc_length(self, a, b):

        loops, intervals = self.split_interval_by_loop((a, b))
        loops %= self.l

        return sum(self._arc_length_single_loop(loops[0], *intervals[0]),
                   self.arc_lengths[loops[1:-1]].sum(),
                   self._arc_length_single_loop(loops[-1], *intervals[-1]))

    def _get_full_arc_lengths(self):
        # estimate arc length for each loop
        # if l odd add middle arc that crosses θ = π/2
        intervals = self.theta_intervals[:(self.l // 2 + self.odd)]
        for loop, interval in enumerate(intervals):
            yield self._arc_length_single_loop(loop, *interval)

    def solve_theta_arc_length(self, s, start):
        # loop, _ = self.get_loop_index(start)
        # stop = self.zeros_0_π[loop, 1]
        stop = self.theta_interval[self.get_loop_index(start, wrap=True), 1]
        return brentq(self._objective_theta_arc_length, start, stop, (start, s))

    def _arc_length_single_loop(self, loop, a, b):

        direction = (-1, 1)[int(a < b)]
        a, b = interval = np.sort([a, b])
        z0, z1 = zeros = self.get_zeros(loop)
        for i, θ in enumerate(interval):
            if (z0 > θ) | (θ > z1):
                raise ValueError(
                    f'Requested {("lower", "upper")[i]} bound (θ={θ}) of arc '
                    f'length integral is outside the boundary of loop {loop}: '
                    f'θ ∈ {(a, b)}'
                )

        logger.debug('Calculating arc length for interval θ ∈ {}', (a, b))

        i = (interval == zeros)
        inner = interval + (1e-5 * i * [1, -1])
        # ds = dr approx for points near singularity
        s0 = np.zeros(2)
        s0[i] = self.r(inner[i]).sum()

        s, err = quad(self.ds, *inner,)
        total = s + s0.sum()
        logger.debug('s={} s0={} total={} err={}', s, s0, total, err)
        return direction * total, err

    def _objective_theta_arc_length(self, b, a, s, **kws):
        # solve for theta that is Δs distance from a
        return quad(self.ds, a, b, **kws)[0] - s

    # ------------------------------------------------------------------------ #
    def solve_theta(self, r, xtol=1e-15):
        assert isinstance(r, numbers.Real)
        θ = list(self._solve_theta(r, xtol))
        return np.hstack([θ, np.subtract(π, θ[::-1])])

    def _solve_theta(self, r, xtol=1e-15):
        for loop in range(self.l // 2):
            yield from self._solve_theta_loop(r, loop, xtol=xtol)

        if self.odd:
            yield from self._solve_theta_loop(r, loop + 1, 0, xtol=xtol)

    def solve_theta_loop(self, r, loop, i=..., xtol=1e-15):

        assert isinstance(loop, numbers.Integral)
        #
        loop, offset, sign = self._get_index_offset_sign(loop)

        unpack = next if isinstance(i, numbers.Integral) else tuple
        return unpack(offset + sign * θ[::sign]
                      for θ in self._solve_theta_loop(r, loop, i, xtol))

    def _solve_theta_loop(self, r, loop, i=...,  xtol=1e-15):

        # logger.debug('Solving r={:.3f} for loop {:d}', r, loop)

        if r == 0:
            yield self.get_zeros(loop)[i]
            # yield self.theta0[loop:(loop + (2 if (i is ...) else i + 1))]
            return

        # get maximum point for loop
        θmax, rmax = self.get_max(loop)

        # check which loops are intersected
        if r > rmax:
            raise ValueError(f'No intersections for r = {r} > rmax = {rmax}')

        # intersection: which zero in this loop (upper / lower / both)
        i = _resolve_intersection_index(i)  # a slice!

        # touch point (single point intersection)
        if (r == rmax):
            yield (θmax, θmax)[i]
            return

        l = self.l
        θ0, θ1 = self.get_zeros(loop, wrap=True)
        for interval in [(θ0, θmax), (θmax, θ1)][i]:
            if r ** l < xtol:
                warnings.warn(f'Attempting to solve for r(θ) = {r} in interval'
                              f' {interval} where r**l < {xtol}. Accuracy of '
                              f'these results are not gauranteed.')
            yield brentq(_angle_solver_F, *interval, (l, r ** l), xtol=xtol)

    # ------------------------------------------------------------------------ #
    def get_theta_r(self, interval=(0, _2π), rshells=CONFIG.rshells,
                    res=CONFIG.points_per_loop):
        θ, r = np.hstack(list(self._get_theta_r(interval, res)))
        θ, r = np.broadcast_arrays(θ, r * np.array(rshells, ndmin=θ.ndim + 1).T)
        return θ.squeeze(), r.squeeze()

    def _get_theta_r(self, interval, res=CONFIG.points_per_loop):

        # split interval by loop
        for loop, interval in zip(*self.split_interval_by_loop(interval)):
            # convert interval to tuple so we can cache
            yield self.get_theta_r_loop(loop, interval, res)

    def get_theta_r_loop(self, loop, interval=None, res=CONFIG.points_per_loop):
        """
        Coordinate vectors (θ, r) (colatitude, radius) with non-linear step size
        in colatitude that produces approximately linearly spaced steps along
        the field line arc. Useful for approximating the arc length integral, as
        well as for producing good looking plots with well sampled field lines.

        Parameters
        ----------
        loop : int
            The loop number
        start : float, optional
            Colatitude interval θ, by default None
        res : int, optional
            Approximate resolution, by default 100

        Returns
        -------
        np.ndarray
            Theta sequence
        """
        logger.debug('Calculating coordinate vectors for loop {}, interval {}',
                     loop, interval)

        _loop, offset, sign = self._get_index_offset_sign(loop)

        if interval is None:
            interval = self.theta_intervals[_loop]
        else:
            self._check_theta_in_loop(loop, interval)

            interval = sign * (interval[::sign] - offset)
            logger.debug('Folded interval {}', interval)

        # loop = _loop
        if self.odd and (_loop == self.l // 2):
            # optimization: reflect array around π since solutions are the same
            start, stop = interval
            if stop > π_2:
                logger.debug('Folding around π/2: {} -> {}', interval, (start, π_2))
                θ, r = self._get_theta_r_loop(_loop, (start, π_2), res)

                i = np.digitize(π - stop, θ) - 1
                θ, r = (np.hstack([θ, π - θ[-2:i:-1], stop]),
                        np.hstack([r, r[-2:i:-1], self.r(stop)]))
            else:
                θ, r = self._get_theta_r_loop(_loop, tuple(interval), res)
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                θ, r = self._get_theta_r_loop(_loop, tuple(interval), res)

        # logger.debug('offset {} sign {}, θ {}', offset, sign, θ[[0, -1]])
        return offset + sign * θ[::sign], r[::sign]

    @ftl.lru_cache()  # TODO use recipes.caching.Cached
    def _get_theta_r_loop(self, loop, interval, res=CONFIG.points_per_loop):

        logger.debug('Calculating coordinate vectors for loop {}, interval {}',
                     loop, interval)

        # for dipole, linearly spaced θ works well
        if self.l == 1:
            return (θ := np.linspace(*interval, res)), self.r(θ)

        # higher order multipole field lines handled below
        ds = self.arc_lengths[loop] / res

        # compute various parts of theta vector for field line loop
        start, stop = interval
        middle, _ = self.get_max(loop)

        θ0, r0 = self._theta_series0(loop, start, ds, 0)
        θ1 = self._theta_series1(θ0[-1] if θ0 else start, middle, ds)

        θ3, r3 = self._theta_series0(loop, stop, ds, 1)
        θ2 = self._theta_series1(θ3[0] if θ3 else stop, middle, ds)

        return (np.hstack([θ0, θ1[1:],          θ2,         θ3[1:]]),
                np.hstack([r0, self.r(θ1)[1:],  self.r(θ2), r3[1:]]))

    def _theta_series0(self, loop, start, ds, j, n_max=1000):
        # approximate ds = dr for the lower (dr/dθ -> inf) part of the fieldline

        loop = self._resolve_loop_int(loop)

        # get maximum point for loop
        θmax, rmax = self.get_max(loop)

        if (op.gt, op.lt)[j](start, θmax):
            return [], []

        Δs = Δθ = 0
        θ, r0 = start, self.r(start)
        theta, rs = [θ], [r0]
        while ((i := len(rs)) < n_max and  # emergency stop
               (r := r0 + i * ds * np.sqrt((1 - Δθ ** 2))) < rmax and
               Δs < ds * 1.1):  # and

            θn = self.solve_theta_loop(r, loop, j)
            Δθ = np.abs(θn - θ)
            Δs = np.sqrt((Δr := r - rs[-1]) ** 2 + (Δθ * (r + Δr/2) ** 2))
            # print(f'{i=:} {Δθ=:} {θ=:} {ds=:}')

            theta.append(θn)
            rs.append(r)
            θ = θn

        direction = (1, -1)[j]
        return theta[::direction], rs[::direction]

    def _theta_series1(self, start, stop, ds):

        b = int(start < stop)
        direction = (-1, 1)[b]
        compare = (op.gt, op.lt)[b]
        ds = abs(ds) * direction

        θ = start
        theta = [θ]
        while compare(θ + ds, stop):
            try:
                θn = brentq(self._objective_theta_arc_length, θ, stop, (θ, ds))
                theta.append(θn)
                θ = θn
            except ValueError as err:
                if 'f(a) and f(b) must have different signs' in str(err):
                    theta.append(stop)
                    break
                raise

        return theta[::direction]

    # ------------------------------------------------------------------------ #
    def plot_r_cart(self, ax=None,  interval=(0, _2π), res=CONFIG.points_per_loop):

        if ax is None:
            fig = plt.figure()
            ax = get_2d_axes_cart(fig)

        # first = True
        zeros = self.get_zeros()
        θ, r = self.get_theta_r(interval, res=res)

        drdθ = derivative(self.r, θ, np.ptp(θ[:2]))

        ax.plot(θ, drdθ, 'x', label='numeric dr/dθ')
        # ax.plot(x,  lpmv(0, l, np.cos(x)), ':', label=r'$P_{\ell}^0$')
        ax.plot(θ, self.r(θ), 'o', mfc='none', label='r(θ, l)')
        ax.autoscale(False)

        ax.plot(θ, self.F(θ), '--', label='F(θ, l)')
        ax.plot(θ, self.dF(θ), ':', label='dF(θ, l)')

        ax.plot(θ, self.dr(θ), 'k-.', label='dr(θ, l)')
        ax.plot(θ, self.ds(θ), '-', label='ds(θ, l)')

        ax.legend(loc='upper left')

        ax.vlines(self.theta_max, -3, 3, '0.7')
        ax.plot(self.θmax, self.rmax, '*')  # ,  color=line.get_color())
        ax.plot(zeros, self.r(zeros), 'ks')

    def _get_plot_coords_polar(self, interval, rshells=CONFIG.rshells,
                               res=CONFIG.points_per_loop, bbox=None):

        base_coords = list(self._get_theta_r(interval, res))
        rshells = np.array(rshells, ndmin=1)
        coord_vectors = np.empty((len(rshells), len(base_coords)), object)

        for j, (θ, r) in enumerate(base_coords):
            for i, rs in enumerate(rshells):
                coord_vectors[i, j] = np.array([θ, rs * r])

        return coord_vectors

    def _get_plot_vectors_xy(self, interval, rshells=CONFIG.rshells,
                             res=CONFIG.points_per_loop, bbox=None):

        θ_r = self._get_plot_coords_polar(interval, rshells, res, bbox)
        # (shells, loops) (colatitude, spatial dimension)
        xy = np.empty(θ_r.shape, object)
        for (i, j), (θ, r) in np.ndenumerate(θ_r):
            xy[i, j] = np.moveaxis(pol2cart(r, θ), 0, -1)

        return xy

    def _get_plot_vectors_xyz(self, interval, phi=0, rshells=CONFIG.rshells,
                              res=CONFIG.points_per_loop, bbox=None):

        φ = np.array(phi, ndmin=2).T
        θ_r = self._get_plot_coords_polar(interval, rshells, res, bbox)
        # (shells, loops, azimuth) (colatitude, spatial dimension)
        xyz = np.empty((*θ_r.shape, phi.size), object)
        for (i, j), (θ, r) in np.ndenumerate(θ_r):
            _xyz = np.moveaxis(sph2cart(*np.broadcast_arrays(r, θ, φ)), 0, -1)
            for k, _xyz in enumerate(_xyz):
                xyz[i, j, k] = _xyz

        return xyz

    def plot2d(self, ax=None, interval=(0, _2π), rshells=CONFIG.rshells,
               res=CONFIG.points_per_loop, bbox=None, **kws):  # loop=...,

        #
        rshells = np.array(rshells, ndmin=1)

        # get axes
        ax = ax or get_axes_polar()

        segments = self._get_plot_vectors_xy(interval, rshells, res, bbox).ravel()
        art = plot_lines_polar(ax,  segments, **kws)

        ax.set_rlim(0, 1.025 * rshells.max() * self.rmax.max())
        # ax.autoscale_view()

        return art

    def plot3d(self, ax=None, interval=None, nshells=3, naz=5,
               res=CONFIG.points_per_loop, bbox=None, **kws):

        if interval is None:
            interval = self.theta_intervals[[0, -1], [0, 1]]

        # collate artist graphical properties
        kws = {**ARTIST_PROPS_3D.bfield, **kws}

        phi = np.linspace(0, _2π, naz + 1)
        rshells = np.arange(1, nshells + 1)

        fieldlines = self._get_plot_vectors_xyz(interval, phi, rshells, res, bbox)
        # fieldlines = fieldlines.ravel()

        art = Line3DCollection(fieldlines.ravel(), **kws)
        ax = ax or SpatialAxes3D().axes
        ax.add_collection3d(art)

        # ax.auto_scale_xyz(*fieldlines[-1, -1, 0].T)
        # ax.auto_scale_xyz(*get_value(fieldlines).T)

        d = nshells * self.rmax.max()
        lim = [-d, d]
        ax.set(xlim=lim, ylim=lim, zlim=lim)

        return art


# alias
MultipoleFieldlines = MultipoleFieldLines


class PhysicalMultipoleFieldLines(MultipoleFieldLines):
    def __init__(self, l, radius):  # TODO: origin
        super().__init__(l)
        self.radius = radius

    def __repr__(self):
        return f'{self.__class__.__name__}(l={self.degree}, r={self.radius})'

    # def __call__(self, shells, naz=5, res=100, bbox=None):
    #     # TODO: each field line is split into sections so that the
    #     # projection zorder gets calculated correctly for all viewing
    #     # angles. Fieldlines terminate on star surface. ???
    #     shells = np.array(shells, ndmin=1).ravel()
    #     nshells = shells.size

    #     fieldlines = np.empty((nshells, naz, res, 3))
    #     # flux_density = np.empty((nshells, naz, res))
    #     for i, shell in enumerate(shells):
    #         # surface intersection

    #         theta0 = np.arcsin(np.sqrt(rmin / shell))
    #         θ = np.linspace(theta0, π - theta0, res)
    #         fieldlines[i] = xyz = shell * self.fieldlines.xyz(θ, φ)
    #         # NOTE: since flux density indep of phi, this calculation is not
    #         # repeated unnecessarily for different phi here
    #         # flux_density[i] = b = self.flux_density(xyz[0])

    #     return super().__call__(shells, naz, res, bbox)

    @property
    def radius(self):
        """Reference radius. Field is undefined inside this radius."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        assert radius >= 0
        self._radius = radius

        del self.theta_intervals

    rmin = radius

    # @lazyproperty  # (depends_on=radius)
    # def theta0(self):
    #     return read_only(self.solve_theta(self.radius)))

    @lazyproperty  # (depends_on=radius)
    def theta_intervals(self):
        return read_only(self.solve_theta(self.radius).reshape((-1, 2)))

    # def solve_theta(self, r, xtol=1e-15):
    #     if r == self.radius:
    #         return self.theta_intervals.ravel()
    #     return super().solve_theta(r, xtol)

    # def r(self, θ):

    # def get_loop_index(self, θ):
    #     if (self.r(θ) < self.radius).any():
    #         raise ValueError('Point inside surface.')

    #     return super().get_loop_index(θ)

    def get_theta_r(self, interval=None, rshells=CONFIG.rshells, res=CONFIG.points_per_loop,
                    reflect=True):
        # changing parameter defaults
        if interval is None:
            interval = ((start := self.theta_intervals[0, 0]), _2π - start)

        return super().get_theta_r(interval, rshells, res, reflect)

    # def _get_theta_r(self, interval, res=100, reflect=True):

    #     # interleave nans so loops are not connected in plot
    #     yield from mit.interleave(
    #         super()._get_theta_r(interval, res, reflect),
    #         itt.repeat(([np.nan], [np.nan]))
    #     )

    def _get_plot_coords_polar(self, interval, rshells=CONFIG.rshells,
                               res=CONFIG.points_per_loop, bbox=None):

        # first get the base loop coord vectors r, θ
        (i, j), w = self.get_loop_index(interval)
        z = (self.get_zeros(i)[0], self.get_zeros(j)[1]) + π * w
        # z = self.get_zeros(i)[[0, 1], [0, 1]] + π * w

        # HACK need to temporarily set `theta_intervals` to the zero angles at
        # r = 0, so we can get the coordinates of the full loop arc
        with temporarily(self, theta_intervals=fold(self.zeros_0_π, 2, 1, pad=False)):
            base_coords = list(self._get_theta_r(z, res))

        # solve surface intersection points for each shell
        rshells = np.array(rshells, ndmin=1)
        coord_vectors = np.empty((len(rshells), len(base_coords)), object)
        for i, rs in enumerate(rshells):
            # optimize for rs == 1
            if rs == 1:
                intersects = self.theta_intervals
            else:
                intersects = self.solve_theta(self.radius / rs).reshape(-1, 2)

            logger.debug('rshell: {} intersects {}', self.radius / rs, intersects)
            for j, (θ, r) in enumerate(base_coords):
                jwind, jfold = divmod(j, self.l)
                θ0, θ1 = intersects[jfold] + jwind * π
                r = r * rs
                l = r > self.radius
                # print(j, jfold, (θ0, θ1 ), θ[l][[0, -1]])

                coord_vectors[i, j] = np.array([
                    np.hstack([θ0, θ[l], θ1]),
                    np.hstack([self.radius, r[l], self.radius])
                ])

        return coord_vectors

    def plot2d(self, ax=None, interval=None, rshells=CONFIG.rshells,
               res=CONFIG.points_per_loop, bbox=None,
               show_r=dict(color='0.5', lw=2, alpha=0.5), **kws):

        if interval is None:
            interval = (start := self.theta_intervals[0, 0],  _2π - start)

        line = super().plot2d(ax, interval, rshells, res, bbox, **kws)
        ax = line.axes

        # plot reference radius
        circle = None
        if show_r:
            circle = Circle((0, 0), self.radius, **show_r,
                            transform=ax.transProjectionAffine + ax.transAxes)
            ax.add_artist(circle)

        ax.set_rlim(0)
        return line, circle


PhysicalMultipoleFieldlines = PhysicalMultipoleFieldLines


class MagneticFlux:
    """ABC for magnetic flux computation."""

    def H(self, xyz):
        raise NotImplementedError

    def B(self, xyz):
        """
        Magnetic flux density vector (B-field) at cartesian position *xyz*.

        If *xyz* is in units of [m], and magnetic moment is in units of [J/T],
        then the H-field has units of [T].
        """
        return mu0 * self.H(xyz)

    def flux_density(self, xyz):
        """
        Magnetic flux density magnitude (B-field) at cartesian position *xyz*.

        If *xyz* is in units of [m], and magnetic moment is in units of [J/T],
        then the H-field has units of [T].
        """
        return np.linalg.norm(self.B(xyz), axis=-1)


class IdealMultipole(MagneticFlux, DegreeProperty):
    """
    Pure axissymetric dipole / quadrupole / octupole etc
    """

    degree = ForwardProperty('fieldlines.degree')

    def __init__(self, degree=1):
        self.fieldlines = MultipoleFieldLines(degree)

    # ------------------------------------------------------------------------ #
    # @default_units(xyz=u.dimensionless_unscaled)  # convert to quantity
    # @u.quantity_input(xyz=['length', 'dimensionless'])
    # def H(self, xyz):
    #     """
    #     Magnetic field strength (H-field) at cartesian position *xyz*.

    #     If *xyz* is in units of [m], and magnetic moment is in units of [J/T],
    #     then the H-field has units of [A/m].

    #     Parameters
    #     ----------
    #     xyz : [type]
    #         [description]

    #     Returns
    #     -------
    #     [type]
    #         [description]
    #     """

    #     assert xyz.shape[-1] == 3
    #     xyz = self._apply_default_spatial_units(np.asanyarray(xyz))
    #     r = np.linalg.norm(xyz, axis=-1, keepdims=True)

    def B(self, xyz):
        # compute in spherical and transfrom back to cartesian
        r, θ, _ = cart2sph(*xyz)
        return np.tensordot(self._rotation_matrix.T, self.B_sph(r, θ), 1)

    def B_sph(self, r, θ):
        # for axisymmetric multipoles only the m = 0 component is present

        l = self.degree
        r, θ = np.broadcast_arrays(r, θ)
        r = np.ma.masked_where(r == 0, r)
        c = (l + 1) * r ** -(l + 2)
        cosθ = np.cos(θ)

        B = np.zeros_like(r, shape=(3, *r.shape))
        B[:, r.mask] = np.ma.masked
        B[0] = Br = c * lpmv(0, l, cosθ)
        B[1] = Bθ = c * lpmv(0, l + 1, cosθ) - Br * cosθ
        # B[1] = Bθ = c * lpmv(0, l + 1, cosθ) - Br * cosθ
        return B

    def plot2d(self, ax, interval=(0, _2π), rshells=CONFIG.rshells, res=CONFIG.points_per_loop, **kws):  # projection='polar'
        # TODO: each field line is split into sections so that the projection
        # zorder gets calculated correctly for all viewing angles. Fieldlines
        # terminate on star surface. ???
        if interval is None:
            interval = self.theta_intervals[[0, -1], [0, 1]]

        if res is None:
            res = BASE_RESOLUTION * self.l

        fieldlines = self.fieldlines._get_plot_vectors_xy(interval, rshells, res)

        segments = []
        array = []
        for xy in fieldlines.ravel():
            seg = fold(xy, 2, 1, pad=False)
            segments.extend(seg)

            array.extend(
                np.linalg.norm(self.B_sph(*cart2pol(*seg.mean(1).T)), axis=0)
            )

        art = plot_lines_polar(ax,
                               segments,
                               cmap='jet',
                               array=np.log10(array))
        ax = art.axes

        # cmap=cmap,
        #                          array=(array := np.log10(get_value(flux))),
        #                          clim=(max(np.floor(m := array.min()), m),
        #                                max(np.ceil(m := array.max()), m))

        # line, = ax.plot(theta, r, **kws)
        # line2, = ax.plot(theta + π, r, **kws)
        # ax.plot(theta, lpmv(1, l, np.cos(theta)))
        ax.set_rlim(0, 1.025 * np.array(rshells).max() * self.fieldlines.rmax.max())
        return art  # , line2

    # alias
    plot2D = plot2d


# alias
PureMultipole = IdealMultipole


class PhysicalMultipole(IdealMultipole, RotationMixin, AxesHelper):

    def __init__(self, magnitude=1, radius=1, degree=1, theta=0, phi=0):
        """
        Pure multipole field with a reference *magnitude* (flux density) at some
        reference *radius*.

        Parameters
        ----------
        magnitude : int, optional
            [description], by default 1
        radius : int, optional
            [description], by default 1
        degree : int, optional
            [description], by default 1
        theta : int, optional
            [description], by default 0
        phi : int, optional
            [description], by default 0

        Examples
        --------
        >>>
        """
        IdealMultipole.__init__(self, degree)
        self.radius = radius
        self.magnitude = magnitude
        self.theta = theta
        self.phi = phi

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        """Reference radius"""
        assert radius > 0
        self._radius = radius

    R = radius

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    @default_units(magnitude=u.dimensionless_unscaled)  # convert to quantity
    @u.quantity_input(magnitude=['magnetic flux density', 'dimensionless'])
    def magnitude(self, magnitude):
        """Magnetic flux density at reference radius"""
        assert magnitude >= 0
        self._magnitude = magnitude

    def B_sph(self, r, θ):
        # for axisymmetric multipoles only the m = 0 component is present
        r = np.ma.masked_where(r < self.radius, r)
        return super().B_sph(r, θ) * self.magnitude * self.radius ** (self.l + 2)

    def fieldline(self, θ, φ, rscale=1):

        if self.rmax * rscale < self.radius:
            raise ValueError(f'No point on this field line (scale = {scale}) '
                             f'lies outside of reference radius {self.radius}.')

        r = rscale * self.r(θ)
        r = np.ma.masked_where(r < self.radius, r)
        if r.mask.any():
            warnings.warn('Masked points inside reference radius.')

        return np.moveaxis(np.asanyarray(sph2cart(r, θ, φ)), 0, -1)

    # def _angle_solver(self, θ, rshell):
    #     _angle_solver(self.l, θ, self.radius / rshell)

    def _fieldlines(self, shells, naz=5, res=100, bbox=None, rmin=0):
        #
        fieldlines, flux_density = super()._fieldlines(shells, naz, res,
                                                       bbox, rmin)

        # tilt
        if self.phi or self.theta:
            # 3D rotation!
            fieldlines = np.rollaxis(
                np.tensordot(self._rotation_matrix, fieldlines, (0, -1)),
                0, fieldlines.ndim)

        # apply default spatial units
        fieldlines = self._apply_default_spatial_units(fieldlines)

        # shift to origin
        fieldlines += self.origin.T

        return fieldlines, flux_density

    # def get_theta_fieldline(self, res=50):

    #     return get_theta_fieldlines(self.degree, self.radius)

    def plot2d(self, ax, shells=1, res=100, phi=0, cmap='jet', **kws):  # projection='polar'

        # TODO: cmap

        # calculate zeros of associated Legendre function, so we include those
        # points in the plotting domain
        shells = np.atleast_1d(np.asanyarray(shells))
        for rshell in shells:
            # arrays = []
            thetas = [np.hstack([np.nan, th, np.nan])
                      for th in get_theta_fieldlines(self.degree, self.radius / rshell)]

        r = self._fieldline_radial(theta, phi).T
        B = self.B_sph(r, theta)
        flux = np.sqrt(np.square(self.B_sph(r, theta)).sum(0))

        # reflect
        theta = np.hstack([theta, theta + π])
        r = np.hstack([r, r])
        flux = np.hstack([flux, flux])
        flux[flux.mask] = 1e-9

        if cmap is not False:
            lines, _ = colorline(ax, np.array([theta, r]), False,
                                 cmap=cmap,
                                 array=(array := np.log10(get_value(flux))),
                                 clim=(max(np.floor(m := array.min()), m),
                                       max(np.ceil(m := array.max()), m)))

        cb = ax.figure.colorbar(lines, ax=ax, shrink=0.75)
        # cb.ax.set_ylabel(r'$\theta\ [rad]$', rotation=0, labelpad=20,
        #                  usetex=True)

        # )

        # line, = ax.plot(theta, r, **kws)
        # line2, = ax.plot(theta + π, r, **kws)
        # ax.plot(theta, lpmv(1, l, np.cos(theta)))
        return lines  # , line2


class MagneticField(MagneticFlux, OriginLabelledAxes):
    """Base Class for Magnetic Field models."""

    def __init__(self, origin=ORIGIN_DEFAULT, moment=None, coeff=None, *,
                 theta=0, phi=0):
        """
        A magnetic field centred at *origin* with magnetic moment tensor
        *moment*.
        """

        #
        OriginLabelledAxes.__init__(self, origin)

        # TODO: for higher order multipoles, the moment is a tensor

        # # get magnetic moment
        # if isinstance(moment, numbers.Real) or \
        #         (isinstance(moment, np.ndarray) and moment.size == 1):
        #     # NOTE Quantity isa ndarray
        #     moment = moment * np.array(sph2cart(1, *np.radians((theta, phi))))

        # self.moment = moment

    @classmethod
    def dipole(cls, origin=ORIGIN_DEFAULT, moment=DIPOLE_MOMENT_DEFAULT, *,
               theta=0, phi=0):
        """
        Magnetic dipole field.

        Parameters
        ----------
        origin : [type], optional
            [description], by default ORIGIN_DEFAULT
        moment : [type], optional
            [description], by default DIPOLE_MOMENT_DEFAULT
        theta : int, optional
            [description], by default 0
        phi : int, optional
            [description], by default 0

        Examples
        --------
        >>>

        Returns
        -------
        [type]
            [description]
        """
        return MagneticDipole(origin, moment, theta=theta, phi=phi)


class MultipoleMagneticField(MagneticField):
    """
    A magnetic field model using the spherical harmonic expansion.
    """

    # The lowest-degree Gauss coefficient, g00, gives the contribution of an
    # isolated magnetic charge, so it is zero. The next three coefficients –
    # g10, g11, and h11 – determine the direction and magnitude of the dipole
    # contribution.
    #  - https://en.wikipedia.org/wiki/Earth%27s_magnetic_field#Spherical_harmonics

    def __init__(self, g, h, r):  # , theta=0, phi=0
        """
        See:
        Lowes & Duka (2012)
        http://link.springer.com/10.5047/eps.2011.08.005

        Alken + (2021)
        https://doi.org/10.1186/s40623-020-01288-x

        Parameters
        ----------
        g : array-like
            Gauss coefficients as a symmetric and trace free matrix with shape
            (l, 2l + 1). The shape of this array determines the highest degree
            (l) of the field (dipole, quadrupole etc.)
        r : float
            Reference radius.
        theta : int, optional
            Direction colatitiude, by default 0.
        phi : int, optional
            Direction azimuth, by default 0.


        """

        # reference radius
        self.R = r

        self._g = self._h = None
        for name, gh in dict(g=g, h=h).items():
            setattr(self, f'_{name}', self._check_coeff_array(gh, name))

        # self._g = g  # cos term coeff
        # self._h = h  # sin term coeff

    @staticmethod
    def _check_coeff_array(g, name):
        g = np.asanyarray(g)
        l, m = g.shape
        if m != l:
            raise ValueError(f'Invalid coefiicient array *{name}* for '
                             f'magnetic multipole field of degree {l}. '
                             f'Expected shape ({l}, {l}), received ({l}, '
                             f'{m}).')

        if g[0, 0] != 0:
            warnings.warn('Magnetic monopole coefficient {name}[0,0] should'
                          ' be 0, ignoring.')

        expect_zero = np.triu_indices(l, 1)
        # r, c = np.triu_indices(l)
        # expect_zero = np.hstack([expect_zero, [r, 2 * l - c - 1]])
        if g[expect_zero].any():
            warnings.warn(
                f'Gauss coefficients *{name}* for magnetic multipole of '
                f'degree l={l} should be 0 for column |m| ≤ l. Found non-'
                f'zero coefficients at '
                f'{named_items("position", list(zip(*expect_zero)))}.'
            )
        return g

    @property
    def degree(self):
        """Maximal degree (of associated Legendre function) for the model."""
        return len(self._g)

    def scalar_potential(self, r, θ, φ):
        """
        Magnetic scalar potential. The magnetic field outside the source region
        (r > R) can be calculated from the magnetic scalar potential via
            B = -∇V
        """

        # pylint: disable=unsubscriptable-object

        invalid = r < self.R
        if invalid.any():
            msg = (f'Magnetic scalar potential is only valid for space '
                   f'outside of the source region r > {self.R}.')
            if invalid.all():
                emit = raises(ValueError)
            else:
                emit = warnings.warn
                msg += f'Masking {invalid.sum()}/{invalid.size} points.'

            emit(msg)

        #
        cosθ = np.cos(θ)

        s = 0
        for l in range(1, self.degree):
            m = np.arange(0, l + 1)
            k = np.sqrt(2 * (factorial(l - m) / factorial(l + m))) * (-1) ** m
            Plm = k * lpmv(m, l, cosθ)
            s += ((self._g[l] * np.cos(m * φ) +
                   self._h[l] * np.sin(m * φ))
                  * Plm * (self.R / r) ** (l + 1)
                  ).sum(0)
        return self.R * s

    V = scalar_potential

    def B(self, r, θ, φ):
        # pylint: disable=unsubscriptable-object

        # l =  np.arange(1, self.degree)
        sinθ = np.sin(θ)
        # sequence associated Legendre function of the first kind of order up to
        # m and degree n
        Plm, dPlm = lpmn(self.degree, self.degree, np.cos(θ))

        B = np.zeros(3)
        for l in range(1, self.degree):
            m = np.arange(0, l + 1)
            sinmφ, cosmφ = np.sin(mφ := m * φ), np.cos(mφ)
            c1 = (self._g[l] * cosmφ + self._h[l] * sinmφ)
            c2 = (self._g[l] * sinmφ - self._h[l] * cosmφ)
            rr = (np.sqrt(2 * (factorial(l - m) / factorial(l + m))) * (-1) ** m
                  * (self.R / r) ** (l + 2))

            B[0] += (l+1) * c1 * rr * Plm  # Br
            B[1] += sinθ * c1 * rr * dPlm  # Bθ
            B[2] += (m / sinθ) * c2 * rr * Plm  # Bφ

        return B

    def H(self, r, θ, φ):
        # since the gauss coefficients already contain the vacuum permiability
        # coefficient, here we simply divide by that to get the H field
        return self.B(r,  θ, φ) / mu0


# alias
MagneticMultipole = MagneticMultipoleField = MultipoleMagneticField


class MagneticDipole(MagneticField):
    # TODO bounding box / sphere
    # TODO plot2D

    # https://en.wikipedia.org/wiki/Magnetic_dipole
    # https://en.wikipedia.org/wiki/Magnetic_moment

    def __init__(self, origin=ORIGIN_DEFAULT, moment=DIPOLE_MOMENT_DEFAULT, *,
                 theta=0, phi=0):
        """
        A magnetic dipole located at *origin* with magnetic *moment*.
        Direction of magnetic moment can be expressed by passing a 3-vector
        moment, or alternately specifying inclination *theta* and azimuth *phi*
        angles in randians.

        Parameters
        ----------
        origin : tuple, optional
            Field centre point, by default (0, 0, 0). Default units are in
            semi-major-axis scale.
        moment :  float, tuple, np.ndarray, Quantity, optional
            Magnetic moment, by default (0, 0, 1). Default units are in J/T or
            equivalently A m².
        theta : float, optional
            Altitude angle of magnetic moment vector in degrees, by default 0.
        phi : float, optional
            Azimuth angle of magnetic moment vector in degrees, by default 0.

        Examples
        --------
        >>>
        """

        #
        OriginLabelledAxes.__init__(self, origin)

        # TODO: for higher order multipoles, the moment is a tensor

        # get magnetic moment
        if isinstance(moment, numbers.Real) or \
                (isinstance(moment, np.ndarray) and moment.size == 1):
            # NOTE Quantity isa ndarray
            moment = moment * np.array(sph2cart(1, *np.radians((theta, phi))))

        self.moment = moment

    def __repr__(self):
        return _repr_helper(self, 'origin', 'moment')

    # ------------------------------------------------------------------------ #
    @property
    def moment(self):
        """
        Magnetic moment vector in Cartesian coordinates. Magnetic moment has
        units of [J/T] or equivalent.
        """
        return self._moment.xyz

    @moment.setter
    @default_units(moment=JoulePerTesla)
    @u.quantity_input(moment=['magnetic moment', 'dimensionless'])
    def moment(self, moment):
        if isinstance(moment, BaseRepresentation):
            self._moment = moment.represent_as(CartesianCoords)
            del self.direction
            return

        if np.size(moment) == 1:
            # note Quantity isa ndarray
            direction = sph2cart(1, *np.radians((self.theta, self.phi)))
            moment = moment * np.array(direction)

        # assert len(moment) == 3
        self._moment = CartesianCoords(moment)
        del self.direction

    @lazyproperty
    def direction(self):
        """Magnetic moment unit vector."""
        return (m := self._moment).xyz / m.norm()

    # alias
    magnetic_axis = direction

    @lazyproperty
    def theta(self):
        """Magnetic moment colatitude."""
        return self._moment.rθφ.theta.value

    @theta.setter
    def theta(self, theta):
        self._moment = SphericalCoords(self.phi, theta, self._moment.norm())
        del self.direction

    # alias
    θ = theta

    @property
    def phi(self):
        """Magnetic moment azimuth."""
        return self._moment.rθφ.phi.value

    @phi.setter
    def phi(self, phi):
        self._moment = SphericalCoords(phi, self.theta, self._moment.norm())
        del self.direction

    # alias
    φ = phi

    @lazyproperty
    def _rotation_matrix(self):
        return EulerRodriguesMatrix(
            (np.sin(φ := self.phi), -np.cos(φ), 0), self.theta
        ).matrix

    # ------------------------------------------------------------------------ #

    @default_units(xyz=u.dimensionless_unscaled)  # convert to quantity
    @u.quantity_input(xyz=['length', 'dimensionless'])
    def H(self, xyz):
        """
        Magnetic field strength (H-field) at cartesian position *xyz*.

        If *xyz* is in units of [m], and magnetic moment is in units of [J/T],
        then the H-field has units of [A/m].

        Parameters
        ----------
        xyz : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """

        assert xyz.shape[-1] == 3
        xyz = self._apply_default_spatial_units(np.asanyarray(xyz))
        r = np.linalg.norm(xyz, axis=-1, keepdims=True)
        m = self.moment

        # catch all warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('ignore')

            # rhat
            v = xyz / r

            # dipole flux density
            flux = (3 * v * (m * v).sum(-1, keepdims=True) - m) / (_4π * r ** 3)

            # handle warnings
            if caught:
                return np.ma.masked_where(r == 0, flux)

        return flux.si

    def B(self, xyz):
        """
        Magnetic flux density vector (B-field) at cartesian position *xyz*.

        If *xyz* is in units of [m], and magnetic moment is in units of [J/T],
        then the B-field has units of [T].
        """
        return mu0 * self.H(xyz)

    def flux_density(self, xyz):
        """
        Magnetic flux density magnitude (B-field) at cartesian position *xyz*.

        If *xyz* is in units of [m], and magnetic moment is in units of [J/T],
        then the B-field has units of [T].
        """
        return np.linalg.norm(self.B(xyz), axis=-1)

    @staticmethod
    def fieldline(θ, φ):
        θ, φ = np.broadcast_arrays(θ, φ)
        return np.moveaxis(np.asanyarray(sph2cart(np.sin(θ) ** 2, θ, φ)), 0, -1)

    def fieldlines(self, nshells=3, naz=5, res=100, bbox=None, rmin=0.,
                   scale=1):
        """
        Compute the field lines for a dipole magnetic field.

        Parameters
        ----------
        nshells : int, optional
            Number of concentric magnetic shells for which to compute field
            lines, by default 3
        naz : int, optional
            Number of azimuthal panels for which to draw field lines, by default
            5.
        res : int, optional
            Number of points on each field line, by default 100.
        scale : float, optional
            Scaling factor, by default 1.
        bbox : int, optional
            [description], by default 0
        rmin : float, optional
            Terminating inner radius for field lines, by default 0.

        Examples
        --------
        >>>

        Returns
        -------
        np.ndarray shape: (nshells, res * naz, 3)
            Cartesian coordiantes of the field lines.
        """
        shells = np.arange(1, nshells + 1) * scale
        return self._fieldlines(shells, naz, res, bbox, rmin)
        # return xyz, flux

    def _fieldlines(self, shells, naz=5, res=100, bbox=None, rmin=0.):

        shells = np.atleast_1d(shells).ravel()
        nshells = shells.size
        if isinstance(rmin, u.Quantity):
            rmin = rmin.to(self.origin.unit).value

        # radial profile of magnetcic field lines
        θ = np.linspace(0, π, res)
        φ = np.linspace(0, _2π, naz, endpoint=False)[None].T
        # self.surface_theta = []
        if rmin > 0:
            fieldlines = np.empty((nshells, naz, res, 3))
            flux_density = np.empty((nshells, naz, res))
            for i, shell in enumerate(shells):
                # surface intersection

                theta0 = np.arcsin(np.sqrt(rmin / shell))
                θ = np.linspace(theta0, π - theta0, res)
                fieldlines[i] = xyz = shell * self.fieldline(θ, φ)
                # NOTE: since flux density indep of phi, this calculation is not
                # repeated unnecessarily for different phi here
                flux_density[i] = b = self.flux_density(xyz[0])
            # flux_unit =

            # TODO: each field line is split into sections so that the
            # projection zorder gets calculated correctly for all viewing
            # angles. Fieldlines terminate on star surface. ???
            # fieldlines = fieldlines.reshape((-1, res, 3))
            # flux_density = flux_density.reshape((-1, res))
            # return fieldlines

            # Flux density unit
            flux_density *= op.AttrGetter('unit', default=1)(b)
        else:
            # convert to Cartesian coordinates
            line = self.fieldline(θ, φ)  # as 3D array
            fieldlines = line * shells[(np.newaxis, ) * line.ndim].T  # .reshape(-1, 3)
            flux_density = self.flux_density(fieldlines)

        # tilt
        if self.phi or self.theta:
            # 3D rotation!
            fieldlines = np.rollaxis(
                np.tensordot(self._rotation_matrix, fieldlines, (0, -1)),
                0, fieldlines.ndim)

        # clip field lines at bounding box boundary
        fieldlines = self.clipped(fieldlines, bbox)

        # scale
        # fieldlines *= scale

        # apply default spatial units
        fieldlines = self._apply_default_spatial_units(fieldlines)

        # shift to origin
        fieldlines += self.origin.T

        return fieldlines, flux_density  # .ravel()

    def clipped(self, fieldlines, extents):
        if extents:
            if np.size(extents) <= 1:
                extents = np.tile([-1, 1], (3, 1)) * extents
            xyz = fieldlines - extents.mean(1)
            # np.ma.masked_inside(xyz, *extents.T)
            fieldlines[(xyz < extents[:, 0]) | (extents[:, 1] < xyz)] = np.ma.masked
        return fieldlines

    # Plotting
    # ------------------------------------------------------------------------ #
    _subplot_kws = dict(
        figsize=(10, 7),
        # facecolor='none',
        subplot_kw=dict(projection='3d',
                        # facecolor='none',
                        computed_zorder=False,
                        azim=-135),
        gridspec_kw=dict(top=1.0,
                         left=-0.125,
                         right=1.0,
                         bottom=0.05)
    )

    def plot3d(self, ax=None, nshells=3, naz=5, res=100, scale=1.,
               rmin=0., bbox=None, cmap='jet', **kws):

        # collate artist graphical properties
        kws = {**ARTIST_PROPS_3D.bfield, **kws}

        # create field lines
        fieldlines, flux = self.fieldlines(nshells, naz, res, bbox,
                                           rmin, scale)

        #
        if (do_cmap := (cmap is not False)):
            # fold the lines into segments iot cmap each segment to a colour
            segments = fold.fold(fieldlines, 2, 1, axis=-2, pad=False).reshape((-1, 2, 3))
            flux = fold.fold(flux, 2, 1, axis=-1, pad=False).mean(-1).reshape(-1)

            unit = None
            if has_unit(flux):
                unit = next((b for b in self.moment.unit.bases if
                             b.physical_type == 'magnetic flux density'))
                flux = flux.to(unit)

            # cbar limits
            kws.update(
                array=(array := np.log10(get_value(flux))),
                clim=(max(np.floor(m := array.min()), m),
                      max(np.ceil(m := array.max()), m))
            )
        else:
            segments = fieldlines

        #
        art = Line3DCollection(get_value(segments), **kws)
        ax = ax or self.axes
        ax.add_collection3d(art)
        ax.auto_scale_xyz(*get_value(fieldlines).T)

        if do_cmap:
            cb = ax.figure.colorbar(art, ax=ax,
                                    pad=0.1, shrink=0.8, aspect=30,
                                    ticks=ticker.MaxNLocator(integer=True),
                                    format=x10_formatter)

            label = get_axis_label(r'B\left(r, \theta)\right)', unit)
            cb.ax.text(0, 1.05, label, transform=cb.ax.transAxes, ha='left')

        return art, cb

    # ------------------------------------------------------------------------ #
    # def as_multipole(self):


class EccentricDipole(MagneticDipole):
    """
    Dipole magnetic field with origin offset from the stellar centre. Magnetic
    moment can be specified by setting the maximal flux density at the
    magnetic north pole via the `B0` attribute.
    """

    def __init__(self,
                 host,
                 north_pole_flux=None,
                 centre_offset=ORIGIN_DEFAULT,
                 moment=None,
                 theta=0, phi=0):

        #
        _check_optional_units(locals(),
                              dict(north_pole_flux=['magnetic flux density']))

        if north_pole_flux is not None:
            moment = (_2π * north_pole_flux * host.radius ** 3 / mu0)

        #
        super().__init__(origin=host.origin + centre_offset,
                         moment=moment, theta=theta, phi=phi)
        self.host = host

    # ------------------------------------------------------------------------ #
    @property
    def centre_offset(self):
        """Offset position of magnetic field origin from stellar centre."""
        return self.host.origin - self.origin

    @centre_offset.setter
    def centre_offset(self, offset):
        if np.linalg.norm(offset) > self.host.radius:
            raise ValueError('Magnetic field origin cannot lie beyond stellar '
                             'surface.')

        MagneticField.origin.fset(self.host.origin + offset)

    # alias
    center_offset = centre_offset

    # classical parameterization
    # eg: Fraser-Smith 1987: https://doi.org/10.1029/RG025i001p00001

    # The dimensionless coordinate quantities (η, ξ, ζ) gives the fractional
    # (x, y, z) distance from the stellar centre

    @property
    def η(self):
        """
        The dimensionless coordinate quantity *η* gives the fractional
        x-distance from the stellar centre.
        """
        return self.centre_offset[0]

    @η.setter
    def η(self, x):
        assert x < 1
        self.centre_offset[0] = float(x) * self.host.radius

    @property
    def ξ(self):
        """
        The dimensionless coordinate quantity *ξ* gives the fractional
        y-distance from the stellar centre.
        """
        return self.centre_offset[1]

    @ξ.setter
    def ξ(self, x):
        assert x < 1
        self.centre_offset[1] = float(x) * self.host.radius

    @property
    def ζ(self):
        """
        The dimensionless coordinate quantity *ζ* gives the fractional
        z-distance from the stellar centre.
        """
        return self.centre_offset[2]

    @ζ.setter
    def ζ(self, x):
        assert x < 1
        self.centre_offset[2] = float(x) * self.host.radius

    @lazyproperty
    def north_pole_flux(self):
        """
        The maximal magnetic flux density on the stellar surface. This is a
        proxy for the magnetic moment vector.
        """
        return μ0_2π * np.linalg.norm(self.moment) / self.host.R ** 3

    @north_pole_flux.setter
    @default_units(north_pole_flux=u.MG)
    @u.quantity_input(north_pole_flux=['magnetic flux density'])
    def north_pole_flux(self, north_pole_flux):
        self.moment = north_pole_flux * self.host.R ** 3 / μ0_2π

    # alias
    B0 = north_pole_flux

    # @MagneticField.moment.setter
    # def moment(self, moment):
    #     MagneticField.moment.fset(moment)
    #     del self.north_pole_flux


# aliases
OffsetCenterDipole = OffsetCentreDipole = OffsetDipole = StellarDipole = EccentricDipole
