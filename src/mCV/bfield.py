"""
Magnetic field models for stars
"""

# std
import numbers
import warnings
import functools as ftl

# third-party
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
from matplotlib import ticker
from loguru import logger
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import factorial, lpmn, lpmv
from astropy import units as u
from astropy.constants import mu0
from astropy.utils.decorators import lazyproperty
from astropy.coordinates import (BaseRepresentation,
                                 PhysicsSphericalRepresentation as SphericalCoords)

# local
from recipes import op
from recipes.array import fold
from recipes.array.fold import fold
from recipes.functionals import raises
from recipes.string import named_items
from recipes.dicts import pformat as pformat_dict
from recipes.transforms import cart2pol, cart2sph, sph2cart
from recipes.transforms.rotation import EulerRodriguesMatrix

# relative
from .roche import ARTIST_PROPS_3D, Ro
from .axes_helpers import AxesHelper, OriginLabelledAxes, get_axis_label
from .utils import _check_optional_units, default_units, get_value, has_unit
from .plotting_helpers import (pi_radian_formatter, plot_line_cmap as colorline,
                               theta_tickmarks,
                               x10_formatter)


# ---------------------------------------------------------------------------- #
π = np.pi
_2π = 2 * π
_4π = 4 * π
π_2 = π / 2
μ0_2π = mu0 / _2π

JoulePerTesla = u.J / u. T
DIPOLE_MOMENT_DEFAULT = 1 * JoulePerTesla
ORIGIN_DEFAULT = (0, 0, 0) * Ro


NAMED_MULTIPOLE_ORDERS = {
    # monopole
    'dipole':               1,
    'quadrupole':           2,
    'sextupole':            3,
    # 'hexapole':             3,
    'octupole':             4,
    'decapole':             5,
    # 'sedecapole':           16,
    # 'dotriacontapole':      32,
    # 'triacontadipole':      32,
    # 'tetrahexacontapole':   64,
    # 'hexacontatetrapole':   64
}


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


# def F(θ, l):
#     return np.sin(θ) * fnalf1(l, θ)


# def dF(θ, l):
#     cosθ = np.cos(θ)
#     return l * lpmv(1, l + 1, cosθ) - cosθ * lpmv(1, l, cosθ)
#     # return l * (fnalf(l + 1, 1, cosθ) - cosθ * fnalf(l, 1, cosθ))
#     #return np.sqrt(2 * l * (l + 1)) * np.sin(θ) * fnalf(l, 0, np.cos(θ))


# def r(θ, l):
#     return r_multipole_fieldline(l, θ)


# def dr(θ, l):
#     return (1 / l) * np.abs((f := F(θ, l))) ** (1 / l - 1) * dF(θ, l) # * np.sign(f)


# def ds(θ, l):
#     cosθ = np.cos(θ)
#     return r(θ, l) * np.sqrt(1 + 2 * (1 + 1/l) * (lpmv(1, l, (cosθ)) / lpmv(1, l, cosθ)) ** 2)
#     #return np.sqrt((dr(θ, l)) ** 2 + r(θ, l) ** 2)
#     # return (f := np.abs(F(θ, l))) ** (1 / l) * np.sqrt(1 + (dF(θ, l) / (l * f)) ** 2)


# def _arc_lengths(l):
#     for interval in mit.pairwise(_alf1_theta_zeros(l)):
#         interval *= np.array([-1, 1]) * 1e-5
#         yield quad(ds, *interval, (l, ))


# def arc_lenghts(l):
#     return np.fromiter(_arc_lengths(l), float)


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
    zeros = [1, *([0] * ((l % 2) == 0))]
    if l <= 2:
        return np.sort(zeros)

    # solve for positive intervals only and use 0-symmetry
    intervals = alf1_zero_intervals(l)[(l+1)//2 - 1:]
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


def _solve_theta_r(l, rshell):
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
        yield brentq(_angle_solver, *interval, (l, rshell))


def solve_theta_r(l, rshell):
    # split into pairs, 2 points of intersection for each loop
    return list(mit.windowed(_solve_theta_r(l, rshell), 2, step=2))


def _angle_solver_F(θ, l, rl):
    return np.abs(np.sin(θ) * fnalf1(l, np.cos(θ))) - rl
    # return r_multipole_fieldline(l, θ) - shell


def _angle_solver(θ, l, shell):
    return r_multipole_fieldline(l, θ) - shell


def _get_theta_fl_rmin(l, rmin=None, res=50):
    last = None
    o = odd(l)
    intervals = solve_theta_r(l, rmin)
    if o:
        *intervals, (last, _) = intervals

    for x0, x1 in intervals:
        r, β = cart2pol((x1 - x0) / 2, r_multipole_fieldline(l, x0))
        yield (x0 + x1) / 2 + r * np.cos(np.linspace(β, π - β, res))

        # δ = np.diff(theta) / 2
        # pivot = np.mean(theta)
        # rx = r_multipole_fieldline(l, theta[0])
        # a_range = ((α := np.arctan2(rx, δ)), π - α)[::-1]
        # alpha = np.linspace(*np.squeeze(a_range), res)
        # yield pivot + np.sqrt((rx*rx + δ*δ)) * np.cos(alpha)

    if o:
        # add the zero from sinθ =  0 at θ = 0
        r, β = cart2pol(π_2 - last, r_multipole_fieldline(l, last))
        yield π_2 + r * np.cos(np.linspace(π - β, π / 2, res))


# def get_theta_intervals():
#     # even
#     thc = _legendre_zeros_angles(l)
#     #zerofold = fold(thc, 2, 1, pad=False)
#     # δ = 0.5 * np.diff(zerofold, axis=1)
#     # thetas = zerofold.mean(1, keepdims=True) + δ * np.cos(np.linspace(π, 0, res))

#     if odd(l):
#         # add the zero from sinθ =  0 at θ = 0

#         # thetas = np.array([
#         #     *thetas,
#         #     π_2 + (π_2 - thc[-1]) * np.cos(np.linspace(π, π_2, res))
#         # ])

def get_theta_fieldlines(l, rmin=None, res=50):
    """
    Choose angles so we sample the fieldlines roughly uniformly along the line.
    """

    # thm = theta_max(l)
    # rmax = r_multipole_fieldline(l, thm)
    # i = np.digitize(rshell, rmax)

    # thetas = np.empty((l + o, res))

    if rmin:
        return np.array(list(_get_theta_fl_rmin(l, rmin, res)))

    # even
    thc = _legendre_zeros_angles(l)
    zerofold = fold(thc, 2, 1, pad=False)
    δ = 0.5 * np.diff(zerofold, axis=1)
    thetas = zerofold.mean(1, keepdims=True) + δ * np.cos(np.linspace(π, 0, res))

    if odd(l):
        # add the zero from sinθ =  0 at θ = 0
        thetas = np.array([
            *thetas,
            π_2 + (π_2 - thc[-1]) * np.cos(np.linspace(π, π_2, res))
        ])

    return thetas.T


# ---------------------------------------------------------------------------- #

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
    ax.xaxis.major.formatter = radian_formatter

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


class MultipoleFieldline:
    def __init__(self, l=1):
        self.degree = l

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree):
        """
        Degree of the multipole, typically denoted l or n
        """
        if degree < 1:
            raise ValueError(f'Degree of magnetic multipole field must be '
                             f'greater than 0. For example:\n'
                             f'{NAMED_MULTIPOLE_ORDERS}')

        self._degree = int(degree)
        del self.theta0

    l = degree

    @property
    def name(self):
        """Name of the field shape eg: dipole"""
        return NAMED_MULTIPOLE_ORDERS[self.degree]

    def F(self, θ):
        return np.sin(θ) * fnalf1(self.degree, np.cos(θ))

    def dF(self, θ):
        # cosθ = np.cos(θ)
        l = self.degree
        return np.sqrt(2 * l * (l + 1)) * np.sin(θ) * fnalf(l, 0, np.cos(θ))

    def r(self, θ):
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

    @lazyproperty
    def theta0(self):
        return _alf1_theta_zeros(self.degree)

    @property
    def theta_max(self):
        return theta_max(self.degree)

    # alias
    θc = θ0 = theta0
    θmax = theta_max

    @lazyproperty
    def rmax(self):
        return self.r(self.θmax)

    def get_loop(self, θ):
        # if θ > π:
        #     raise Exception('todo')

        # loop number wrapped
        return np.digitize(np.array(θ) % π, self.get_zeros()) - 1

    def _resolve_loop(self, loop):

        loop = int(loop)
        l = self.l
        assert loop < l

        # wrap negative
        return (loop + l) % l

    def get_zeros(self, loop=...):

        # all loops
        if loop is ...:
            return np.hstack([self.theta0,
                              π - self.theta0[(-1 - even(self.l))::-1]])

        # single loop
        loop = self._resolve_loop(loop)

        if loop > (l_2 := self.l // 2):
            i = l_2 - loop
            return π - self.theta0[i:i-2:-1]

        elif loop == l_2:
            return np.array((θ := self.theta0[-1], π - θ))

        return self.theta0[loop:(loop+2)]

    @lazyproperty
    def arc_lengths(self):
        """Estimate the arc length of all *l* loops"""
        s, err = zip(*self._get_full_arc_lengths())
        # reflect for higher loops with identical shape
        return np.array(s + s[(-1 - odd(self.l))::-1])

    def _get_full_arc_lengths(self):
        # estimate arc length for each loop
        th0 = self.theta0

        if odd(self.l):
            # add middle arc that crosses θ = π/2
            th0 = [*th0, π - th0[-1]]

        u = np.array([1, -1])
        # since there is a
        # deltas = 10 ** np.linspace(-5, -4, (self.l + 1) // 2)
        for loop, interval in enumerate(mit.pairwise(th0)):
            yield self._arc_length_single_loop(loop, *interval)

    def arc_length(self, a, b):
        loops = loop0, loop1 = self.get_loop([a, b])
        if loop0 == loop1:
            return self._arc_length_single_loop(loop0, a, b)

        intervals = self.get_zeros()[loops]
        return (self._arc_length_single_loop(loop0, a, intervals[0, 1])
                + self.arc_lengths[loops[1:-1]].sum()
                + self._arc_length_single_loop(loop0, intervals[-1, 0], b))

    def _arc_length_single_loop(self, loop, a, b):

        direction = (-1, 1)[int(a < b)]
        a, b = interval = np.sort([a, b])
        z0, z1 = zeros = self.get_zeros(loop)
        for i, θ in enumerate(interval):
            if not (z0 <= θ <= z1):
                raise ValueError(
                    f'Requested {("lower", "upper")[i]} bound (θ={θ}) of arc '
                    f'length integral is outside the boundary of loop {loop}: '
                    f'θ ∈ {(a, b)}'
                )

        logger.debug('Calculating arc length for interval θ ∈ {}', (a, b))

        i = (interval == zeros)
        inner = interval + 1e-5 * i * [1, -1]
        # ds = dr approx for points near singularity
        s0 = np.zeros(2)
        s0[i] = self.r(inner[i]).sum()

        s, err = quad(self.ds, *inner,)
        total = s + s0.sum()
        logger.debug('s={} s0={} total={} err={}', s, s0, total, err)
        return total, err

    def solve_theta_arc_length(self, s, start):
        # stop = zeros[loop]
        return brentq(self._objective_theta_arc_length, start, zeros[loop], (start, s))

    def _objective_theta_arc_length(self, b, a, s, **kws):
        # solve for theta that is Δs distance from a
        return quad(self.ds, a, b, **kws)[0] - s

    #  def _arc_lengths(self):
    #     # estimate arc length for each loop
    #     th0 = self.theta0

    #     if odd(self.l):
    #         # add middle arc that crosses θ = π/2
    #         th0 = [*th0, π - th0[-1]]

    #     u = np.array([1, -1])
    #     # since there is a
    #     # deltas = 10 ** np.linspace(-5, -4, (self.l + 1) // 2)
    #     for interval in mit.pairwise(th0):
    #         inner = (interval + 1e-5 * u)
    #         print(f'{interval=:}')
    #         print(f'{inner=:}')
    #         # ds = dr approx for points near singularity
    #         s0 = self.r(inner).sum()
    #         s, err = quad(self.ds, *inner,)
    #         print(f'{s=:}\n{s0=:}\n{err=:}')
    #         # print(f'total={s+s0}')
    #         yield s + s0, err

    def _solve_theta_r_loop(self, rshell, loop, i=...,  xtol=1e-15):

        #
        loop = self._resolve_loop(loop)

        # get maximum point for loop
        j = l_2 - loop if loop > (l_2 := self.l // 2) else loop
        θmax, rmax = self.theta_max[j], self.rmax[j]

        # touch point
        if (rshell == rmax):
            return ((θ := self.theta_max[j]), θ)[i]

        # intersection
        # which zero in this loop (upper / lower / both)
        if isinstance(i, numbers.Integral):
            i = {0: slice(1),
                 1: slice(1, None)}[i]
        elif i in (..., slice(None)):
            i = slice(None)
        elif not isinstance(i, slice):
            raise ValueError(f'Invalid intersection specifier: {i}. Should one'
                             ' of {0, 1, ...}')

        θ0, θ1 = self.get_zeros(loop)
        for interval in [(θ0, θmax), (θmax, θ1)][i]:
            if rshell ** self.l < xtol:
                warnings.warn(f'Attempting to solve for r(θ) = {rshell} in '
                              f'interval {interval} where r**l < {xtol}. '
                              f'Accuracy of these results are not gauranteed.')
            yield brentq(_angle_solver_F, *interval, (self.l, rshell ** self.l),
                         xtol=xtol)

    def solve_theta_r_loop(self, rshell, loop, i=..., xtol=1e-15):
        unpack = next if isinstance(i, numbers.Integral) else tuple
        return unpack(self._solve_theta_r_loop(rshell, loop, i, xtol))

    def _solve_theta_r(self, rshell, xtol=1e-15):
        # thz = _alf1_theta_zeros(l)
        thm = self.theta_max
        # rmax = self.r(l, thm)

        # touch point (single point intersection)
        w, = np.where(rshell == self.rmax)
        for θ in thm[w]:
            yield θ
            yield θ

        # check which loops are intersected
        i = np.digitize(rshell, rmax)
        if w.size:
            i = max(i, w.max())

        for interval in mit.pairwise(
                mit.interleave_longest(self.theta0[i:], thm[i:])):
            yield brentq(self._theta_objective, *interval, (rshell, ), xtol=xtol)

    def get_theta_r(self, loop, start=None, stop=None, res=100):
        """
        Theta vector with non-linear step size which are approximately linearly 
        spaced along field line arc length.

        Parameters
        ----------
        loop : int
            The loop number
        start : float, optional
            Interval start value θ (infimum), by default None
        stop : float, optional
            Interval end point (supremum) θ, by default None
        res : int, optional
            Approximate resolution, by default 100


        Returns
        -------
        np.ndarray
            Theta sequence
        """

        loop = int(loop)
        assert loop < self.degree

        zeros = self.get_zeros(loop)
        # if start is None:

        start = start or zeros[0]
        stop = stop or zeros[1]
        middle = np.mean([start, stop])

        # for dipole, linearly spaced θ works well
        if self.l == 1:
            return np.linspace(start, stop, res)

        # higher order multipole field lines handled below
        ds = self.arc_lengths[loop] / res

        # compute various parts of theta vector for field line loop
        r0, θ0 = self._theta_series0(loop, start, ds, 0)
        r3, θ3 = self._theta_series0(loop, stop, ds, 1)
        θ1 = self._theta_series1(θ0[-1], middle, ds)
        θ2 = self._theta_series1(θ3[0], middle, ds)

        return (np.hstack([θ0, (θ1 := θ1[1:]), (θ2 := θ2[1:]), θ3[1:]]),
                np.hstack([r0, self.r(θ1),     self.r(θ2),     r3[1:]]))

    def _theta_series0(self, loop, start, ds, j, n_max=1000):
        # approximate ds = dr for the lower (dr/dθ -> inf) part of the fieldline

        loop = self._resolve_loop(loop)
        zeros = self.get_zeros(loop)
        # ptp = np.ptp(zeros)

        # get maximum point for loop
        k = l_2 - loop if loop > (l_2 := self.l // 2) else loop
        θmax, rmax = self.theta_max[k], self.rmax[k]

        Δs = Δθ = 0
        θ, r0 = start, self.r(start)
        theta, rs = [θ], [r0]
        while ((i := len(rs)) < n_max and
               (r := r0 + i * ds * np.sqrt((1 - Δθ ** 2))) < rmax and
               Δs < ds * 1.1):  # and

            θn = self.solve_theta_r_loop(r, loop, j)
            Δθ = np.abs(θn - θ)
            Δs = np.sqrt((Δr := r - rs[-1]) ** 2 + (Δθ * (r + Δr/2) ** 2))
            # print(f'{i=:} {Δθ=:} {θ=:} {ds=:}')

            theta.append(θn)
            rs.append(r)
            θ = θn

        rs.append(rmax)
        theta.append(θmax)

        direction = (1, -1)[j]
        return rs[::direction], theta[::direction]

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

    # def _approx_arc(self, loop, start):
    #     theta = get_theta2(self, loop, start=None, stop=None, Δs=0.05, res=100)
    #     r = self.r(theta)
    #     dθ = np.diff(theta)
    #     dr = np.diff(r)
    #     ds = np.sqrt(np.square([((r[:-1] + dr/2) * dθ), dr]).sum(0))
    #     return ds.sum()

    # def _solve_Δθ(self, θ0, Δs):
    #     return brentq(self._Δθ_objective, θ0, θ0 + π / self.l,  (θ0, Δs))

    # def _Δθ_objective(self, θ, θ0, Δs):
    #     return self.ds(θ) * (θ - θ0) - Δs

    def plot_r_cart(self, ax, res=100):
        from scipy.misc import derivative

        # intervals = list()
        first = True
        for i in mit.pairwise(self.get_zeros()):
            x = np.linspace(*(i + np.array([-1, 1]) * 1e-5), res)
            drdθ = derivative(self.r, x, np.ptp(x[:2]))

            ax.plot(x, drdθ, 'x', label='numeric dr/dθ')
            #ax.plot(x,  lpmv(0, l, np.cos(x)), ':', label=r'$P_{\ell}^0$')
            ax.plot(x, self.r(x), 'o', mfc='none', label='r(x, l)')
            ax.autoscale(False)

            ax.plot(x, self.dr(x), 'k-.', label='dr(x, l)')

            ax.plot(x, self.F(x), '--', label='F(x, l)')
            ax.plot(x, self.dF(x), ':', label='dF(x, l)')

            ax.plot(x, self.ds(x), '-', label='ds(x, l)')
            # ax.plot(x, self.dsfuck(x), '-', label='piss')

            ax.set_prop_cycle(None)
            if first:
                first = False
                ax.legend(loc='upper left')

        ax.vlines(self.theta_max, -3, 3, '0.7')
        ax.plot(self.θmax, self.rmax, '*')  # ,  color=line.get_color())
        ax.plot(zeros, self.r(zeros), 'ks')

        xticks = np.linspace(0, np.pi/2, 7)
        ax.set(xlabel=r'$\theta$',
               xticks=xticks,
               ylim=(-2.5, 2.5),
               )
        ax.grid()
        ax.set_ylabel(r'$r(\theta)$', rotation=0, labelpad=20)

        # ax.parasite.yaxis.offsetText.set_visible(False)
        ax.parasite.set(xlabel=r'$\theta$', xticks=xticks)
        ax.parasite.xaxis.set_major_formatter(pi_radian_formatter)
        #ax.set(yscale='log', ylim=(0.1, 100))
        ax.parasite.yaxis.set_ticklabels([])

    def plot2d(self, ax, loop=..., interval=(0, π), res=100, **kws):  # projection='polar'

        # TODO: cmap

        # calculate zeros of associated Legendre function, so we include those
        # points in the plotting domain
        self.get_theta()
        r = self._fieldline_radial(theta, phi).T

        # reflect
        theta = np.hstack([theta, theta + π])
        r = np.hstack([r, r])

        line, = ax.plot(theta, r, **kws)
        # line2, = ax.plot(theta + π, r, **kws)
        # ax.plot(theta, lpmv(1, l, np.cos(theta)))
        return line  # , line2


# alias
MultipoleFieldLine = MultipoleFieldline


class MagneticField(OriginLabelledAxes):
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


class IdealMultipole(MagneticField):  # IdealMultipole
    """Pure axissymetric dipole / quadrupole / octupole etc"""

    def __init__(self, degree=2):
        self.degree = degree

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree):
        """
        Degree of the multipole, typically denoted
        """
        if degree < 1:
            raise ValueError(f'Degree of magnetic multipole field must be '
                             f'greater than 1. For example:\n'
                             f'{NAMED_MULTIPOLE_ORDERS}')

        self._degree = int(degree)
        del self.theta_c

    l = degree

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
        r, θ, φ = cart2sph(*xyz)
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

    def _fieldline_radial(self, θ):
        """
        Axisymmetric field line radial distance from origin parameterized by θ
        """
        return r_multipole_fieldline(self.degree, np.asanyarray(θ))

    def fieldline(self, θ, φ, rscale=1):
        """
        Axisymmetric field line cartesian distance from origin.
        """
        r = rscale * self._fieldline_radial(θ)
        return np.moveaxis(np.asanyarray(sph2cart(r, θ, φ)), 0, -1)

    # def fieldline_r(self, rshell, res=50):
    #     theta_c = self.solve_surface_angles(rshell)

    def fieldlines(self, nshells=3, naz=5, res=100, bounding_box=None, rmin=0.,
                   scale=1):
        """
        Compute the field lines for a pure multipole magnetic field.

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
        bounding_box : int, optional
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
        fieldlines = self._fieldlines(shells, naz, res, bounding_box, rmin)

        # apply default spatial units
        fieldlines, flux_density = self._apply_default_spatial_units(fieldlines)

        # shift to origin
        fieldlines += self.origin.T

        return fieldlines, flux_density

    def get_theta_fieldline(self, res=50):
        return np.linspace(0, π, res)

    def _fieldlines(self, shells, naz=5, res=100, bounding_box=None, rmin=0.):

        shells = np.atleast_1d(shells).ravel()
        nshells = shells.size
        if isinstance(rmin, u.Quantity):
            rmin = rmin.to(self.origin.unit).value

        # radial profile of magnetcic field lines
        θ = self.get_theta_fieldline()
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

            # TODO: each field line is split into sections so that the
            # projection zorder gets calculated correctly for all viewing
            # angles. Fieldlines terminate on star surface. ???

            # Flux density unit
            flux_density *= op.AttrGetter('unit', default=1)(b)
        else:
            # convert to Cartesian coordinates
            line = self.fieldline(θ, φ)  # as 3D array
            fieldlines = line * shells[(np.newaxis, ) * line.ndim].T  # .reshape(-1, 3)
            flux_density = self.flux_density(fieldlines)

        return fieldlines, flux_density

    def plot2d(self, ax, res=100, phi=0, **kws):  # projection='polar'

        # TODO: cmap

        # calculate zeros of associated Legendre function, so we include those
        # points in the plotting domain
        theta = np.sort(np.hstack(get_theta_fieldlines(self.degree)))
        r = self._fieldline_radial(theta, phi).T

        # reflect
        theta = np.hstack([theta, theta + π])
        r = np.hstack([r, r])

        line, = ax.plot(theta, r, **kws)
        # line2, = ax.plot(theta + π, r, **kws)
        # ax.plot(theta, lpmv(1, l, np.cos(theta)))
        return line  # , line2

    # alias
    plot2D = plot2d


# alias
PureMultipole = IdealMultipole


class PhysicalMultipole(IdealMultipole, RotationMixin, AxesHelper):

    def __init__(self, magnitude=1, radius=1, degree=2, theta=0, phi=0):
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
            [description], by default 2
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

        r = rscale * self._fieldline_radial(θ)
        r = np.ma.masked_where(r < self.radius, r)
        if r.mask.any():
            warnings.warn('Masked points inside reference radius.')

        return np.moveaxis(np.asanyarray(sph2cart(r, θ, φ)), 0, -1)

    # def _angle_solver(self, θ, rshell):
    #     _angle_solver(self.l, θ, self.radius / rshell)

    def _fieldlines(self, shells, naz=5, res=100, bounding_box=None, rmin=0):
        #
        fieldlines, flux_density = super()._fieldlines(shells, naz, res,
                                                       bounding_box, rmin)

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

    def get_theta_fieldline(self, res=50):

        return get_theta_fieldlines(self.degree, self.radius)

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

    @ staticmethod
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

    @ property
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
    @ property
    def moment(self):
        """
        Magnetic moment vector in Cartesian coordinates. Magnetic moment has
        units of [J/T] or equivalent.
        """
        return self._moment.xyz

    @ moment.setter
    @ default_units(moment=JoulePerTesla)
    @ u.quantity_input(moment=['magnetic moment', 'dimensionless'])
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

    @ lazyproperty
    def direction(self):
        """Magnetic moment unit vector."""
        return (m := self._moment).xyz / m.norm()

    # alias
    magnetic_axis = direction

    @ lazyproperty
    def theta(self):
        """Magnetic moment colatitude."""
        return self._moment.rθφ.theta.value

    @ theta.setter
    def theta(self, theta):
        self._moment = SphericalCoords(self.phi, theta, self._moment.norm())
        del self.direction

    # alias
    θ = theta

    @ property
    def phi(self):
        """Magnetic moment azimuth."""
        return self._moment.rθφ.phi.value

    @ phi.setter
    def phi(self, phi):
        self._moment = SphericalCoords(phi, self.theta, self._moment.norm())
        del self.direction

    # alias
    φ = phi

    @ lazyproperty
    def _rotation_matrix(self):
        return EulerRodriguesMatrix(
            (np.sin(φ := self.phi), -np.cos(φ), 0), self.theta
        ).matrix

    # ------------------------------------------------------------------------ #

    @ default_units(xyz=u.dimensionless_unscaled)  # convert to quantity
    @ u.quantity_input(xyz=['length', 'dimensionless'])
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

    @ staticmethod
    def fieldline(θ, φ):
        θ, φ = np.broadcast_arrays(θ, φ)
        return np.moveaxis(np.asanyarray(sph2cart(np.sin(θ) ** 2, θ, φ)), 0, -1)

    def fieldlines(self, nshells=3, naz=5, res=100, bounding_box=None, rmin=0.,
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
        bounding_box : int, optional
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
        return self._fieldlines(shells, naz, res, bounding_box, rmin)
        # return xyz, flux

    def _fieldlines(self, shells, naz=5, res=100, bounding_box=None, rmin=0.):

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
        fieldlines = self.clipped(fieldlines, bounding_box)

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
               rmin=0., bounding_box=None, cmap='jet', **kws):

        # collate artist graphical properties
        kws = {**ARTIST_PROPS_3D.bfield, **kws}

        # create field lines
        fieldlines, flux = self.fieldlines(nshells, naz, res, bounding_box,
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
    @ property
    def centre_offset(self):
        """Offset position of magnetic field origin from stellar centre."""
        return self.host.origin - self.origin

    @ centre_offset.setter
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

    @ property
    def η(self):
        """
        The dimensionless coordinate quantity *η* gives the fractional
        x-distance from the stellar centre.
        """
        return self.centre_offset[0]

    @ η.setter
    def η(self, x):
        assert x < 1
        self.centre_offset[0] = float(x) * self.host.radius

    @ property
    def ξ(self):
        """
        The dimensionless coordinate quantity *ξ* gives the fractional
        y-distance from the stellar centre.
        """
        return self.centre_offset[1]

    @ ξ.setter
    def ξ(self, x):
        assert x < 1
        self.centre_offset[1] = float(x) * self.host.radius

    @ property
    def ζ(self):
        """
        The dimensionless coordinate quantity *ζ* gives the fractional
        z-distance from the stellar centre.
        """
        return self.centre_offset[2]

    @ ζ.setter
    def ζ(self, x):
        assert x < 1
        self.centre_offset[2] = float(x) * self.host.radius

    @ lazyproperty
    def north_pole_flux(self):
        """
        The maximal magnetic flux density on the stellar surface. This is a
        proxy for the magnetic moment vector.
        """
        return μ0_2π * np.linalg.norm(self.moment) / self.host.R ** 3

    @ north_pole_flux.setter
    @ default_units(north_pole_flux=u.MG)
    @ u.quantity_input(north_pole_flux=['magnetic flux density'])
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
