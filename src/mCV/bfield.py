"""
Magnetic field models for stars
"""

# std
import numbers
import warnings

# third-party
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
from recipes.functionals import raises
from recipes.string import named_items
from recipes.transforms import sph2cart
from recipes.dicts import pformat as pformat_dict
from recipes.transforms.rotation import EulerRodriguesMatrix

# relative
from .roche import ARTIST_PROPS_3D, Ro
from .axes_helpers import OriginInAxes, get_axis_label
from .utils import _check_optional_units, default_units, get_value, has_unit
from .plotting_helpers import (degree_as_pi_frac_formatter, theta_tickmarks,
                               x10_formatter)


# ---------------------------------------------------------------------------- #
π = np.pi
_2π = 2 * π
_4π = 4 * π
μ0_2π = mu0 / _2π

JoulePerTesla = u.J / u. T
DIPOLE_MOMENT_DEFAULT = 1 * JoulePerTesla
ORIGIN_DEFAULT = (0, 0, 0) * Ro


NAMED_MULTIPOLE_ORDERS = {
    # monopole
    'dipole':               1,
    'quadrupole':           2,
    'hexapole':             3,
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


def dipole_fieldline(θ, φ):
    θ, φ = np.broadcast_arrays(θ, φ)
    return np.asanyarray(sph2cart(np.sin(θ) ** 2, θ, φ))


def legendre_zeros(l):
    """Zeros of the associated Legendre function of degree l and order 1."""

    # we know there is a zero at cosθ = π/2 if l is even
    even = (l % 2) == 0
    zeros = [np.pi / 2] * even

    if l <= 2:
        return zeros

    # Lacroix '84
    v = np.arccos(np.sqrt(1 - (1 / (l * (l + 1)))))
    intervals = np.cos(np.linspace(v, np.pi - v, l))
    intervals = list(mit.pairwise(intervals))

    if even:  # l is even
        intervals.pop((l - 1) // 2)  # already have this zero at cosθ = π/2

    for interval in intervals:
        x0 = brentq(objective_legendre_zeros, *interval, (l, ))
        zeros.append(np.arccos(x0))

    return zeros


def objective_legendre_zeros(x, l):
    return lpmv(1, l, x)


def _repr_helper(obj, *attrs, **kws):
    return pformat_dict({att: getattr(obj, att) for att in attrs},
                        type(obj).__name__, str, '=', brackets='()', **kws)


# ---------------------------------------------------------------------------- #


def plot2d_multipoles(n, nrows=None, ncols=2, **kws):

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
        b = PureMultipole(0, l)

        ax = axes[divmod(i, 2)]
        ax.set_theta_zero_location('N')
        b.plot2d(ax, 500)

        # label degree l
        ax.text(-0.125, 1.035,
                fr'$\bf \ell = {l}$',
                size=14,
                transform=ax.transAxes)
        #ax.plot(theta, lpmv(1, l, np.cos(theta)))

    # radial ticks
    ax.set_yticks(np.linspace(0, 1, 5)[1:])

    # angular ticks
    ax.xaxis.major.formatter = degree_as_pi_frac_formatter

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


# ---------------------------------------------------------------------------- #
class MagneticField(OriginInAxes):
    """Base Class for Magnetic Field models."""

    def __init__(self, origin=ORIGIN_DEFAULT, moment=None, coeff=None, *,
                 theta=0, phi=0):
        """
        A magnetic field centred at *origin* with magnetic moment tensor
        *moment*.
        """

        #
        OriginInAxes.__init__(self, origin)

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


class PureMultipole(MagneticField):
    """Pure dipole / quadrupole / octupole etc"""

    def __init__(self, magnitude=1, degree=2, theta=0, phi=0):

        self._magnitude = self._theta = self._phi = None  # placeholders
        self.magnitude = magnitude
        self.degree = degree
        self.theta = theta
        self.phi = phi

    @property
    def magnitude(self):
        return self._magnitude

    @default_units(magnitude=u.dimensionless_unscaled)  # convert to quantity
    @u.quantity_input(magnitude=['magnetic flux density', 'dimensionless'])
    @magnitude.setter
    def magnitude(self, magnitude):
        assert magnitude >= 0
        self._magnitude = magnitude

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree):
        if degree < 1:
            raise ValueError(f'Degree of magnetic multipole field must be '
                             f'greater than 1. For example:\n'
                             f'{NAMED_MULTIPOLE_ORDERS}')

        self._degree = int(degree)

    @lazyproperty
    def direction(self):
        """
        Directional unit vector of the magnetic axis. ie. The normalized dipole
        magnetic moment in Cartesian coordinates.
        """
        return sph2cart(1, self.theta, self.phi)

    @lazyproperty
    def theta(self):
        """Colatitude of magnetic axis."""
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = float(theta)
        del self.direction

    θ = theta

    @property
    def phi(self):
        """Magnetic moment azimuth."""
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = float(phi)
        del self.direction

    φ = phi

    @lazyproperty
    def _rotation_matrix(self):
        return EulerRodriguesMatrix(
            (np.sin(self.phi), -np.cos(self.phi), 0), self.theta
        ).matrix

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

    def _fieldline_radial(self, θ, φ):
        """Axisymmetric field line"""
        θ, φ = np.broadcast_arrays(θ, φ)
        l = self.degree
        k = -np.sqrt(2 * factorial(l - 1) / factorial(l + 1))
        return np.abs(np.sin(θ) * k * lpmv(1, l,  np.cos(θ))) ** (1 / l)

    def fieldline(self, θ, φ):
        """Axisymmetric field line"""
        θ, φ = np.broadcast_arrays(θ, φ)
        l = self.degree
        k = -np.sqrt(2 * factorial(l - 1) / factorial(l + 1))
        r = np.abs(np.sin(θ) * k * lpmv(1, l,  np.cos(θ))) ** (1 / l)
        return np.moveaxis(np.asanyarray(sph2cart(r, θ, φ)), 0, -1)

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

    def plot2d(self, ax, res=100, phi=0, **kws):  # projection='polar'

        # TODO: cmap

        # calculate zeros of associated Legendre function, so we include those
        # points in the plotting domain

        theta0 = legendre_zeros(self.degree)
        theta = np.sort(np.hstack([theta0, np.linspace(0, np.pi, res)]))
        r = self._fieldline_radial(theta, phi).T

        # reflect
        theta = np.hstack([theta, theta + np.pi])
        r = np.hstack([r, r])

        line, = ax.plot(theta, r, **kws)
        #line2, = ax.plot(theta + np.pi, r, **kws)
        # ax.plot(theta, lpmv(1, l, np.cos(theta)))
        return line  # , line2

    # alias
    plot2D = plot2d


class MagneticMultipoleField(MagneticField):
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
        # sequence associated Legendre function of the first kind of order up to
        # m and degree n

        # l =  np.arange(1, self.degree)
        sinθ = np.sin(θ)
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

    def H(self, r,  θ, φ):
        # since the gauss coefficients already contain the vacuum permiability
        # coefficient, here we simply divide by that to get the H field
        return self.B(r,  θ, φ) / mu0


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
        OriginInAxes.__init__(self, origin)

        # TODO: for higher order multipoles, the moment is a tensor

        # get magnetic moment
        if isinstance(moment, numbers.Real) or \
                (isinstance(moment, np.ndarray) and moment.size == 1):
            # NOTE Quantity isa ndarray
            moment = moment * np.array(sph2cart(1, *np.radians((theta, phi))))

        self.moment = moment

    def __repr__(self):
        return _repr_helper(self, 'origin', 'moment')

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

    @lazyproperty
    def theta(self):
        """Magnetic moment colatitude."""
        return self._moment.rθφ.theta.value

    @theta.setter
    def theta(self, theta):
        self._moment = SphericalCoords(self.phi, theta, self._moment.norm())
        del self.direction

    θ = theta

    @property
    def phi(self):
        """Magnetic moment azimuth."""
        return self._moment.rθφ.phi.value

    @phi.setter
    def phi(self, phi):
        self._moment = SphericalCoords(phi, self.theta, self._moment.norm())
        del self.direction

    φ = phi

    @lazyproperty
    def _rotation_matrix(self):
        return EulerRodriguesMatrix(
            (np.sin(φ := self.phi), -np.cos(φ), 0), self.theta
        ).matrix

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

    @staticmethod
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


class StellarMagneticField(MagneticField):
    """
    Dipole magnetic field with origin offset from the stellar centre. Magnetic
    moment can be specified by setting the maximal surface flux density at the
    magnenetic north pole via the `B0` attribute.
    """

    def __init__(self, star,
                 pole_flux=None,
                 centre_offset=ORIGIN_DEFAULT,
                 moment=DIPOLE_MOMENT_DEFAULT,
                 theta=0, phi=0):

        #
        _check_optional_units(locals(), pole_flux=['magnetic flux density'])

        if pole_flux is not None:
            moment = (_2π * pole_flux * star.R ** 3 / mu0)

        #
        super().__init__(origin=star.origin + centre_offset,
                         moment=moment, theta=theta, phi=phi)
        self.host = star

    @property
    def centre_offset(self):
        """Offset position of magnetic field origin from stellar centre."""
        return self.host.origin - self.origin

    @centre_offset.setter
    def centre_offset(self, offset):
        if np.linalg.norm(offset) > self.host.R:
            raise ValueError('Magnetic field origin cannot lie beyond stellar '
                             'surface.')

        MagneticField.origin.fset(self.host.origin + offset)

    # alias
    center_offset = centre_offset

    @lazyproperty
    def pole_flux(self):
        """
        The maximal magnetic flux density on the stellar surface. This is a
        proxy for the magnetic moment vector.
        """
        return μ0_2π * np.linalg.norm(self.moment) / self.host.R ** 3

    @pole_flux.setter
    @default_units(pole_flux=u.MG)
    @u.quantity_input(pole_flux=['magnetic flux density'])
    def pole_flux(self, pole_flux):
        self.moment = pole_flux * self.host.R ** 3 / μ0_2π

    # alias
    B0 = pole_flux

    # @MagneticField.moment.setter
    # def moment(self, moment):
    #     MagneticField.moment.fset(moment)
    #     del self.pole_flux


# aliases
# EccentricDipole = OffsetCenterDipole = OffsetCentreDipole = OffsetDipole
MagneticMultipole = MagneticMultipoleField
