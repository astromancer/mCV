
# std
import numbers
import warnings

# third-party
import numpy as np
from matplotlib import ticker
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from astropy import units as u
from astropy.constants import mu0
from astropy.utils.decorators import lazyproperty
from astropy.coordinates import (BaseRepresentation, CartesianRepresentation,
                                 PhysicsSphericalRepresentation as SphericalCoords)

# local
from recipes import op
from recipes.array import fold
from recipes.transforms import sph2cart
from recipes.dicts import pformat as pformat_dict
from recipes.transforms.rotation import EulerRodriguesMatrix

# relative
from .roche import ARTIST_PROPS_3D
from .axes_helpers import OriginInAxes, get_axis_label
from .utils import default_units, get_value


# ---------------------------------------------------------------------------- #
π = np.pi
_2π = 2 * π
_4π = 4 * π

JoulePerTesla = u.J / u. T
MOMENT_DEFAULT = 1 * JoulePerTesla
ORIGIN_DEFAULT = (0, 0, 0) * u.m


_POW10 = {-1: 0.1,
          0:  1,
          1:  10,
          2:  100}

# ---------------------------------------------------------------------------- #


def format_x10(n, _pos=None):
    n = int(n)
    return str(_POW10.get(n, None) or rf'10^{{{n}}}').join('$$')


def dipole_fieldline(θ, φ):
    θ, φ = np.broadcast_arrays(θ, φ)
    return np.asanyarray(sph2cart(np.sin(θ) ** 2, θ, φ))


def _repr_helper(obj, *attrs, **kws):
    return pformat_dict({att: getattr(obj, att) for att in attrs},
                        type(obj).__name__, str, '=', brackets='()', **kws)

# ---------------------------------------------------------------------------- #
# class PPrint:
#     def __init__(self)


class CartesianCoords(CartesianRepresentation):

    @property
    def rθφ(self):
        """Spherical coordinate representation."""
        return self.represent_as(SphericalCoords)


class MagneticField:
    """Abstract Base Class for MagneticField models."""

    @classmethod
    def dipole(cls, origin=ORIGIN_DEFAULT, moment=MOMENT_DEFAULT, *,
               theta=0, phi=0):
        return MagneticDipole(origin, moment, theta=theta, phi=phi)


class MagneticDipole(OriginInAxes):
    # TODO bounding box
    # TODO plot2D
    # TODO multipoles

    # https://en.wikipedia.org/wiki/Magnetic_dipole
    # https://en.wikipedia.org/wiki/Magnetic_moment

    # _unit_converter = default_units()

    def __init__(self, origin=ORIGIN_DEFAULT, moment=MOMENT_DEFAULT, *,
                 theta=0, phi=0):
        """
        A magnetic dipole located at *origin* with magnetic *moment*. 
        Direction of magnetic moment can be expressed by passing a 3-vector 
        moment, or alternately specifying inclination *theta* and azimuth *phi* 
        angles in randians.

        Parameters
        ----------
        origin : tuple, optional
            Field centre, by default (0, 0, 0). Default units are in
            semi-major-axis scale.
        moment :  float, tuple, np.ndarray, Quantity, optional
            Magnetic moment, by default (0, 0, 1). Default units are in J/T or
            equivalently A m2.
        theta : float, optional
            Altitude angle of magnetic moment vector in degrees, by default 0.
        phi : float, optional
            Azimuth angle of magnetic moment vector in degrees, by default 0.

        Examples
        --------
        >>> 
        """
        self._origin = None  # placeholder
        self.origin = origin

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
    def _R(self):
        return EulerRodriguesMatrix(
            (np.sin(φ := self.phi), -np.cos(φ), 0), self.theta
        ).matrix

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
        v = xyz / r  # rhat
        m = self.moment

        # catch all warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('ignore')

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

    def fieldlines(self, nshells=3, naz=5, res=100, bounding_box=None, rmin=0.):
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
        shells = np.arange(1, nshells + 1)
        return self._fieldlines(shells, naz, res, bounding_box, rmin)
        # return xyz, flux

    @staticmethod
    def fieldline(θ, φ):
        θ, φ = np.broadcast_arrays(θ, φ)
        return np.moveaxis(np.asanyarray(sph2cart(np.sin(θ) ** 2, θ, φ)), 0, -1)

    def _fieldlines(self, shells, naz=5, res=100, bounding_box=None, rmin=0.):

        shells = np.atleast_1d(shells).ravel()
        nshells = shells.size

        # radial profile of magnetcic field lines
        θ = np.linspace(0, π, res)
        φ = np.linspace(0, _2π, naz, endpoint=False)[None].T
        # self.surface_theta = []
        if rmin:
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
                flux_unit = op.AttrGetter('unit.si', default=1)(b)

            # TODO: each field line is split into sections so that the
            # projection zorder gets calculated correctly for all viewing
            # angles. Fieldlines terminate on star surface. ???
            # fieldlines = fieldlines.reshape((-1, res, 3))
            # flux_density = flux_density.reshape((-1, res))
            # return fieldlines
            flux_density *= flux_unit
        else:
            # convert to Cartesian coordinates
            line = self.fieldline(θ, φ)  # as 3D array
            fieldlines = line * shells[(np.newaxis, ) * line.ndim].T  # .reshape(-1, 3)
            flux_density = self.flux_density(fieldlines)

        # tilt
        if self.phi or self.theta:
            # 3D rotation!
            fieldlines = np.rollaxis(np.tensordot(self._R, fieldlines, (0, -1)),
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

        fieldlines, flux = self.fieldlines(nshells, naz, res, bounding_box, rmin)
        flux_unit = op.AttrGetter('unit.si', default=None)(flux)

        kws = {**ARTIST_PROPS_3D.bfield,  **kws}
        #
        do_cmap = (cmap is not False)
        if do_cmap:
            segments = fold.fold(fieldlines, 2, 1, axis=-2, pad=False).reshape((-1, 2, 3))
            flux = fold.fold(flux, 2, 1, axis=-1, pad=False).mean(-1).reshape(-1)
            kws.update(array=np.log10(flux))
        else:
            segments = fieldlines

        #
        art = Line3DCollection(segments, **kws)
        ax = ax or self.axes
        ax.add_collection3d(art)
        ax.auto_scale_xyz(*get_value(fieldlines).T)

        if do_cmap:
            cb = ax.figure.colorbar(art, ax=ax, pad=0.01, shrink=0.9, aspect=30,
                                    ticks=ticker.MaxNLocator(integer=True),
                                    format=ticker.FuncFormatter(format_x10))
            # \log_{10}\left(
            label = get_axis_label(r'B\left(r, \theta)\right)', flux_unit)
            cb.ax.text(0, 1.01, label, transform=cb.ax.transAxes, ha='left')
        return art, cb


class OffsetCentreDipole(MagneticDipole):
    def __init__(self, wd, offset=(0, 0, 0), moment=MOMENT_DEFAULT, *,
                 theta=0, phi=0):
        #
        # offset /
        super().__init__(origin=wd.origin + offset,
                         moment=moment, theta=theta, phi=phi)


# aliases
OffsetCenterDipole = OffsetDipole = OffsetCentreDipole


# class MagneticMultipole
