
# std
import numbers

# third-party
import numpy as np
import more_itertools as mit
from astropy import units as u
from astropy.constants import mu0
from astropy.units.quantity import Quantity
from astropy.utils.decorators import lazyproperty
from astropy.coordinates import (BaseRepresentation, CartesianRepresentation,
                                 PhysicsSphericalRepresentation)

# local
from recipes.transforms import sph2cart
from recipes.transforms.rotation import EulerRodriguesMatrix

# relative
from .utils import get_value
from .roche.core import SpatialAxes3D, get_unit_string


π = np.pi

JoulePerTesla = u.J / u. T
MOMENT_DEFAULT = 1 * JoulePerTesla


def dipole_fieldline(θ, φ):

class CartesianRepresentation(CartesianRepresentation):

    @property
    def rθφ(self):
        return self.represent_as(PhysicsSphericalRepresentation)


class MagneticField:
    """ABC for MagneticField models"""

    @classmethod
    def dipole(cls, origin=(0, 0, 0), moment=MOMENT_DEFAULT, *,
               theta=0, phi=0):
        return MagneticDipole(origin, moment, theta=theta, phi=phi)


class MagneticDipole(Axes3DHelper):
    # TODO bounding box
    # TODO multicolor shells?
    # TODO plot2D
    # TODO multipoles

    def __init__(self, origin=(0, 0, 0), moment=MOMENT_DEFAULT, *,
                 theta=0, phi=0):
        """
        [summary]

        Parameters
        ----------
        origin : tuple, optional
            [description], by default (0, 0, 0)
        moment : tuple, optional
            Magnetic moment, by default (0, 0, 1)
        alt : [type], optional
            [description], by default None
        az : [type], optional
            [description], by default None

        Examples
        --------
        >>> 
        """
        self.origin = np.asanyarray(origin)
        
        if isinstance(moment, numbers.Real) or \
                (isinstance(moment, np.ndarray) and moment.size == 1):
            # NOTE Quantity isa ndarray
            moment = moment * np.array(sph2cart(1, *np.radians((theta, phi))))
        
        self.moment = moment


    def __repr__(self):
        return _repr_helper(self, 'origin', 'moment')

    @property
    def moment(self):
        return self._moment.xyz

    @moment.setter
    def moment(self, moment):
        if isinstance(moment, BaseRepresentation):
            self._moment = moment.represent_as(CartesianRepresentation)
            del self.direction
            return

        if isinstance(moment, numbers.Real) or \
                (isinstance(moment, np.ndarray) and moment.size == 1):
            # note Quantity isa ndarray
            direction = sph2cart(1, *np.radians((self.theta, self.phi)))
            moment = moment * np.array(direction)

        assert len(moment) == 3
        if not isinstance(moment, Quantity):
            moment = np.array(moment) * JoulePerTesla
            # direction = (v := moment.value) / np.linalg.norm(v)

        self._moment = CartesianRepresentation(moment)
        del self.direction

    @lazyproperty
    def direction(self):
        return (m := self._moment).xyz / m.norm()

    @lazyproperty
    def theta(self):
        return self._moment.rθφ.theta.value

    @theta.setter
    def theta(self, theta):
        self._moment = PhysicsSphericalRepresentation(self.phi, theta, self._moment.norm())
        del self.direction

    θ = theta

    @property
    def phi(self):
        return self._moment.rθφ.phi.value

    @phi.setter
    def phi(self, phi):
        self._moment = PhysicsSphericalRepresentation(phi, self.theta, self._moment.norm())

    φ = phi

    @lazyproperty
    def _R(self):
        return EulerRodriguesMatrix(
            (np.sin(self.phi), -np.cos(self.phi), 0), self.theta
        ).matrix

    def H(self, xyz):
        """
        Flux density (H-field) at cartesian position *xyz*.
        """

        assert xyz.shape[-1] == 3
        r = np.array(xyz)
        rd = np.linalg.norm(r, axis=-1, keepdims=True)
        rhat = r / rd
        m = self.moment
        return (3 * rhat * (m * rhat).sum(-1, keepdims=True) - m) / (4 * π * rd ** 3)

    def B(self, xyz):
        return mu0 * self.H(xyz)

    def flux_density(self, xyz):
        return np.linalg.norm(self.B(xyz), axis=-1)

    # def fieldline(self, theta, phi)

    def fieldlines(self, nshells=3, naz=5, res=100, scale=1., bounding_box=None,
                   rmin=0.):
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

        # radial profile of B-field lines
        # Re = 1
        θ = np.linspace(0, π, res)
        φ = np.linspace(0, 2 * π, naz, endpoint=False)[None].T
        # self.surface_theta = []
        if rmin:
            fieldlines = np.empty((nshells, naz, res, 3))
            flux_density = np.empty((nshells, naz, res))
            φ = np.linspace(0, 2 * π, naz, endpoint=False)[None].T

            for i in range(1, nshells + 1):
                theta0 = np.arcsin(np.sqrt(rmin / scale / i))
                θ = np.linspace(theta0, π - theta0, res)
                fieldlines[i-1] = xyz = np.moveaxis(i * dipole_fieldline(θ, φ), 0, -1)

                # Note: since flux density indep of phi, this calculation is not
                # repeated unnecessarily for phi above
                flux_density[i-1] = self.flux_density(xyz[0])

            # embed()

            # TODO: each field line is split into sections so that the
            # projection zorder gets calculated correctly for all viewing
            # angles. Fieldlines terminate on star surface. ???
            # fieldlines = fieldlines.reshape((-1, res, 3))
            # flux_density = flux_density.reshape((-1, res))
            # return fieldlines
        else:
            # convert to Cartesian coordinates
            line = dipole_fieldline(θ, φ)  # as 3D array
            shells = np.arange(1, nshells + 1)[(np.newaxis, ) * line.ndim].T
            fieldlines = np.linalg.norm(
                self.moment) * shells * np.rollaxis(line, 0, 3)  # .reshape(-1, 3)
            flux_density = self.flux_density(fieldlines)

        # tilt
        if self.phi or self.theta:
            # 3D rotation!
            fieldlines = np.rollaxis(np.tensordot(self._R, fieldlines, (0, -1)),
                                     0, fieldlines.ndim)

        origin = get_value(self.origin).T
        bbox = bounding_box
        if bbox:
            if np.size(bbox) <= 1:
                bbox = np.tile([-1, 1], (3, 1)) * bbox
            xyz = fieldlines - bbox.mean(1)
            fieldlines[(xyz < bbox[:, 0]) | (bbox[:, 1] < xyz)] = np.ma.masked

        # scale
        fieldlines *= scale

        # shift to origin
        fieldlines += origin

        return fieldlines, flux_density  # .ravel()

    def boxed(self, fieldlines, box):
        box = 3

    def plot3d(self, ax=None, nshells=3, naz=5, res=100, scale=1.,
               bounding_box=None, rmin=0., **kws):
        #
        # ax = kws.get('ax', None):
        # if ax

        segments = self.fieldlines(nshells, naz, res, scale, bounding_box,
                                   rmin)
        # segments
        art = Line3DCollection(segments, **kws)
        ax = ax or self.axes
        ax.add_collection3d(art)
        ax.auto_scale_xyz(*segments.T)

        return art


class OffsetCentreDipole(MagneticDipole):
    def __init__(self, wd, offset=(0, 0, 0), moment=1 * u.Unit('J/T'), *,
                 alt=None, az=None):
        #
        # offset / 
        super().__init__(origin=wd.origin + offset,
                         moment=moment, alt=alt, az=az)


# aliases
OffsetCenterDipole = OffsetDipole = OffsetCentreDipole


# class MagneticMultipole
