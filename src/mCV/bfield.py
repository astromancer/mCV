
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
from .roche import Axes3DHelper


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

    def flux_density(self, r):
        """Flux density at vector at position r"""

        # If r is unitless, assume its given ito orbital semi-major axis *a*
        # if isinstance(r, Quantity):

        r = np.array(r)

        rd = np.linalg.norm(r)  # *
        rhat = r / rd
        m = self.moment
        return mu0 * (3 * rhat * (m * rhat).sum(0) - m) / (4 * π * rd ** 3)

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
            # NOTE: each field line is split into sections so that the
            # projection zorder gets calculated correctly for all viewing angles.
            # Fieldlines terminate on star surface.
            shells = []
            for i in range(1, nshells + 1):
                theta0 = np.arcsin(np.sqrt(rmin / scale / i))
                # self.surface_theta.append(theta0 + self.phi)

                break1 = π / 4
                intervals = mit.pairwise((theta0, break1,  π / 2 + break1, π - theta0))
                for th0, th1 in intervals:
                    θ = np.linspace(th0, th1, res)
                    θ = np.append(θ, np.nan)  # so the shells are not connected
                    shells.append(i * dipole_fieldline(θ, φ))
                    # break
            fieldlines = np.rollaxis(np.array(shells), 1, 4
                                     ).reshape((nshells, -1, 3))
            # return fieldlines
        else:
            # convert to Cartesian coordinates
            shells = np.arange(1, nshells + 1)[None, None].T  # as 3D array
            fieldlines = shells * np.rollaxis(dipole_fieldline(θ, φ),
                                              0, 3).reshape(-1, 3)

        # tilt
        if self.az or self.alt:
            # 3D rotation!
            fieldlines = np.rollaxis(np.tensordot(self.R, fieldlines, [0, -1]),
                                     0, fieldlines.ndim)

        # scale
        fieldlines *= scale

        # shift to origin
        fieldlines += self.origin.T

        bbox = bounding_box
        if bbox:
            if np.size(bbox) <= 1:
                bbox = np.tile([-1, 1], (3, 1)) * bbox
            xyz = (fieldlines - self.origin.T) - bbox.mean(1)
            fieldlines[(xyz < bbox[:, 0]) | (bbox[:, 1] < xyz)] = np.ma.masked

        return fieldlines

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
