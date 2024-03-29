

# third-party
import numpy as np
from astropy import units as u

# relative
from .utils import default_units


class Origin:
    """
    Origin property for locating objects in space.
    """

    _origin = None  # placeholder

    def __init__(self, origin):
        self.origin = origin

    @property
    def origin(self):
        """
        Location of object (or field) centre in Cartesain coordinates.
        """
        return self._origin

    @origin.setter
    @default_units(origin=u.dimensionless_unscaled)
    @u.quantity_input(origin=['length', 'dimensionless'])
    def origin(self, origin):
        origin = self._apply_default_spatial_units(np.asanyarray(origin))
        assert origin.size == 3
        self._origin = origin

    centre = center = origin

    def _apply_default_spatial_units(self, xyz):
        if (u := getattr(self.origin, 'unit', None)):
            xyz = default_units(xyz=u).apply(xyz)
        return xyz
