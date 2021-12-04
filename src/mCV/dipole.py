
# third-party
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# local
from recipes.transforms.rotation import rotation_matrix_3d


# class MagneticMultipole


class MagneticDipole:
    # TODO bounding box
    # TODO multicolor shells?
    # TODO plot2D
    # TODO tilted dipole!
    # TODO multipoles

    def __init__(self, center=(0, 0, 0), inclination=0, azimuth=0):
        self.center = center
        self.phi = np.radians(inclination)
        self.az = azimuth

    def fieldlines(self, nshells=3, naz=5, res=100, scale=1., bounding_box=0,
                   rmin=1.):
        """Compute the field lines for a dipole magnetic field"""

        # radial profile of B-field lines
        # Re = 1
        theta = np.linspace(0, np.pi, res)
        sinθ = np.sin(theta)
        r = sinθ * sinθ # * Re 
        rsinθ = r * sinθ

        phi = np.c_[np.linspace(0, 2 * np.pi, naz + 1)[:-1]]  # wraps at 2π

        # convert to Cartesian coordinates
        L = scale * np.r_['-1,3,0', range(1, nshells + 1)]  # as 3D array
        fieldlines = L * np.r_['-1,3,0',
                               rsinθ * np.cos(phi),
                               rsinθ * np.sin(phi),
                               r * np.cos(theta) * np.ones_like(phi)
                               ].reshape(-1, 3)
        # fieldlines *= scale

        # from IPython import embed
        # embed(header="Embedded interpreter at 'dipole.py':45")
        
        
        # tilt
        if self.phi:
            # 3D rotation!
            R = rotation_matrix_3d((0, 1, 0), np.radians(self.phi))
            fieldlines = np.tensordot(R, fieldlines, [0, -1]).transpose(1, 2, 0)

        # mask fieldlines inside WD
        if rmin:
            L = np.tile(np.linalg.norm(fieldlines.T, axis=0) <= rmin, (3, 1, 1))
            fieldlines = np.ma.masked_array(fieldlines, mask=L.T)

        # shift to center
        fieldlines += self.center.T

        bbox = bounding_box
        if bbox:
            if np.size(bbox) <= 1:
                bbox = np.tile([-1, 1], (3, 1)) * bbox
            xyz = (fieldlines - self.center.T) - bbox.mean(1)
            fieldlines[(xyz < bbox[:, 0]) | (bbox[:, 1] < xyz)] = np.ma.masked

        return fieldlines

    def boxed(self, fieldlines, box):
        box = 3

    def plot3d(self, ax, nshells=3, naz=5, **kw):
        #
        # ax = kw.get('ax', None):
        # if ax
        segments = self.fieldlines(nshells, naz)
        ax.add_collection3d(Line3DCollection(segments, **kw))
        ax.auto_scale_xyz(*segments.T)

        # dipole = MagneticDipole()
