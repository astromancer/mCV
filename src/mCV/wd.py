"""
Classes to represent White Dwarf stars.
"""


# std
import textwrap as txw

# third-party
import numpy as np
from matplotlib.patches import Circle
from astropy import units as u
from astropy.constants import mu0
from astropy.utils.decorators import lazyproperty

# local
from recipes import pprint as pp
from recipes.transforms import sph2cart

# relative
from .axes_helpers import OriginInAxes
from .utils import _check_units, default_units
from .bfield import MagneticDipole, MagneticField
from .roche.core import ARTIST_PROPS_2D, ARTIST_PROPS_3D, Mo, Ro


π = np.pi

RESOLUTION = 25
ZERO = (0, 0, 0) * Ro

# TODO: various types of WDs exist


# def Chandrasekhar(M):
#     """

#     Parameters
#     ----------
#     M

#     Returns
#     -------

#     References
#     ----------
#     An introduction to the Study of stellar structure
#     Chandrasekhar, S.
#     http://adsabs.harvard.edu/abs/1939C&T....55..412C

#     """
#     raise NotImplementedError

class MassRadiusRelation:
    """Base class for White Dwarf mass-radius relation."""


class Nauenberg(MassRadiusRelation):
    """
    The Nauenberg Mass-Radius relation for White Dwarfs.
    """

    def __init__(self, mu=2):
        """
        The Nauenberg Mass-Radius relation for White Dwarfs with a given average
        molecular weight *mu*.

        Parameters
        ----------
        mu : int, optional
            The average molecular weight per electron in the stellar plasma.
            He-4 C-12 and O-16 which predominantly compose white dwarf all
            have atomic # number equal to half their atomic weight, one should
            take μe equal to 2, which is the default.

        Examples
        --------
        >>> 
        """
        self.mu = mu
        self.M3 = 5.816 / self.mu ** 2
        self.k = 0.0225 / mu
        self.kσ = self.k / 18 / self.M3

    def __call__(self, m):
        """
        Compute WD radius in units of solar radii from input mass in solar mass
        units.

        Parameters
        ----------
        M

        Returns
        -------
        R : float
            The White dwarf radius in solar radii

        References
        ----------
        "Analytic Approximations to the Mass-Radius Relation and Energy of Zero-Temperature Stars"
        Nauenberg (1972)
        http://adsabs.harvard.edu/full/1972ApJ...175..417N
        """

        mm3 = np.cbrt(m / self.M3)
        return self.k * np.sqrt(1. - mm3 ** 4) / mm3

    def std(self, m, σm):
        # linear propagation of uncertainty
        mm3 = np.cbrt(m / self.M3)
        return abs(self.kσ * (1 - 2 / mm3) / np.sqrt(1 - mm3) / mm3 ** 5 * σm)

    @default_units(r=Ro)
    @u.quantity_input(r='length')
    def inverse(self, r):
        return self.M3 * ((r.to(Ro).value / self.k) ** 2 + 1) ** (-3/2)


# class Carvalho(MassRadiusRelation):
#     c = [20.86, 0.66, 2.48e-5, -2.43e-5]
#     np.polyval(c, r)


class WhiteDwarf(OriginInAxes):
    """
    Object representing a White Dwarf star.
    """

    # Mass radius relation unit of solar radii
    mass_radius = Nauenberg()

    def __init__(self, mass=1 * Mo, centre=ZERO, **kws):
        """
        Object representing a White Dwarf star.

        Parameters
        ----------
        mass : float or Quantity, optional
            Mass, by default 1*Mo
        centre : tuple, optional
            Location, by default (0, 0, 0)

        Examples
        --------
        >>> 

        Raises
        ------
        TypeError
            [description]
        """
        #
        _check_units(locals(), {'mass': 'mass',
                                'centre': ['length', 'dimensionless']})

        self._mass = mass
        self.origin = centre

        self.artists = []

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'M={self.mass.value:.3g} M⊙, '
                f'R={self.radius.to(u.km):4.1f})')

    def pformat(self):
        return txw.dedent(f'''\
            WD radius         : R1 = {self.R.to(Ro).value:.5f} R⊙
                                   = {self.R.to(u.R_earth).value:.5f} R⊕
                                   = {pp.nr(self.R.to('km').value)} km'''
                          )

    def pprint(self):
        """Pretty print binary parameters"""
        print(self.pformat())

    @property
    def mass(self):
        return self._mass

    @mass.setter
    @default_units(mass=Mo)
    @u.quantity_input(mass='mass')
    def mass(self, mass):
        self._mass = mass
        # if isinstance(mass, numbers.Real):
        #     self.mass = mass * Mo
        # elif isinstance(mass, Quantity):
        #     self.mass = mass.to(Mo)
        # else:
        #     raise TypeError('Mass should be a real number (in solar mass), or a'
        #                     ' Quantity with mass units.')
        del self.radius

    @lazyproperty
    def radius(self):
        return self.mass_radius(self.mass.value) * u.Rsun

    @radius.setter
    def radius(self, r):
        self.mass = self.mass_radius.inverse(r) * Mo

    # alias
    M = mass
    R = radius

    
    @property
    def origin(self):
        """
        Location of field centre - origin of the magnetic moment in Cartesain
        coordinates.
        """
        return self._origin

    @origin.setter
    @default_units(origin=u.dimensionless_unscaled)
    @u.quantity_input(origin=['length', 'dimensionless'])
    def origin(self, origin):
        origin = self._apply_default_spatial_units(np.asanyarray(origin))
        assert origin.size == 3
        self._origin = origin
    
    # Plotting
    # ------------------------------------------------------------------------ #
    def make_surface(self, res, scale):
        """
        Generate spherical surface with radius scale around centre.

        Parameters
        ----------
        res : [type]
            [description]
        scale : [type]
            [description]

        Examples
        --------
        >>> 

        Returns
        -------
        [type]
            [description]
        """
        r = np.ones(res + 1) * scale * self.radius
        phi = np.linspace(0, 2 * np.pi, res + 1)
        theta = np.linspace(0, np.pi, res + 1)[None].T
        return type(r)(sph2cart(r, theta, phi)) + self.centre.reshape((3, 1, 1))

    def plot2d(self, ax=None, scale=1, **kws):
        # check: position and radius different units

        # WD surface
        cir = Circle(self.centre[:-1],
                     self.radius.value * scale,
                     **{**ARTIST_PROPS_2D.wd, **kws})
        ax.add_patch(cir)
        return cir

    # alias
    plot2D = plot2d

    def plot_wireframe(self, ax=None, res=RESOLUTION, scale=1, **kws):
        ax = ax or self.axes
        return ax.plot_wireframe(*self.make_surface(res, scale), **kws)

    def plot_surface(self, ax=None, res=RESOLUTION, scale=1, **kws):
        ax = ax or self.axes

        return ax.plot_surface(*self.make_surface(res, scale),
                               **{'color': 'c', 'shade': True, **kws})

    # alias
    plot3d = plot_surface

    # def on_first_draw(self, event):
    #     # HACK to get the zorder right at first draw
    #     self.set_zorder()
    #     canvas = self.axes.figure.canvas
    #     canvas.mpl_disconnect(self._cid)
    #     canvas.draw()

    # def get_zorder(self):
    #     return 1. / camera_distance(self.axes, self._panel_centres).ravel()

    # def set_zorder(self):
    #     for i, o in enumerate(self.get_zorder().ravel()):
    #         panel = self.artists[i]
    #         panel._sort_zpos = o
    #         panel.set_zorder(o)

    # def on_rotate(self, event):  # sourcery skip: de-morgan
    #     ax = self.axes
    #     if ((event.inaxes is not ax) or
    #             (ax.button_pressed not in ax._rotate_btn)):
    #         return

    #     # logger.debug('Setting zorder on rotate: {}', ax)
    #     self.set_zorder()


# def bfield(bs)

class MagneticWhiteDwarf(WhiteDwarf):
    """
    A White Dwarf star hosting a magnetic field.
    """

        # \SIrange{7}{230}{\MG} \citep{Ferrario+2015}

        super().__init__(mass, centre)
        #
        if B is None:
            moment = (2 * π * Bs * self.R ** 3 / mu0)  # .to('A m2')
            # Boff * self.R
            
            self.B = MagneticDipole(np.add(self.centre, Boff), moment,
                                    theta=Balt, phi=Baz)

        elif isinstance(B, MagneticField):
            self.B = B
        else:
            raise TypeError('Parameter *B* must be an instance of '
                            '*MagneticField*')

    # def __repr__(self):
    #     return f'{super().__repr__()[:-1]}, '

    def plot3d(self, ax=None, res=RESOLUTION, bfield=(), **kws):
        ax = ax or self.axes
        polygons = self.plot_surface(ax, res, **kws)

        fieldlines = self.B.plot3d(ax, **{**ARTIST_PROPS_3D.bfield,
                                          'rmin': self.R.value,
                                          **dict(bfield)})
        return polygons, fieldlines


# def make_panel(xyz, **kws):
#     verts = np.hstack([xyz, xyz[:, -1:]]).T
#     #verts = np.hstack([xy := np.array(xy), xy[:, -1:]]).T
#     *xy, z = verts.T
#     codes = np.array([1,  2,  2,  2, 79])
#     path = Path(np.array(xy).T, codes)
#     panel = PathPatch(path, **kws)
#     pathpatch_2d_to_3d(panel, z)
#     panel._path2d = path
#     return panel

# def _plot_surface(self, ax=None, res=RESOLUTION, scale=1, **kws):
#     ax = ax or self.axes
#     self._axes = ax

#     # Redo zorder when rotating axes
#     canvas = ax.figure.canvas
#     canvas.mpl_connect('motion_notify_event', self.on_rotate)
#     self._cid = canvas.mpl_connect('draw_event', self.on_first_draw)

#     _xyz = self.make_surface(res, scale)
#     self._panel_centres = (_xyz[:, :-1, :-1] + _xyz[:, 1:, 1:]) / 2
#     polys, colors = make_surface(ax, *_xyz, color='c')

#     for points, c in zip(polys, colors):
#         panel = make_panel(points.T, facecolor=c, edgecolor='w', lw=0.1)
#         ax.add_patch(panel)
#         self.artists.append(panel)

#     ax.auto_scale_xyz(*_xyz, False)

#     return self.artists  # ax.plot_surface(*self.make_surface(res, scale), **kws)


# def make_surface(axes, X, Y, Z, lightsource=None, **kwargs):
#     self = axes

#     if Z.ndim != 2:
#         raise ValueError("Argument Z must be 2-dimensional.")
#     if np.any(np.isnan(Z)):
#         _api.warn_external(
#             "Z contains NaN values. This may result in rendering "
#             "artifacts.")

#     # TODO: Support masked arrays
#     X, Y, Z = np.broadcast_arrays(X, Y, Z)
#     rows, cols = Z.shape

#     has_stride = 'rstride' in kwargs or 'cstride' in kwargs
#     has_count = 'rcount' in kwargs or 'ccount' in kwargs

#     if has_stride and has_count:
#         raise ValueError("Cannot specify both stride and count arguments")

#     rstride = kwargs.pop('rstride', 10)
#     cstride = kwargs.pop('cstride', 10)
#     rcount = kwargs.pop('rcount', 50)
#     ccount = kwargs.pop('ccount', 50)

#     if rcParams['_internal.classic_mode']:
#         # Strides have priority over counts in classic mode.
#         # So, only compute strides from counts
#         # if counts were explicitly given
#         compute_strides = has_count
#     else:
#         # If the strides are provided then it has priority.
#         # Otherwise, compute the strides from the counts.
#         compute_strides = not has_stride

#     if compute_strides:
#         rstride = int(max(np.ceil(rows / rcount), 1))
#         cstride = int(max(np.ceil(cols / ccount), 1))

#     if 'facecolors' in kwargs:
#         fcolors = kwargs.pop('facecolors')
#     else:
#         color = kwargs.pop('color', None)
#         if color is None:
#             color = self._get_lines.get_next_color()
#         color = np.array(mcolors.to_rgba(color))
#         fcolors = None

#     cmap = kwargs.get('cmap', None)
#     shade = kwargs.pop('shade', cmap is None)
#     if shade is None:
#         _api.warn_deprecated(
#             "3.1",
#             message="Passing shade=None to Axes3D.plot_surface() is "
#                     "deprecated since matplotlib 3.1 and will change its "
#                     "semantic or raise an error in matplotlib 3.3. "
#                     "Please use shade=False instead.")

#     colset = []  # the sampled facecolor
#     if (rows - 1) % rstride == 0 and \
#         (cols - 1) % cstride == 0 and \
#             fcolors is None:
#         polys = np.stack(
#             [cbook._array_patch_perimeters(a, rstride, cstride)
#                 for a in (X, Y, Z)],
#             axis=-1)
#     else:
#         # evenly spaced, and including both endpoints
#         row_inds = list(range(0, rows-1, rstride)) + [rows-1]
#         col_inds = list(range(0, cols-1, cstride)) + [cols-1]

#         polys = []
#         for rs, rs_next in zip(row_inds[:-1], row_inds[1:]):
#             for cs, cs_next in zip(col_inds[:-1], col_inds[1:]):
#                 ps = [
#                     # +1 ensures we share edges between polygons
#                     cbook._array_perimeter(a[rs:rs_next+1, cs:cs_next+1])
#                     for a in (X, Y, Z)
#                 ]
#                 # ps = np.stack(ps, axis=-1)
#                 ps = np.array(ps).T
#                 polys.append(ps)

#                 if fcolors is not None:
#                     colset.append(fcolors[rs][cs])

#     # note that the striding causes some polygons to have more coordinates
#     # than others
#     # polyc = art3d.Poly3DCollection(polys,  **kwargs)

#     colset = self._shade_colors(
#         color, self._generate_normals(polys), lightsource)

#     return polys, colset
# relative
