
# std
from collections import defaultdict

# third-party
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from astropy import units as u
from astropy.utils import lazyproperty

# local
from recipes.array import fold

# relative
from .utils import get_unit_string
from .origin import Origin


def set_viewlim_equal(ax):
    vl = np.array([getattr(ax, f'get_{xyz}lim')() for xyz in 'xyz'])
    span = np.ptp(vl, axis=1).max()
    lims = np.mean(vl, axis=1, keepdims=1) + span * np.array([[-1, 1]]) / 2
    ax.set(**{f'{xyz}lim': l for xyz, l in zip('xyz', lims)})


def _get_axis_label(name, val, default_unit='', brackets='[]'):
    left, right = r'\left{} \right{}'.format(*brackets).split()
    yield '$'
    yield str(name)
    if u := get_unit_string(val, default_unit):
        yield fr'\ {left}{u}{right}'
    yield '$'


def get_axis_label(name, val, default_unit='', brackets='[]'):
    return ''.join(_get_axis_label(name, val, default_unit, brackets))


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


class Axes3DHelper:
    """Launch a figure with 3D axes on demand."""

    _cid = None

    _subplot_kws = {'subplot_kw': dict(projection='3d')}

    @lazyproperty
    def axes(self):
        return self.get_axes()

    def get_axes(self):
        """Create the figure and  axes"""
        fig, ax = plt.subplots(**self._subplot_kws)
        self._label_axes(ax)
        self._cid = fig.canvas.mpl_connect('draw_event', self._on_first_draw)
        return ax

    def _on_first_draw(self, _event):
        # have to draw rectangle inset lines after first draw else they point
        # to wrong locations on the edges of the lower axes
        fix_axes_offset_text(self.axes)

        # disconnect so this only runs once
        self.axes.figure.canvas.mpl_disconnect(self._cid)

    def _label_axes(self, ax, units=(), **kws):
        raise NotImplementedError


class SpatialAxes3D(Axes3DHelper):

    _subplot_kws = dict(
        subplot_kw=dict(projection='3d'),
        gridspec_kw=dict(top=0.95,
                         left=0.05,
                         right=0.95,
                         bottom=-0.05)
    )

    def get_axes(self):
        ax = super().get_axes()
        ax.set_box_aspect((1, 1, 1))
        return ax

    def _label_axes(self, ax, units=(), default_unit='', **kws):
        if not isinstance(units, dict):
            units = dict(units)

        kws = {**dict(usetex=True, labelpad=10), **kws}
        for xyz in 'xyz':
            label = get_axis_label(xyz, units.get(xyz), default_unit)
            getattr(ax, f'set_{xyz}label')(label, **kws)

    def _on_first_draw(self, _event):
        super()._on_first_draw(_event)
        set_viewlim_equal(self.axes)
        # print('PING!')


class OriginInAxes(Origin, SpatialAxes3D):
    """"""

    def _label_axes(self, ax, units=(), **kws):
        # units = defaultdict(lambda: get_unit_string(self.origin))
        return super()._label_axes(ax, units, get_unit_string(self.origin), **kws)
