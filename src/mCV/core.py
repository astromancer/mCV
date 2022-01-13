

# std
import textwrap as txw

# third-party
import numpy as np
import matplotlib.pyplot as plt

# local
from recipes import pprint as pp

# relative
from .wd import MagneticWhiteDwarf, WhiteDwarf
from .roche.core import Axes3DHelper, BinaryParameters, Ro, RocheLobes


class ParameterDescriptor:
    def __init__(self, name):
        self.name = str(name)

    def __get__(self, instance, kls=None):
        if instance is None:
            return self  # lookup from class

        return getattr(instance.roche.u, self.name)

    def __set__(self, instance, value):
        setattr(instance.roche.u, self.name, value)


class ForwardParameters:
    a = ParameterDescriptor('a')
    P = ParameterDescriptor('P')
    m = ParameterDescriptor('m')
    m1 = ParameterDescriptor('m1')
    m2 = ParameterDescriptor('m2')
    q = ParameterDescriptor('q')
    r1 = ParameterDescriptor('r1')
    r2 = ParameterDescriptor('r2')


class CompactBinaryStar(ForwardParameters, Axes3DHelper):
    def __init__(self, q=None, m=None, m1=None, m2=None, a=None, P=None):
        self.roche = RocheLobes(q, m, m1, m2, a, P)

    def _label_axes(self, ax):
        return self.roche._label_axes(ax)


class CataclysmicVariable(CompactBinaryStar):
    Primary = WhiteDwarf

    def __init__(self, q=None,  m=None,  m1=None, m2=None, a=None, P=None, **kws):
        super().__init__(q, m, m1, m2, a, P)

        # create white dwarf object
        self.primary = self.wd = self.Primary(m1, self.r1.xyz, **kws)

        if (u := getattr(self.roche.u.a, 'unit')):
            self.wd.radius = self.wd.radius.to(u)
            self.wd.centre = self.wd.centre.to(u)
            
        # self.stream = None
        # self.secondary = RedDwarf(m2)  # TODO: BrownDwarf etc...

    @property
    def a(self):
        return CompactBinaryStar.a.__get__(self)

    @a.setter
    def a(self, value):
        CompactBinaryStar.a.__set__(self, value)
        u = getattr(self.roche.u.a, 'unit')
        if u is None:
            return
        
        self.wd.radius = self.wd.radius.to(u)
        self.wd.centre = self.wd.centre.to(u)
        
        if isinstance(self.wd, MagneticWhiteDwarf):
            self.wd.B.origin = self.wd.B.origin.to(u)

    def pformat(self):
        return '\n'.join([
            self.roche.u.pformat(),
            self.wd.pformat(),
            f"{' '* 22} = {(self.wd.R / self.a).si:.5f} a"
        ])

    def pprint(self):
        """Pretty print binary parameters"""
        print(self.pformat())

    def plot2d(self, ax=None):
        # plot both lobes on same axes
        if ax is None:
            _, ax = plt.subplots()

        # Roche Lobes
        _, art = self.roche.plot2d(ax)
        texts = self.roche.label_lagrangians(ax)

        # WD
        self.wd.plot2d(ax)

        return ax, (*art, texts)

    def plot3d(self, ax=None, wd=(), roche=()):
        ax = ax or self.roche.axes
        polygons, *bfieldlines = self.wd.plot3d(ax, **dict(wd))
        ax, wf = self.roche[1].plot3d(ax, **dict(roche))

        aspect = np.ptp([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()], 1)
        aspect /= aspect[0]
        aspect[1:] = aspect[1:].max()
        ax.set_box_aspect(aspect)

        return ax, (polygons, *bfieldlines, wf)


class MagneticCataclysmicVariable(CataclysmicVariable):

    Primary = MagneticWhiteDwarf

    def __init__(self, q=None, m=None, m1=None, m2=None, a=None, P=None,
                 Bs=None, Balt=0, Baz=0, Boff=(0, 0, 0)):
        super().__init__(q, m, m1, m2, a, P,
                         Bs=Bs, Balt=Balt, Baz=Baz, Boff=Boff)
        if (u := getattr(self.a, 'unit')):
            self.wd.B.origin = self.wd.centre.to(u)

    def plot2d(self, ax=None):
        # plot both lobes on same axes
        if ax is None:
            _, ax = plt.subplots()

        #
        ax, art = super().plot2d()
        # B field
        # lc = self.wd.B.plot3d(ax, scale=1e-6)

        return ax, (*art, None)

    # def plot3d(self, ax=None):
    #     ax = ax or self.roche.axes
    #     surf = self.wd.plot3d(ax)
    #     wf = self.roche[1].plot3d(ax)

    #     aspect = np.ptp([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()], 1)
    #     aspect /= aspect[0]
    #     aspect[1:] = aspect[1:].max()
    #     ax.set_box_aspect(aspect)
    #     return ax, (surf, wf)


CV = CataclysmicVariable
MCV = MagneticCataclysmicVariable

# mcv = MagneticCataclysmicVariable(q)
# mcv.secondary.roche_lobe.plot3d()
# mcv.primary.plot3d()                # will plot on the same axis by default
# mcv.stream.plot3d()
# MagnetoBallisticStream / MagnetoHydrodynamicStream

# cv = CataclysmicVariable()
# mcv.label_lagrangians()
