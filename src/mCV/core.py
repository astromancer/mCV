

from .wd import MagneticWhiteDwarf, WhiteDwarf
from .roche.core import RocheLobes, semi_major_axis


class CompactBinaryStar:
    def __init__(self, q=None, m1=None, m2=None, a=None, P=None):

        #
        self.roche = RocheLobes(q, m1, m2, a, P)

        



class CataclysmicVariable(CompactBinaryStar):
    def __init__(self, q=None, m1=None, m2=None, a=None, P=None):
        super().__init__(q=q, m1=m1, m2=m2, a=a, P=P)
        
        self.primary = WhiteDwarf(m1)

        # self.stream = None
        # self.secondary = RedDwarf(m2)  # TODO: BrownDwarf etc...


class MagneticCataclysmicVariable(CataclysmicVariable):
    def __init__(self, q=None, m1=None, m2=None, a=None, P=None,
                 Bs=None, Balt=0, Baz=0):
        super().__init__(q=q, m1=m1, m2=m2, a=a, P=P)
        self.primary = MagneticWhiteDwarf(m1, (self.roche.primary.r1, 0, 0),
                                          Bs, Balt, Baz)
        

CV = CataclysmicVariable
MCV = MagneticCataclysmicVariable

# mcv = MagneticCataclysmicVariable(q)
# mcv.secondary.roche_lobe.plot3d()
# mcv.primary.plot3d()                # will plot on the same axis by default
# mcv.stream.plot3d()
# MagnetoBallisticStream / MagnetoHydrodynamicStream

# cv = CataclysmicVariable()
# mcv.label_lagrangians()
