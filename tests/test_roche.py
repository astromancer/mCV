# third-party
import numpy as np
import matplotlib.pyplot as plt

# local
from mCV.roche import RocheLobes, RocheLobe, BinaryParameters


class TestBinaryParameters:
    def test_init(self):
        BinaryParameters(1)
        
    def check_attrs(self, bp):
        #bp.q, bp.m1, bp.m2
    

def test_solver():
    # test primary solver
    fig, ax = plt.subplots()
    for q in np.linspace(0, 1, 11):
        try:
            roche = RocheLobes(q)
            x1, y1 = roche.primary()
            pri, = ax.plot(x1, y1, label=f'q = {q}')
            print(q)
        except Exception as e:
            print(q, e)
        ax.legend()
        
test_solver()
