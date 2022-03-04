# third-party
import matplotlib.pyplot as plt
from astropy import units as u

# local
from mCV.bfield import MagneticDipole, PureMultipole
from mCV.bfield import PhysicalMultipole
from mCV.bfield import _legendre_zeros, _legendre_zero_intervals

from scipy.special import  lpmv
from scipy.optimize import brentq

π = np.pi

def test_dipole():
    return MagneticDipole(theta=30, phi=30).plot3d(rmin=0.1, alpha=1)


def test_pure_multipole(degree):
    PureMultipole(0, degree)


# plt.show()
def test_legendre_zeros():

    res=100

    fig, ax = plt.subplots()

    for l in range(8, 12):
        x = np.linspace(0, 1, res)
        line, = ax.plot(x, lpmv(1, l, x), ':',  ) # color=line.get_color()
        thc = _legendre_zeros(l)
        print(f'l = {l}\nzeros = {thc}\n')
        ax.plot(thc, np.zeros_like(thc), 'x',  color=line.get_color())
        
        
def test_legendre_zero_angles():
    res=100

    fig, ax = plt.subplots()

    for l in range(2, 8):
        θ = np.linspace(np.pi / 2, 0, res)
        sinθ, cosθ = np.cos(θ), np.sin(θ)
        #pl0 = lpmv(0, l, cosθ)
        pl1 = lpmv(1, l, cosθ)
        sl = np.sin(θ) * pl1
        line, = ax.plot(θ, sl, ':',  ) # color=line.get_color()
        thc = np.pi / 2 - np.arccos(_legendre_zeros(l))
        
        print(f'l = {l}\nzeros = {thc}\n')
        ax.plot(thc, np.zeros_like(thc), 'x',  color=line.get_color())

    ax.grid()
    ax.legend()
    fig