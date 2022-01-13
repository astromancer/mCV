# third-party
import matplotlib.pyplot as plt
from astropy import units as u

# local
from mCV.bfield import MagneticDipole


b = MagneticDipole(theta=30, phi=30)
art = b.plot3d(rmin=0.1, alpha=1)


plt.show()
