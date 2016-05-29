import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#****************************************************************************************************
class MagneticDipole():
    #TODO bounding box
    #TODO multicolor shells
    #TODO plot2D
    #TODO tilted dipole!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fieldlines(self, Nshells=5, Naz=10, res=100, bounding_box=None):
        '''Compute the field lines for a dipole magnetic field'''

        Re = 1
        L = np.r_['-1,3,0', range(1, Nshells+1)] #as 3D array

        theta = np.linspace(0, np.pi, res)
        sin_theta = np.sin(theta)
        r = Re * sin_theta ** 2     #radial profile of B-field lines
        r_sin_theta = r * sin_theta

        phi = np.c_[np.linspace(0, 2*np.pi, Naz)]
        
        #convert to Cartesian coordinates
        v = np.r_['-1,3,0',
                  r_sin_theta * np.cos(phi),
                  r_sin_theta * np.sin(phi),
                  r * np.cos(theta) * np.ones_like(phi)]
        v = v.reshape(-1,3)

        return L * v
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot3d(self, ax, **kw):
        #
        segments = self.fieldlines()
        ax.add_collection3d( Line3DCollection(segments, **kw) )
        ax.auto_scale_xyz( *segments.T )


#dipole = MagneticDipole()    