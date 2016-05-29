import numpy as np
from scipy.optimize import brentq  #root finding
#from scipy.optimize import fmin #Nelder-Mead simplex algorithm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IPython import embed
##############################################################################################################

############################################   SECONDARY   ##############################################################
#****************************************************************************************************
class Binary():
    pass

#****************************************************************************************************
class Roche():
    #TODO: Coordinate options
    #TODO: grid options
    #root_finder = brentq       #causes RuntimeError: Unable to parse arguments (in scipy/optimize/zeros.py)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, q):
        self.q = q
        self.r1 = q / (q + 1.)           #position of the primary
        self.r2 = 1.0 / (q + 1.)         #position of the secondary
        self.l1 = self.L1()
        
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def L1(self, xtol=1.e-9):
        '''
        Determines the position of the inner Lagrange point L1
        Returns x1 in units of a (semi-major axis)
        '''
        q = self.q
        interval = -self.r1 + xtol, self.r2 - xtol
        
        return brentq(self.dpsidx, *interval, xtol=xtol)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def Psi(self, grid):
        '''
        graviatational potential of binary system in rotating coordiante frame (Cartesian)
        Takes xyz grid in units of a = r1 + r2 the semi-major axis
        '''
        x, y, z = grid
        
        q = self.q
        r1, r2 = self.r1, self.r2

        ysq = y * y
        zsq = z * z
        yzsq = ysq + zsq        #for efficiency, only calculate these once

        f = 2.0*r1 / np.sqrt((x-r2)**2.0 + yzsq) \
            + 2.0*r2 / np.sqrt((x+r1)**2.0 + yzsq) \
                + x*x + ysq
        #k = -G*(M1+M2)/(2*(r1+r2))
        k = -1
        return k * f
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def psi(self, x):
        '''graviatational potential of binary system in rotating coordiante frame 
        along the line joining the two stars
        '''
        return self.Psi((x, 0, 0))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def dpsidx(self, r):
        '''
        differential of psi(x). i.e. gravitational force magnitude
        '''
        r = np.atleast_1d(r)

        r1 = self.r1
        r2 = self.r2

        f1 = lambda r: 2.0 * r2 * (r + r1)** -2.0 - 2.0 * r1 * (r - r2)** -2.0 - 2.0 * r
        f2 = lambda r: 2.0 * r2 * (r + r1)** -2.0 + 2.0 * r1 * (r - r2)** -2.0 - 2.0 * r

        dpdx = np.piecewise(r, [r < r2, r > r2], [f1, f2])

        return dpdx
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _2d(self, X, xtol=1.e-9, ytol=1.e-9, **kw):
        #TODO: log exceptions??
        #TODO: log / suppress warnings??
        '''
        Calculates the equipotential line in the orbital plane.
        '''
        exval                   = kw.get('exception_val')
        break_at_exception      = kw.get('break_at_exception')  #return when bork
        
        X = np.atleast_1d(X)
        Y = np.ma.empty_like(X)         #non-solutions are masked
        Y.mask = False                  #broadcasts the mask to the size of the array
        
        #l1 = self.l1
        #print( 'l1 = ', self.l1 )
        #print( 'psi0', self.psi(self.l1) )

        
        interval = 0, self.r2  #LIMITS???
        psi0 = self.psi(self.l1)

        for i,x in enumerate(X):
            func = lambda y: self.Psi((x, y, 0)) - psi0
            try:
                y, r = brentq(func, *interval, xtol=ytol, full_output=True)
                if r.converged:
                    Y[i] = y
                else:
                    Y.mask[i] = True
                    
            except Exception as err:
                #print( 'Error in Roche lobe calculation:', err )
                if exval is None:
                    if break_at_exception:
                        Y.mask[i:] = True
                        return Y
                    else:
                        Y.mask[i] = True
                else:
                    Y[i] = exval

        return Y
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _2d_unstable(self, X, xtol=1.e-9, ytol=1.e-9, **kw):
        '''
        Calculates the equipotential surface in the orbital plane (Roche lobe)
        '''
        X = np.atleast_1d(X)

        l1 = self.l1(xtol)

        sx = []
        interval = 0, 2.0*self.r2  #LIMITS???
        psi0 = self.psi(l1)
        
        def solver(x):
            fun = lambda y: self.Psi((x, y, 0)) - psi0
            return brentq(fun, *interval, xtol=ytol)
        
        return np.vectorize(solver)(X)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def rmax(self):
        '''
        Calculates the maximal limits of the Roche lobe along the line of centers
        '''

        f = lambda x: self._2d(x, 1.e-9, 1.e-9, exception_val=-1)[0]       #TOTAL HACK!
        interval = self.r2, 2 * self.r2 - self.l1
        #print( [f(i) for i in interval] )
        #embed()
        return brentq(f, *interval, xtol=1e-9)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def make_segments(self, res=(30,20), scale=1.):
        ''' '''
        res = np.atleast_1d(res)
        if len(res)==1:
            res = np.ravel([res,res])
        resr, resaz = res
        
        x = np.linspace(self.l1, self.rmax(), resr)
        y = self._2d(x)
        z = np.zeros_like(x)
        
        x *= scale      #scale * (x - self.r2) =     #center and scale
        y *= scale
        
        #NOTE, you can probably avoid doing the transposes and speed this up by using np.tensordot
        radians = np.linspace(0, 2*np.pi, resaz)
        Y, Z = rota(radians).dot( np.c_[y,z].T ).transpose(1,0,2)
        return x, Y, Z
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_wireframe(self, ax, res=(30,20), scale=1., **kw):
        '''
        res : {int, tuple}
                resolution in r and/or theta
        '''
        return ax.plot_wireframe( *self.make_segments(res, scale), **kw )
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_surface(self, res):
        '''
        res : {int, tuple}
                resolution in r and/or theta
        '''
        raise NotImplementedError


def rota(theta):
    '''rotation helper '''
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin],
                     [sin,  cos]]).transpose(2,0,1)
        
#convenience function
#====================================================================================================
def L1(q, xtol=1.e-9):
    return Roche(q).L1(xtol)


#====================================================================================================
def Roche_limits(coo_match=False):
    #Calculates the x,y maximal limits of the Roche lobe
    global xwpDm

    sx = Roche_2D(linspace(0.85 * r2, 1.2 * r2, rres2 / 2))
    ymax = max(sx)

    f = lambda x: Roche_2D(x, 1.e-9, 1.e-9, exception_val=-1)[0]
    xmax = brentq(f, r2, 2 * r2 - l1, xtol=1e-9)

    if coo_match:
        #shift = (1.-r2)*a/R1
        xmax = (xmax - r2 + 1.) * a / R1
        ymax = ymax * a / R1
    #else:
    #shift = 0.
    #xDmax = xmax
    f = lambda x: x / Roche_3D(x, 0., coo_match=('shift', 'scale'))[0]
    #print a/R1
    #print l1
    #print f(a/R1)[0]
    xwpDm, fopt, _, _, _ = fmin(f, a / R1, xtol=1e-6, full_output=1, disp=0)
    phiDONOR = 2.0 * arctan(1. / fopt)
    #print phiDONOR

    return xmax, ymax, phiDONOR
###


#====================================================================================================
def Roche_2D(X, xtol=1.e-9, ytol=1.e-9, **kw):
    #Calculates the equipotential surface (Roche lobe)
    X = num2arr(X)

    #l1 = L1(q,xtol)
    #print 'l1 = ', l1
    #print psi(l1)

    sx = []
    y_low = 0
    y_hi = 2.0 * r2  #LIMITS???

    for x in X:
        fun = lambda y: Psi(x, y, 0) - psi(l1)
        try:
            y, r = brentq(fun, y_low, y_hi, xtol=ytol, full_output=True)
            if r.converged:
                sx.append(y)
                #print y
            else:
                #print 'Non-convergence!'
                pass
        except:
            #print 'Function error in Roche lobe calculation'
            if 'exception_val' in kw:
                if kw['exception_val'] != None:
                    sx.append(kw['exception_val'])

    return array(sx)
###


#====================================================================================================
def Roche_2D_extended(X, xtol=1.e-9, ytol=1.e-9, **kw):
    X = num2arr(X)

    condition = X >= l1

    SX = where(condition, Roche_2D(X), -Roche_2D(X))

    #print SX
    #print type(SX)
    #print len(SX)
    if len(SX) == 0:
        SX = 0

        if 'return_type' in kw:
            if kw['return_type'] == 'f':
                SX = float(SX)

    return SX



#====================================================================================================
def Roche_3D(Xin, Yin, t=0, xtol=1.e-9, ytol=1.e-9, **kw):
    #Does what Roche_half() does, but using an input XY grid.
    #WARNING:  This function won't work if input grid does not overlap with Roche lobe
        
    Xin = num2arr(Xin)
    Yin = num2arr(Yin)

    flag = 0
    if 'coo_match' in kw:  #ELSE???????????????????????
        if kw['coo_match'] == 'all':
            kw['coo_match'] = ('scale', 'phase', 'shift')

        if 'scale' in kw['coo_match']:
            scale = a / R1
            flag = 1
        else:
            scale = 1.

        if 'phase' in kw['coo_match'] or t != 0:
            phase = ORBphase(t)
        else:
            phase = 0.

        if 'shift' in kw['coo_match']:
            if flag:
                shift = (1. - r2)
            else:
                shift = (1. - r2) * scale
        else:
            shift = 0.

        Xin = Xin / scale
        Yin = Yin / scale

        x0, y0 = pol2cart(shift, phase)
        X, Y = Xin - x0, Yin - y0  #Takes the input grid to coordinate center

        if len(X.shape) == 1:
            X = X[None]  #THIS IS AN UGLY HACK!
            Y = Y[None]
        R, PHI = cart2pol(X, Y, 'grid')
        PHI = PHI - phase  #De-rotates the Roche lobe
        Xin, Yin = pol2cart(
            R, PHI)  #WARNING!  THIS ALGORITHM HANDLES THE EDGES POORLY....

    Z = []
    for j in range(len(Xin)):  #REFINE!!!
        if 'extended' in kw:
            if kw['extended']:
                sx = Roche_2D_extended(Xin[j], xtol, ytol, **kw)
                #raw_input(sx)
        else:
            sx = Roche_2D(Xin[j], xtol, ytol, **kw)
        z = sqrt(sx * sx - Yin[j] * Yin[j])
        Z.append(list(z))

    Z = array(Z) * scale
    if 'exception_val' in kw:
        Z[isnan(Z)] = kw['exception_val']

    if 'return_type' in kw:
        if kw['return_type'] == 'f':
            #print Z
            Z = Z[0, 0]

    return Z
###


#====================================================================================================
def Roche_half(xtol=1.e-9, ytol=1.e-9, **kw):

    xDmax, _, _ = Roche_limits()
    X = linspace(l1, xDmax, rres2)
    sx = Roche_2D(X, xtol, ytol)

    Y = []
    Z = []

    phispace = linspace(-pi / 2, pi / 2, phires2
                        )  #Used to divide interval for y --> Aesthetics
    for sxi in sx:
        _, y = pol2cart(sxi, phispace)
        z = sqrt(sxi * sxi - y * y)
        Y.append(y)
        Z.append(z)

    X = X[None] * ones((phires2, 1))
    Y = array(Y).T
    Z = array(Z).T

    if 'coo_scale_match' in kw:
        if kw['coo_scale_match'] == True:
            #print 'a/R1 = ', a/R1
            c = a / R1
            X = X * c
            Y = Y * c
            Z = Z * c

    if 'pr' in kw:
        if kw['pr'] == True:
            #print 'For M1 = %s Mo and M2 = %s Mo \nr1 = %s and r2 = %s\n' %(M1s,M2s,r1,r2)
            print('L1: x = %s' % l1)
            print('psi(x1) = %s' % psi(l1))

    return X, Y, Z
###


#WARNING!!!  CURRENTLY THIS FUNCTION PRODUCES DOUBLED (OVERLAPPING) LINES IN ORBITAL PLANE--->  MAY YIELD INACCURATE FLUXES
#====================================================================================================
def Roche_full(xtol=1.e-9, ytol=1.e-9, **kw):

    if 'coo_scale_match' in kw:
        X, Y, Z = Roche_half(coo_scale_match=kw['coo_scale_match'])
    else:
        X, Y, Z = Roche_half()

    XX = concatenate((X, X))
    YY = concatenate((Y, Y[::-1, ...]))
    ZZ = concatenate((Z, -Z))

    return XX, YY, ZZ
###


#====================================================================================================
def dRoche_dx(X, Y, **kw):
    if 'coo_match' not in kw:
        kw['coo_match'] = 'all'  #default to match coordinates
    deriv = zeros(X.shape)
    for u in range(X.shape[0]):
        for v in range(X.shape[1]):
            y = Y[u, v]
            #print Roche_3D(X[u,v],y,t,coo_match=kw['coo_match'],return_type='f')
            dRoche_dx = lambda x: Roche_3D( x, y,
                                            coo_match=kw['coo_match'],
                                            return_type='f',
                                            exception_val=nan,
                                            extended=1)
            dR_dx = derivative(
                dRoche_dx,
                X[u, v],
                dx=abs(X[u, -1] - X[u, 0]) / (20.0 * len(X[u, :]))
            )  #NEED TO CHOOSE THE RESOLUTION IN A BETTER WAY...
            deriv[u, v] = dR_dx

    return deriv
###


#====================================================================================================
def dRoche_dy(X, Y, **kw):
    if 'coo_match' not in kw:
        kw['coo_match'] = 'all'  #default to match coordinates
    deriv = zeros(X.shape)
    for u in range(X.shape[0]):
        for v in range(X.shape[1]):
            x = X[u, v]
            #raw_input(all(Roche_3D(x,Y[u,v],t,coo_match=kw['coo_match'],return_type='f',exception_val=nan,extended=1)==Roche_3D(x,Y[u,v],t,coo_match=kw['coo_match'],return_type='f',exception_val=nan)))
            dRoche_dy = lambda y: Roche_3D(x,y,coo_match=kw['coo_match'],return_type='f',exception_val=nan,extended=1)
            dR_dy = derivative(
                dRoche_dy,
                Y[u, v],
                dx=abs(Y[u, -1] - Y[u, 0]) / (20.0 * len(Y[u, :]))
            )  #NEED TO CHOOSE THE RESOLUTION IN A BETTER WAY...
            deriv[u, v] = dR_dy

    #raw_input(where(Y==0))
    #deriv[isnan(deriv)] = 0                              #EDGES?????????????????
    return deriv
