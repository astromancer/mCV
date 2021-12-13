
# std
from recipes.transforms import pol2cart, sph2cart
from recipes.transforms.rotation import rotation_matrix
import numbers

# third-party
import numpy as np
from astropy.constants import k_B, u as atomic_mass
from astropy.units import Kelvin, meter
from scipy.stats import maxwell
from scipy.integrate import odeint, solve_ivp

# relative
from .roche import RocheSolver
from scipy.interpolate import interp1d


SOLAR_MEAN_MOLECULAR_MASS = 0.53


def norm(xyz):
    return np.sqrt(np.square(xyz).sum())


def maxwell_boltzmann(T, m, lengthscale=1):

    # Mean molecular weight
    if isinstance(m, numbers.Real):
        m *= atomic_mass
    # Temperature
    if isinstance(T, numbers.Real):
        T *= Kelvin
    if isinstance(lengthscale, numbers.Real):
        T *= meter

    # Maxwell-Boltzmann velocity distribution
    # velocity scale value in units of a per sec
    vscale = (np.sqrt(k_B * T.si / m.si) / lengthscale.si).to('/s')
    return maxwell(0, vscale.value)


class BallisticStream(RocheSolver):
    def __init__(self, q):
        super().__init__(q)
        # self.scale = a

        theta = np.linspace(0, np.pi, 100)
        r, *_ = self._solve_radius(theta)
        self.interpr = interp1d(theta, r)

    def _trajectory_ode(self, t, u, mu):
        """
        Ballistic particle trajectory ODE. Computes 3-velocities and
        accelerations from position velocity state.
        """
        # print(u.shape)

        # positions # velocities
        x, y, z, vx, vy, vz = u
        y2 = y * y
        z2 = z * z
        yz2 = y2 + z2

        #
        xmu = x - mu
        f1 = mu / ((xmu + 1) ** 2 + yz2) ** 1.5
        f2 = (1 - mu) / (xmu ** 2 + yz2) ** 1.5

        # acceleration vector
        ax = 2 * vy + x - f1 * (xmu + 1) - f2 * xmu
        ay = 2 * vx + y - f1 * y - f2 * y
        az = (f1 + f2) * z

        return (vx, vy, vz,  # velocities
                ax, ay, az)  # acceleration

    # TODO: Magnetic trajectory
    #     - average plasma charge
    #     - magnetic field distributions
    # boundary conditions at L1
    # def nozzle():

    def l1_spray(self, t, a=1, T=4000, m=SOLAR_MEAN_MOLECULAR_MASS, n=100):
        """
        particle spray into cone at L1

        Parameters
        ----------
        t : Sequence
            time steps
        a : float
            semi-major axis of orbit
        T : float
            Temperature (K)
        m : float
            Mean molecular weight of particles.
        n : int
            number of particles / trajectories


        Returns
        -------

        """

        # t = np.linspace(0, tmax, nt)

        # initial position
        x0 = np.tile([self.l1.x, 0, 0], (n, 1))

        Pv = maxwell_boltzmann(T, m, a)
        v0 = Pv.rvs((n, 3))         # initial speeds
        u0 = np.hstack([x0, v0])

        return self.solve_trajectory(t, u0)

    def lobe_overflow_ic(self, a=1, T=4000, m=SOLAR_MEAN_MOLECULAR_MASS, n=100,
                         theta_max=np.radians(5), bloat=0.01):
        # initial positions uniformly distributed on Roche surface cone.
        # theta is the opening angle of a point on the surface of the donor from
        # the line of centers (LoS) line connecting L1 with donor centre and
        # CoM. phi is the azimuthal angle from the LoS
        theta0 = np.abs(np.random.randn(n) * theta_max / 5)
        phi0 = np.random.uniform(0, 2 * np.pi, n)
        r0 = self.interpr(theta0)
        x0, z0 = pol2cart(r0, theta0)
        x0 += (self.mu - 1) + bloat
        # random surface positions for ivp
        y0, z0 = np.einsum('ijk,ik->jk',
                           rotation_matrix(phi0),
                           [np.zeros_like(x0), z0])

        # initial particle speeds
        v0 = maxwell_boltzmann(T, m, a).rvs((n, 3))
        return np.column_stack([x0, y0, z0, v0])

    def lobe_overflow(self, t, a=1, T=4000, m=SOLAR_MEAN_MOLECULAR_MASS, n=100,
                      theta_max=np.radians(5), bloat=0.01):
        """
        Simulate initial conditions for Roche lobe overflow on a Main Sequence 
        secondary star with temperature *T*, mean molecular plasma mass, *m*
        with fractional bloating of *bloat*. Particle trajectories for *n*
        are solved for in the binary gravitational potential. Initial positions
        for particles are at random locations on the bloated donor surface
        within a maximal opening angle *theta_max* from the donor center.
        Initial velocities are taken from the Maxwell-Boltzmann distribution at
        the given temperature and mean molecular mass.
        
         

        Parameters
        ----------
        t : Sequence
            Time steps for ode integration.
        T : float, optional
            Donor temperature of Maxwell-Boltzmann velocity distribution for
            random initial conditions, by default 4000
        m : float, optional
            Mean molecular mass of accreting material, by default SOLAR_MEAN_MOLECULAR_MASS
        n : int, optional
            Number of particles, by default 100
        theta_max : [type], optional
            Determines the thickness of the stream.
            Maximum opening angle for theta, by default np.radians(15).
            theta is the opening angle of a point of the surface from the line
            connecting L1 with donor centre.

        Returns
        -------
        xyz: np.ndarray
            Particle trajectories in corotating center-of-mass frame.
        
        Examples
        --------
        >>> 
        """

        u0 = self.lobe_overflow_ic(a, T, m, n, theta_max, bloat)
        return self.solve_trajectory(t, u0)

    def solve_trajectory(self, t, u0):
        traject = np.full((len(u0), len(t), 6), np.nan)
        t_span = t[[0, -1]]

        for i, u00 in enumerate(u0):
            result = solve_ivp(self._trajectory_ode, t_span, u00,
                               args=(self.mu,), t_eval=t, vectorized=True,
                               events=[self.impact_donor, self.ejected])
            data = result.y.T
            traject[i, :len(data)] = data

        # position solutions
        return traject[..., :3].T

    # Events (ODE)
    # ------------------------------------------------------------------------ #

    # def impact_primary(self, t, u):

    #     xyz = u[:3] - (self.mu, 0, 0)
    #     r = np.sqrt(np.square(xyz).sum())
    #     return r <

    def impact_donor(self, t, u, *args):

        xyz = u[:3] - (self.mu - 1, 0, 0)
        r = np.linalg.norm(xyz)
        θ = np.arccos(xyz[0] / r)
        if r < self.interpr(θ):
            return 1
        return -1

    impact_donor.terminal = True

    def ejected(t, u. *args):
        return [-1, 1][norm(u[:3]) > 2]

    ejected.terminal = True
    
    # ------------------------------------------------------------------------ #
    
# xyz = U[:3] - (self.mu - 1, 0, 0)
# class MagnetoBallisticStream(BallisticStream):
#     def trajectory(self, npart=100, nt=101, tmax=0.75, rv=None, B_thresh = 1e3):


class FakeMagnetoBallisticStream(BallisticStream):
    def trajectory(self, npart=100, nt=101, tmax=0.75, rv=None, B_thresh=1e3):

        from recipes.transforms import sph2cart

        xyz = self.l1_spray(npart, nt, tmax, rv)

        # distance from primary
        cen = np.array([self.r1, 0, 0], ndmin=3).T
        xyz_c = xyz - cen
        xyz_u = xyz_c / np.sqrt(np.square(xyz_c).sum(0))
        zena = np.arccos(xyz_u[-1])  # zenith angle
        r_c = np.sqrt(np.square(xyz_c).sum(0))
        # magnetic field strength
        Br = np.sqrt(3 * np.square(np.cos(zena)) + 1) / np.power(r_c, 3)
        # Simulate magnetic accretion at this field threshold

        for j in range(npart):
            l = Br[:, j] > B_thresh
            if l.any():
                i = np.where(l)[0][0]
                th = zena[i, j]
                Re = r_c[i, j] / np.square(np.sin(th))

                # artificially choose pole #[0, np.pi][th < np.pi / 2]
                thb = np.linspace(th, 0, nt - i)
                rb = Re * np.square(np.sin(thb))
                phi = np.arctan2(xyz_c[1, i, j], xyz_c[0, i, j])
                xyz[:, i:, j] = sph2cart(rb, thb, phi) + cen[..., 0]

        return xyz


# TODO: Read Kriz 1970
# TODO: read Canelle 2005
