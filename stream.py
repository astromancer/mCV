import numpy as np
from scipy.integrate import odeint
from scipy.stats import multivariate_normal

from .roche import RochePotential


class BallisticStream(RochePotential):
    def _trajectory_ode(self, t, U, mu):
        """ballistic particle trajectory ODE"""
        # positions # velocities
        x, y, z, vx, vy, vz = U

        y2 = y * y
        z2 = z * z
        yz2 = y2 + z2

        # mu = self.mu

        f1 = mu / ((x + 1 - mu) ** 2 + yz2) ** 1.5
        f2 = (1 - mu) / ((x - mu) ** 2 + yz2) ** 1.5

        ax = 2 * vy + x - f1 * (x + 1 - mu) - f2 * (x - mu)
        ay = 2 * vx + y - f1 * y - f2 * y
        az = (f1 + f2) * z

        return (vx, vy, vz,  # velocities
                ax, ay, az)  # acceleration

    def surface_impact(self, U, R):

        x, y, z, vx, vy, vz = U
        r = np.sqrt(np.square([x - self.mu + 1, y, z]).sum())
        return r < R

    # def trajectory(self, U, t, R):
    #     #
    #     if self.surface_impact(U, R):
    #         return np.zeros(6)  # HACK to mimic surface impact
    #     else:
    #         return self._trajectory_ode(U, t, self.mu)

    # TODO: Magnetic trajectory
    #     - average plasma charge
    #     - magnetic field distributions
    # boundary conditions at L1
    # def nozzle():

    def l1_spray(self, npart=100, nt=101, tmax=0.75, rv=None):
        """
        particle spray into cone at L1

        Parameters
        ----------
        npart:
            number of particles / trajectories
        nt:
            number of time steps
        tmax:
            final time step value for integration
        rv:
            random variable `scipy.stats.rv_continuous` that will be used to
            generate initial particle states


        Returns
        -------

        """
        # Choose initial velocity
        t = np.linspace(0, tmax, nt)

        # initial position
        x0 = np.tile([self.l1.x, 0, 0], (npart, 1))
        # velocity distribution
        if rv is None:
            μv = (-0.5, 0, 0)  # mean velocity
            σv = np.eye(3) * (1e-3, 1e-4, 1e-5)  # covariance matrix
            rv = multivariate_normal(μv, σv)
            # TODO this should be Maxwell-Boltzmann

        v0 = rv.rvs(npart)
        u0 = np.hstack([x0, v0])

        U = np.empty((npart, nt, 6))
        for i, u00 in enumerate(u0):
            U[i] = odeint(self._trajectory_ode, u00, t, args=(self.mu,),
                          tfirst=True)

        # position solutions
        return U[..., :3].T
        # x, y, z = xyz = U[..., :3].T


class FakeMagnetoBallisticStream(BallisticStream):
    def trajectory(self, npart=100, nt=101, tmax=0.75, rv=None, B_thresh = 1e3):

        from recipes.transformations.rotation import sph2cart

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
