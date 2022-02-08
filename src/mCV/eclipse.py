"""
Constrain the inclination for eclipsing systems.
"""

# Note. The results in Chanan+ 1987 are not reproducible given only
# what is said in the paper.  Beyond that, there are far more efficient ways
# of calculating the allowed inclinations given the eclipse width.  The
# method employed in this module do just that. Read my thesis for details...

# ADS link to the article
# http://adsabs.harvard.edu/abs/1976ApJ...208..512C
# (use this article as a starting point for the construction of the problem.
# don't use their method for a solution, it's a poor way of solving the problem)


# third-party
import numpy as np
from astropy.utils import lazyproperty
from scipy.misc import derivative
from scipy.optimize import brentq, minimize

# relative
from .roche import RocheSolver, _binary_potential_polar2


# from math import pi as π


# TODO: 3D (q, i, φ) surface plots!!!

class InclinationConstraints(RocheSolver):

    # TODO: might want to add some slots since you will make many many of these
    # objects. alternatively create vectorized versions of the functions below.

    def ρ(self, θ):
        # and θ = π : ValueError: f(a) and f(b) must have different signs
        if θ == 0:
            # sub exact L1 point
            # self.psi doesn't behave well at θ = 0, which leads to
            # inaccurate results. Here we sub the exact value of the inner
            # Lagrange point L1
            return self.mu - self.l1.x

        # rmax = self.r2.x - self.l1.x
        interval = (1e-6, self.r2.x - self.l1.x)
        return brentq(_binary_potential_polar2, 
                      *interval, 
                      (self.mu, θ, self.psi0))

    def dρdθ(self, θ, dθ=1e-3):
        # note: default spacing dθ=1e-3 is about twentieth of a degree
        # TODO: might be able to solve using implicit function theorem
        return derivative(self.ρ, θ, dθ)

    def dydx(self, θ, dθ=1e-3):
        r = self.ρ(θ)
        dρ = self.dρdθ(θ, dθ)
        sinθ, cosθ = np.sin(θ), np.cos(θ)
        return (dρ * sinθ + r * cosθ) / (dρ * cosθ - r * sinθ)

    def solve_θ_crit(self, θ):
        r = self.ρ(θ)
        dρ = self.dρdθ(θ)
        return r * r + r * np.cos(θ) + dρ * np.sin(θ)

    @lazyproperty
    def θ_crit(self):
        # critical theta at which the Roche surface normal is perpendicular to
        # the LoS
        return brentq(self.solve_θ_crit, *np.radians((75, 125)))

    @lazyproperty
    def ρ_crit(self):
        # critical theta at which the Roche surface normal is perpendicular to
        # the LoS
        return self.ρ(self.θ_crit)

    def φ(self, i):
        #
        r, θ = self.ρ_crit, self.θ_crit
        ρcosθ = r * np.cos(θ)
        k = (ρcosθ + 1) / np.sqrt(r * r + 2 * ρcosθ + 1) / np.sin(i)
        return np.arccos(k)

    @lazyproperty
    def φmax(self):
        # maximum eclipse half width
        # given mass ratio q, we can calculate θ, which means we can calculate
        # maximal φ. That is, any LoS vector with φ > φmax will not intersect
        # the Roche lobe.
        return self.φ(np.pi / 2)

    def i(self, φ):

        if φ > self.φmax:
            return np.nan

        r, θ = self.ρ_crit, self.θ_crit
        ρcosθ = r * np.cos(θ)
        k = (ρcosθ + 1) / np.sqrt(r * r + 2 * ρcosθ + 1)

        return np.arcsin(k / np.cos(φ))


# def foo1(rp, r, θ, μ):
#     # use to obtain derivative along equipotential curve (NOT WORKING)
#     sinθ = np.sin(θ)
#     rsinθ = r * sinθ
#     cosθ = np.cos(θ)
#     rpcosθ = rp*cosθ
#     return -μ/r/r*rp - 0.5 * (1 - μ) * (2 * r * rp - 2*rpcosθ + 2*rsinθ) / (r*r - 2*r*cosθ + 1)**(3/2) + (μ+1)*(rpcosθ - rsinθ) + r*rp

# def foo2(rp, r, θ, μ):
#     # use to obtain derivative along equipotential curve (NOT WORKING)
#     sinθ = np.sin(θ)
#     rsinθ = r * sinθ
#     cosθ = np.cos(θ)
#     rpcosθ = rp*cosθ
#     return (μ-1)/r/r*rp - μ * (r * rp - rpcosθ + rsinθ) / (r*r - 2*r*cosθ + 1)**(3/2) + μ*(rpcosθ - rsinθ) + r*rp


if __name__ == '__main__':
    # The following section attempts to reproduce the Chanan 1976+ fig 1,
    # given *only what is said in the paper*.  This clearly does not work,
    # as you will see from the resulting figure. Below we use the same
    # symbols in the source code as in the paper for maximal clarity.

    from scrawl.imagine import ImageDisplay


    def 𝜓(r, θ, 𝜙, q):
        """
        Effective potential in the co-rotating frame is then G m 𝜓(r, θ, 𝜙)
        """
        rsinθcos𝜙 = r * np.sin(θ) * np.cos(𝜙)

        return -1 / r - q / np.sqrt(r * r - 2 * rsinθcos𝜙 + 1) - \
               (q + 1) / 2 * ((rsinθcos𝜙 - q / (q + 1)) ** 2 + rsinθcos𝜙 ** 2)


    def _lagrange_solver(l, q):
        return (l ** -2 - l) / ((1 - l) ** -2 - 1 + l) - q


    def solve_l1(q):
        """Solve for the inner Lagrange point given q"""
        return brentq(_lagrange_solver, 0.01, 0.9, q)


    def Ω(q):
        l = solve_l1(q)
        return 𝜓(l, 0.5 * np.pi, 0, q) + 0.5 * q * q / (q + 1)


    # def f1(r, 𝜙, q, i):
    #     u = np.cos(𝜙)
    #     sini = np.sin(i)
    #     rsini = r * sini
    #     ursini = u * rsini
    #     return (r * r - 2 * ursini + 1)**-0.5 - ursini + \
    #            1 / q * (Ω(q) + 1 / r * 0.5 * (q + 1) * rsini ** 2)
    #
    #
    # def f2(r, 𝜙, q, i):
    #     u = np.cos(𝜙)
    #     sini = np.sin(i)
    #     usini = u * sini
    #     ursini = r * usini
    #     return (usini - r) * (r * r - 2 * ursini + 1) ** (-3 / 2) - \
    #            usini + 1 / q * ((q + 1) * r * sini * sini - 1 / r / r)
    #
    def f1(r, u, q, i):
        sini = np.sin(i)
        rsini = r * sini
        ursini = u * rsini
        return (r * r - 2 * ursini + 1) ** -0.5 - ursini + \
               1 / q * (Ω(q) + 1 / r * 0.5 * (q + 1) * rsini ** 2)


    def f2(r, u, q, i):
        sini = np.sin(i)
        usini = u * sini
        ursini = r * usini
        return (usini - r) * (r * r - 2 * ursini + 1) ** (-3 / 2) - \
               usini + 1 / q * ((q + 1) * r * sini * sini - 1 / r / r)


    def objective(p, q, i):
        r, u = p
        return f1(r, u, q, i) ** 2 + f2(r, u, q, i) ** 2


    # test case
    q = 10
    i = np.radians(80)
    r = np.linspace(1e-1, 2, 100)
    u = np.linspace(0, 1, 100)[None].T

    # minimize
    p0 = 0.5, 0.5
    res = minimize(objective, p0, (q, i), bounds=[(1e-3, 2), (0, 1)])

    # visualise the objective as an image
    o = objective(r, u, q, i)
    extent = [r.min(), r.max(), u.min(), u.max()]
    ImageDisplay(o, origin='lower', extent=extent)

    #######################################################
    # The next part makes fig 1 from Chanan 1976+ YAY!!!
    #######################################################

    from matplotlib import pyplot as plt
    from matplotlib.transforms import blended_transform_factory as btf

    fig, ax = plt.subplots(figsize=(10, 8))
    #

    qres = 10
    res = 50
    imin = np.empty(res)
    Q = np.linspace(1e-2, 2, qres)
    Q = [0.01, 0.02, 0.05, 0.01, 0.2, 0.5, 1, 1.5, 2, 3,
         5]  # , 7, 10, 20, 50, 100, 200, 500, 1e3, 1e4]
    Th_crit = np.empty(len(Q))
    for j, q in enumerate(Q):
        ic = InclinationConstraints(q)
        Th_crit[j] = ic.θ_crit
        # print(Table(dict(q=ic.q, ic.φmax=φmax)))

        # linspace not the best choice for hyperbolic relation
        Φ = ic.φmax * np.sin(np.linspace(0, np.pi / 2, res))
        i = np.vectorize(ic.i, 'f')(Φ)

        # plot
        ax.plot(np.degrees(i), np.degrees(Φ), '-')
        # ax.plot(np.sin(i), np.cos(Φ), 'o') # np.degrees(
        label = 'q=%.3g' % q
        ax.text(1, np.degrees(Φ[-1]), label,
                transform=btf(ax.transAxes, ax.transData))

    ax.set(xlabel='$i$', ylabel='$\phi$')
    ax.grid()
    # ax.legend()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


