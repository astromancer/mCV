# third-party
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# local
from mCV.bfield import dipole_fieldline
from recipes.transforms import sph2cart

# binary parameters
# cv = CV(m1=0.68 * Mo,     # primary mass (solar masses)
#         m2=0.16 * Mo,     # secondary mass
#         P=2.08 * u.hour)  # Orbital period
# cv.pprint()


def dipole_fieldline(θ, φ):
    r = (3 * np.cos(θ) ** 2 + 1) ** (1/6)  # * Re
    return sph2cart(r, θ, φ)
    
#     return np.array([rsinθ * np.cos(φ),
#                      rsinθ * np.sin(φ),
#                      r * np.cos(θ) * np.ones_like(φ)])

nshells = 2
res=100
naz = 5
scale=1
rmin=0.1
π = np.pi


# Fieldlines terminate on star surface.
nsections = 1
fieldlines = np.empty((nshells, naz, res, 3))
φ = np.linspace(0, 2 * π, naz, endpoint=False)[None].T
for i in range(1, nshells + 1):
    theta0 = np.arcsin(np.sqrt(rmin / i))
    θ = np.linspace(theta0, π - theta0, res)
    fieldlines[i-1] = np.moveaxis(i * dipole_fieldline(θ, φ), 0, -1)

fieldlines = scale * fieldlines.reshape((-1, res, 3))
#fieldlines += cv.wd.B.origin.T.value
    
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
kws = dict(color='m', alpha=0.5, linewidth=1)
art = Line3DCollection(fieldlines, **kws)
# np.rollaxis(dipole_fieldline(θ, 0), 0, 3)
ax.add_collection3d(art)
ax.auto_scale_xyz(*fieldlines.T)
#ax.set(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1), box_aspect=[1, 1, 1])
