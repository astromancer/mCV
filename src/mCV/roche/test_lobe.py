from mCV.roche.core import RocheLobe
from recipes.transforms import cart2sph

import matplotlib.pyplot as plt
import numpy as np

π = np.pi

lobe = RocheLobe(q=5, primary=True)
naz, nal = 25, 20
α, β = np.mgrid[0:π/2:nal, 0:π:naz]
_, θ, φ = cart2sph(np.cos(β),
                   (sinβ:=np.sin(β)) * np.cos(α),
                   sinβ * np.sin(α))

xyz = lobe.solve_surface(theta=θ, phi=φ)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot_wireframe(*xyz)
#plot_wireframe(α, β, )

plt.show()
