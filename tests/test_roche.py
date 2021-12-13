# third-party
import numpy as np
import matplotlib.pyplot as plt

# local
from mCV import RocheLobes

# test primary solver
fig, ax = plt.subplots()
for q in np.linspace(0, 1, 11):
    try:
        roche = RocheLobes(q)
        x1, y1 = roche.primary()
        pri, = ax.plot(x1, y1, label='q = %.1f' % q)
        print(q)
    except Exception as e:
        print(q, e)
    ax.legend()
