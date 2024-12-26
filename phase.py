import numpy as np
from vortex import Vortex
import phasemap as pm

winding = np.arange(-6, 6)
R = 40
vorticies = [Vortex(n=n, R=R) for n in winding]


def phase(pos):
    J1, J2 = pos
    energy = []
    for i in range(len(winding)):
        E = -J1 * vorticies[i].E_xy - J2 * vorticies[i].E_dd
        energy.append(E)
    n = np.argmin(energy)
    print(J1, J2, n)
    return winding[n]


res = pm.run(
    phase, limits=[(-1, 1), (-0.1, 0.1)], mesh=6, num_steps=6, save_file="results.json"
)
