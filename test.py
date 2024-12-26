import numpy as np
import matplotlib.pyplot as plt
from vortex import Vortex


##test for energy density

# vortex = Vortex(n=-1, R=20)
# vortex.plot()
# vortex.plot_energy_density()



##test for behaviour of energies with radius


radii = np.arange(10, 60)  
E_xy_values = []
E_dd_values = []

for R in radii:
    vortex = Vortex(n=1, R=R)  
    E_xy_values.append(vortex.E_xy) 
    E_dd_values.append(vortex.E_dd)  



plt.figure(figsize=(10, 6))
plt.plot(radii, E_xy_values, label=r"$E_{xy}$ ", marker="o", color="blue")
plt.plot(radii, E_dd_values, label=r"$E_{dd}$ ", marker="s", color="red")
plt.xlabel("Radius (R)")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.show()
