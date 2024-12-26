import numpy as np
import matplotlib.pyplot as plt

from vortex import Vortex,eps



### 1

# vortex=Vortex(n=1,R=20)
# vortex.plot()


# vortex=Vortex(n=1,R=20)
# vortex.plot()

# vortex=Vortex(n=0,R=20)
# vortex.plot()

# vortex=Vortex(n=-1,R=20)
# vortex.plot()

### 2

winding = np.arange(-1, 2)  
R_values = np.linspace(10, 60, 50)  



# energies_xy = np.zeros((len(winding),len(R_values)))
# energies_dd = np.zeros((len(winding),len(R_values)))

# for i, n in enumerate(winding):
#     for j, R in enumerate(R_values):
#         vortex = Vortex(n=n,R=R)
#         energies_xy[i,j] = vortex.E_xy
#         energies_dd[i,j] = vortex.E_dd
#         print(len(winding)-i, len(R_values)-j)


# np.savetxt('xy_file',energies_xy)
# np.savetxt('dd_file',energies_dd)



### 3

# J = 0.1

# energies_xy = np.loadtxt('xy_file')
# energies_dd = np.loadtxt('dd_file')

# # energies = np.zeros((len(winding),len(R_values)))


# # for i in range()
# #     # energies[i,:] = energies_xy[i,:] + J* energies_dd[i,:]
# #     plt.plot(R_values,energies[i,:], label = 'n = ' + str(windingn))

# for i, n in enumerate(winding):
#     plt.plot(R_values,energies_xy[i,:]+1e-3*i, label = 'n = ' + str(n))
# plt.legend()
# plt.ylabel("$E_{xy}$ [$J_1$]")
# plt.xlabel("R [$R_0$]")
# plt.savefig('xy_energies')
# plt.show()

# for i, n in enumerate(winding):
#     plt.plot(R_values,energies_dd[i,:], label = 'n = ' + str(n))
# plt.legend()
# plt.ylabel("$E_{dd}$ [$J_2$]")
# plt.xlabel("R [$R_0$]")
# plt.savefig('dd_energies')
# plt.show()
    



#####4

# R = 5

# x = np.arange(-R - 1, R) + 0.5
# x, y = np.meshgrid(x, x)
# print(x)

# print(y)


######5

# for i in [-1,0,1]:
#     vortex = Vortex(n=i, R=20)
#     vortex.plot_density()
# plt.show()


#####6

R = 20

vortex1 = Vortex(n=1, R=R)
vortex2 = Vortex(n=-1, R=R)





plt.figure()

im = plt.scatter(
    vortex1.x,
    vortex1.y,
    c=vortex1.E_dd_density- vortex2.E_dd_density,
    vmin=-2,
    vmax=2,
    cmap='coolwarm' 
)
cbar=plt.colorbar(im)


plt.title("Vortex")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.axis("equal")
# plt.legend(loc="upper right")
plt.show()