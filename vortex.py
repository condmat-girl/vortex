import numpy as np
import matplotlib.pyplot as plt
import numba

eps = 1e-8


@numba.njit(fastmath=True)
def calculate_dipole_dipole(theta, beta, d):
    N = len(theta)

    E_dd = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                prefact = 1 / (d[i, j] ** 3)
                term_1 = np.cos(theta[i] - theta[j])
                term_2 = (
                    3 * np.cos(theta[i] - beta[i, j]) * np.cos(theta[j] - beta[i, j])
                )
                E_dd[i,j] += prefact * (term_1 + term_2)
    return np.sum(E_dd), E_dd.sum(axis=1)


class Vortex(object):
    def __init__(self, n=0, R=1) -> None:

        x = np.arange(-R - 1, R) + 0.5
        x, y = np.meshgrid(x, x)

        r = np.sqrt(x**2 + y**2)
        self.x, self.y = x[r <= R], y[r <= R]
        phi = np.arctan2(self.y, self.x) + np.pi/2
        self.u = np.cos(n * phi)
        self.v = np.sin(n * phi)
        self.theta = np.arctan2(self.v, self.u) 

        dx = self.x[:, None] - self.x[None, :]
        dy = self.y[:, None] - self.y[None, :]
        self.d = np.sqrt(dx**2 + dy**2)

        self.nn = np.where(np.abs(self.d - 1) < eps)
        self.beta = np.arctan2(dy, dx)

        i, j = self.nn
        self.E_xy = - np.sum(np.cos(self.theta[i] - self.theta[j]))

        self.N = len(self.x)

        self.E_dd, self.E_dd_density = calculate_dipole_dipole(theta=self.theta, beta=self.beta, d=self.d)

        self.E_dd /= self.N
        self.E_xy /= self.N


    def plot(self):

        plt.figure()

        plt.quiver(
            self.x,
            self.y,
            self.u,
            self.v,
            scale=0.8,
            scale_units="xy",
            pivot="mid",
            color="blue",
        )

        plt.title("Vortex")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.axis("equal")
        plt.legend(loc="upper right")
        plt.show()



    def plot_density(self):

        plt.figure()

        im = plt.quiver(
            self.x,
            self.y,
            self.u,
            self.v,
            self.E_dd_density,
            scale=0.8,
            scale_units="xy",
            pivot="mid",

            
        )
        cbar=plt.colorbar(im)


        plt.title("Vortex")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.axis("equal")
        plt.legend(loc="upper right")
        plt.show()





    
    def plot_energy_density(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(
            self.E_dd_density,
            extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
            origin="lower",
            cmap="viridis",
        )
        plt.colorbar(label="Dipole-Dipole Energy Density")
        plt.title("Energy Density on Lattice")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.show()