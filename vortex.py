import numpy as np
import matplotlib.pyplot as plt
import numba

eps = 1e-5

@numba.njit(fastmath=True)
def calculate_dipole_dipole(theta, beta, d):
    N = len(theta)
    E_density = np.zeros(N)
    E_dd = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                prefact = 1 / (d[i, j] ** 3)
                term_1 = np.cos(theta[i] - theta[j])
                term_2 = (
                    3 * np.cos(theta[i] - beta[i, j]) * np.cos(theta[j] - beta[i, j])
                )
                energy = prefact * (term_1 - term_2)
                E_dd = energy
                E_density[i] = energy
    return E_dd, E_density


class Vortex(object):
    def __init__(self, n=0, R=1) -> None:
        x = np.arange(-R - 1, R) + 0.5
        x, y = np.meshgrid(x, x)

        r = np.sqrt(x**2 + y**2)
        self.x, self.y = x[r <= R], y[r <= R]
        phi = np.arctan2(self.y, self.x) + np.pi / 2
        self.u = np.cos(n * phi)
        self.v = np.sin(n * phi)
        self.theta = np.arctan2(self.v, self.u)

        dx = self.x[:, None] - self.x[None, :]
        dy = self.y[:, None] - self.y[None, :]
        self.d = np.sqrt(dx**2 + dy**2)

        self.nn = np.where(np.abs(self.d - 1) < eps)
        self.beta = np.arctan2(dy, dx)

        i, j = self.nn
        self.E_xy = np.sum(np.cos(self.theta[i] - self.theta[j]))

        self.N = len(self.x)

        self.E_dd, self.E_density = calculate_dipole_dipole(
            theta=self.theta, beta=self.beta, d=self.d
        )

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
        plt.show()

    # def plot_energy_density(self):
    #     plt.figure()
    #     plt.tricontourf(
    #         self.x, self.y, self.E_density, cmap="viridis", levels=100
    #     )
    #     plt.colorbar(label="Energy Density")
    #     plt.title("Energy Density of Dipole-Dipole Interaction")
    #     plt.xlabel("x-axis")
    #     plt.ylabel("y-axis")
    #     plt.axis("equal")
    #     plt.show()


    def plot_energy_density(self):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            self.x, self.y, c=self.E_density, cmap="coolwarm"#, s=500
        )
        plt.colorbar(scatter, label="Energy Density")
        plt.title("Energy Density")
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.axis("equal")
        plt.show()