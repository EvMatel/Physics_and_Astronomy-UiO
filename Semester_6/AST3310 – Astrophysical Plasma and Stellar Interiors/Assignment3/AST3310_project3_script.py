# visualiser
import FVis3 as FVis
import numpy as np
import astropy.constants as ac
import matplotlib.pyplot as plt



class TWODconvection:

    def __init__(self):
        """
        define variables
        """
        self.N_x = 300
        self.N_y = 100
        self.dX = 12e6 / (self.N_x - 1)  # km
        self.dY = 4e6 / (self.N_y - 1)  # km

        self.u_arr = np.zeros((self.N_y, self.N_x))
        self.w_arr = np.zeros((self.N_y, self.N_x))
        self.T_arr = np.zeros((self.N_y, self.N_x))
        self.P_arr = np.zeros((self.N_y, self.N_x))
        self.rho_arr = np.zeros((self.N_y, self.N_x))
        self.e_arr = np.zeros((self.N_y, self.N_x))

        self.y_arr = np.linspace(0, 4e6, self.N_y)
        self.x_arr = np.linspace(0, 12e6, self.N_x)

        self.nabla = (2/5) + 1e-6
        self.gamma = 5/3
        self.mu = 0.61
        self.m_u = ac.u.value
        self.k_b = ac.k_B.value
        self.T_0 = 5778  # K
        self.rho_0 = 2.3e-4  # kg/m^3
        self.P_0 = 1.8e4  # Pa

        self.g = -(ac.G.value * ac.M_sun.value) / (ac.R_sun.value**2)  # constant gravity

    def initialise(self, pert=False):
        """
        initialise temperature, pressure, density and internal energy
        """
        for i in range(self.N_y):
            y = np.linspace(4e6, 0, self.N_y)
            self.T_arr[i, :] = -((self.nabla * self.mu * self.m_u * self.g * y[i]) / self.k_b) + self.T_0
            self.P_arr[i, :] = self.P_0 * (self.T_arr[i, :] / self.T_0) ** (1 / self.nabla)

        # adding pertubations
        if pert==True:
            x, y = np.meshgrid(self.x_arr, self.y_arr)
            sigma = 0.9e6
            mu_x, mu_y = 6e6, 2e6
            p_g = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))
            p_g *= 25_000

            self.T_arr[:, :] += p_g[:, :]

        for i in range(self.N_y):
            self.rho_arr[i, :] = (self.P_arr[i, :] * self.mu * self.m_u) / (self.T_arr[i, :] * self.k_b)
            self.e_arr[i, :] = self.P_arr[i, :] / (self.gamma - 1)

    def boundary_conditions(self):

        # Vertical boundary: Vertical velocity
        self.w_arr[0, :] = 0
        self.w_arr[-1, :] = 0

        # Vertical boundary: Horizontal velocity
        self.u_arr[0, :] = (- self.u_arr[2, :] + 4*self.u_arr[1, :]) / 3
        self.u_arr[-1, :] = (4*self.u_arr[-2, :] - self.u_arr[-3, :]) / 3

        # Vertical boundary: Density and energy

        self.e_arr[0, :] = (4*self.e_arr[1, :] - self.e_arr[2, :]) / (3 + 2 * self.dY * self.g * self.mu * self.m_u / (self.k_b * self.T_arr[0, :]))
        self.e_arr[-1, :] = (4*self.e_arr[-2, :] - self.e_arr[-3, :]) / (3 - 2 * self.dY * self.g * self.mu * self.m_u / (self.k_b * self.T_arr[-1, :]))

        self.rho_arr[0, :] = (self.gamma - 1) * self.e_arr[0, :] * ((self.mu * self.m_u) / (self.k_b * self.T_arr[0, :]))
        self.rho_arr[-1, :] = (self.gamma - 1) * self.e_arr[-1, :] * ((self.mu * self.m_u) / (self.k_b * self.T_arr[-1, :]))

    def central_x(self, var): # take [i, j] and gives [i+1, j]
        """
        central difference scheme in x-direction
        """
        forward = np.roll(var, axis=1, shift=-1)
        back = np.roll(var, axis=1, shift=1)

        dvar_dx = (forward - back) / (2 * self.dX)

        return dvar_dx

    def central_y(self, var):
        """
        central difference scheme in y-direction
        """

        dvar_dy = np.zeros((self.N_y, self.N_x))

        for j in range(self.N_y-1):
            dvar_dy[j, :] = (var[j+1, :] - var[j-1, :]) / (2*self.dY)

        '''forward = np.roll(var, axis=1, shift=-1)
        back = np.roll(var, axis=1, shift=1)

        dvar_dy = (forward - back) / (2 * self.dY)'''

        return dvar_dy

    def upwind_x(self, var, v):
        """
        upwind difference scheme in x-direction
        """
        dvar_dx = np.zeros((self.N_y, self.N_x))

        forward = np.roll(var, axis=1, shift=-1)  # i+1
        back = np.roll(var, axis=1, shift=1)      # i-1

        dvar_dx[v >= 0] = (var[v >= 0] - back[v >= 0]) / self.dX
        dvar_dx[v < 0] = (forward[v < 0] - var[v < 0]) / self.dX

        return dvar_dx

    def upwind_y(self, var, v):
        """
        upwind difference scheme in y-direction
        """
        dvar_dy = np.zeros((self.N_y, self.N_x))

        forward = np.roll(var, axis=0, shift=-1)
        back = np.roll(var, axis=0, shift=1)

        dvar_dy[v >= 0] = (var[v >= 0] - back[v >= 0]) / self.dY
        dvar_dy[v < 0] = (forward[v < 0] - var[v < 0]) / self.dY

        return dvar_dy

    def step(self):
        """
        hydrodynamic equations solver
        """

        # the time differentials

        drho_dt = - self.rho_arr * (self.central_x(self.u_arr) + self.central_y(self.w_arr))\
                  - (self.u_arr * self.upwind_x(self.rho_arr, self.u_arr)) \
                  - (self.w_arr * self.upwind_y(self.rho_arr, self.w_arr))

        drhou_dt = - (self.rho_arr * self.u_arr) * (self.upwind_x(self.u_arr, self.u_arr) + self.central_y(self.w_arr))\
                   - (self.u_arr * self.upwind_x((self.rho_arr * self.u_arr), self.u_arr)) \
                   - (self.w_arr * self.upwind_y((self.rho_arr * self.u_arr), self.w_arr)) - self.central_x(self.P_arr)

        drhow_dt = - (self.rho_arr * self.w_arr) * (self.upwind_y(self.w_arr, self.w_arr) + self.central_x(self.u_arr))\
                   - (self.w_arr * self.upwind_y((self.rho_arr * self.w_arr), self.w_arr)) \
                   - (self.u_arr * self.upwind_x((self.rho_arr * self.w_arr), self.u_arr)) \
                   - self.central_y(self.P_arr) + (self.g * self.rho_arr)

        de_dt = - ((self.u_arr * self.upwind_x(self.e_arr, self.u_arr)) + (self.e_arr * self.central_x(self.u_arr))) \
                - ((self.w_arr * self.upwind_y(self.e_arr, self.w_arr)) + (self.e_arr * self.central_y(self.w_arr))) \
                - (self.P_arr * (self.central_x(self.u_arr) + self.central_y(self.w_arr)))


        # calculating dt

        rel_rho = abs(drho_dt / self.rho_arr)
        rel_e = abs(de_dt / self.e_arr)
        rel_x = abs(self.u_arr / self.dX)
        rel_y = abs(self.w_arr / self.dY)

        rel_list = [rel_rho.max(), rel_e.max(), rel_x.max(), rel_y.max()]
        delta = max(rel_list)
        p = 0.1

        dt = p / delta

        if np.isnan(dt) == True:
            dt = 0.1

        if dt < 0.01:
            dt = 0.01

        if dt > 0.5:
            dt = 0.1


        # calculating next step

        # define rho_n+1 first so that i can use it along with rho_n+1 to calculate u and w_n+1
        rho_arr_new = self.rho_arr[:, :] + drho_dt[:, :] * dt

        self.u_arr[:, :] = ((self.rho_arr[:, :] * self.u_arr[:, :]) + (drhou_dt[:, :] * dt)) / rho_arr_new[:, :]
        self.w_arr[:, :] = ((self.rho_arr[:, :] * self.w_arr[:, :]) + (drhow_dt[:, :] * dt)) / rho_arr_new[:, :]

        # can now redefine rho, since rho_n is no longer needed
        self.rho_arr[:, :] = rho_arr_new[:, :]
        self.e_arr[:, :] += (de_dt[:, :] * dt)

        # applying boundary conditions
        self.boundary_conditions()

        # calculating P and T with the already calculated e_n+1 and rho_n+1
        self.P_arr[:, :] = (self.gamma - 1)*self.e_arr[:, :]
        self.T_arr[:, :] = (self.P_arr[:, :] * self.mu * self.m_u) / (self.rho_arr[:, :] * self.k_b)

        return dt


if __name__ == '__main__':

    solver = TWODconvection()
    solver.initialise(pert=True)

    vis = FVis.FluidVisualiser()
    vis.save_data(250, solver.step, u=solver.u_arr, w=solver.w_arr, rho=solver.rho_arr,
                  T=solver.T_arr, P=solver.P_arr, e=solver.e_arr, sim_fps=2)
    vis.animate_2D('T', units={"Lx": "Mm", "Lz": "Mm"}, extent=[0, 4e6, 0, 12e6], save=True)


