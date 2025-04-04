import matplotlib . pyplot as plt
import numpy as np
from scipy.integrate import simpson, solve_ivp

class BBN:

    def __init__(self, N_species):
        self.N_species = N_species  # The number of particle species
        self.species_labels = ["n", "p", "D", "T", "He3 ", " He4", "Li7", "Be7"]  # The particle names
        self.mass_number = [1, 1, 2, 3, 3, 4, 7, 7]  # The particle atomic numbers
        self.NR_points = 1001 # the number of points for our solution arrays

        h = 0.7
        H_0 = 100 * h / 1e19  # converting to cgs
        self.H_0 = H_0

        self.T_0 = 2.725

        self.Omega_r0 = self.get_Omega_r0(self.T_0)


    def get_a(self, T):
        return self.T_0 / T


    def get_Omega_r0(self, T_0):

        Neff = 3
        c = 2.99792458e10  # cgs
        G = 6.67e-8  # cgs
        k_b = 1.3807e-16  # cgs
        h_bar = 1.0546e-27  # cgs

        part1 = ((8 * (np.pi ** 3) * G) * (k_b * T_0) ** 4) / (45 * (self.H_0 ** 2) * (h_bar ** 3) * (c ** 5))
        part2 = (1 + Neff * (7 / 8) * (4 / 11) ** (4 / 3))
        Omega = part1 * part2

        return Omega

    def get_Hubble(self, T):
        a = self.T_0 / T
        H = self.H_0 * np.sqrt(self.Omega_r0) / a ** 2

        return H

    def get_ODE(self, lnT, Y):
        T = np.exp(lnT)
        T9 = T / 1e9
        H = self.get_Hubble(T9)

        a = self.get_a(T9)
        G = 6.67e-8
        rho_c0 = 3 * self.H_0**2 / 8*np.pi*G

        rho_b = a * rho_c0 * self.Omega_r0

        Hubble_inv = 1 / H
        dY = np.zeros_like(Y)

        # Weak interactions , always included ~ (n <-> p) (a 1-3)
        Y_n, Y_p = Y[0], Y[1]

        def gamma_integral(x, T, q):
            T_v = T * (4 / 11) ** (1 / 3)
            T_9 = T / 1e9
            T_9v = T_v / 1e9
            Z = 5.93 / T_9
            Z_v = 5.93 / T_9v
            first = (((x + q) ** 2) * (((x ** 2) - 1) ** (1 / 2)) * x) / (
                        (1 + np.exp(x * Z)) * (1 + np.exp(-(x + q) * Z_v)))
            second = (((x - q) ** 2) * (((x ** 2) - 1) ** (1 / 2)) * x) / (
                        (1 + np.exp(-x * Z)) * (1 + np.exp((x - q) * Z_v)))

            return first + second

        q = 2.53
        tau = 1700  # s
        x = np.arange(1, 200)
        y_pn = gamma_integral(x, T, -q)  # an array with all the values
        y_np = gamma_integral(x, T, q)
        lambda_p = simpson(y_pn, x) / tau  # adds the array together and divides by tau
        lambda_n = simpson(y_np, x) / tau


        dY[0] += Y_p * lambda_p - Y_n * lambda_n  # The change to the neutron fraction
        dY[1] += Y_n * lambda_n - Y_p * lambda_p  # The change to the proton fraction

        if self.N_species > 2:  # Include deuterium
            Y_D = Y[2]
            # (n+p <-> D+ gamma ) (b.1)
            Y_np = Y_n * Y_p
            rate_np, rate_D = self.RR.get_np_to_D(T9, rho_b)
            dY[0] += Y_D * rate_D - Y_np * rate_np  # Update the change to neutron fraction
            dY[1] += Y_D * rate_D - Y_np * rate_np  # Update the change to proton fraction
            dY[2] += Y_np * rate_np - Y_D * rate_D  # Update the change to deuterium fraction

        if self.N_species > 3:  # Include tritium
            """
            Continue with the reactions including tritium ...
            """

        if self.N_species > 4:  # Include He3
            """
            Continue with the reactions including Helium 3 ...
            This could be extended all the way to self . NR_species = 8 to include all the species
            """
        # Each reaction equation above should be multiplied with ( -1/ Hubble ) before return :

        return -dY * Hubble_inv


    def set_IC(self, T_init):
        """
        Defines the initial condition array used in solve_BBN .
        The only nonzero values are for neutrons and protons , but the shape of self . Yinit is
        determined by the number of particles included .
        Arguments :
        T_init { float } -- the initial temperature
        """

        def Ys(T_i):  # takes T and returns Y_n and Y_p
            c = 2.99792458e10  # cgs
            k_b = 1.3807e-16  # cgs

            m_n = 1.6749286e-24  # g
            m_p = 1.67262e-24  # g
            Y_n = (1 + np.exp(((m_n - m_p) * c ** 2) / (k_b * T_i))) ** -1
            Y_p = 1 - Y_n

            return Y_n, Y_p

        self.Yinit = np.zeros(self.N_species)
        Yn_init, Yp_init = Ys(T_init)  # solves equations (16 -17) from the project
        self.Yinit[0] = Yn_init
        self.Yinit[1] = Yp_init

    def solve_BBN(self, T_init: float = 100e9, T_end: float = 0.01e9):

        """
        Solves the BBN - system for a given range of temperature values
        Keyword Arguments :
        T_init { float } -- the initial temperature ( default : { 100e9 })
        T_end { float } -- the final temperature ( default : {0. 01e9 })
        """
        self.T_init = T_init
        self.T_end = T_end
        self.set_IC(T_init)  # This is just a function defining self . Yinit using eq. (16 -17)
        sol = solve_ivp(  # solve the ODE - system using scipy . solve_ivp
            self.get_ODE,
            [np.log(T_init), np.log(T_end)],  # our equations are defined over ln(T),
            # so we have to define the integration interval using the log of temperature
            y0=self.Yinit,
            method="Radau",
            rtol=1e-12,
            atol=1e-12,
            dense_output=False  # this allows us to extract the solvables following the procedure below
        )

        '''        
        # Now we need to extract the solution from solve_ivp . BE AWARE OF THE INTERNAL VARIABLE :
        # Define array of linearly spaced ln(T) values using the start and endpoint from the solver :

        lnT = np.linspace(sol.t[0], sol.t[-1], self.NR_points)  # This corresponds to the internal variable

        self.Y = sol.sol(lnT)  # use ln(T) to extract the solved solutions ( dense_output = True )

        # Use the linearly spaced ln(T) to create a corresponding logarithmically spaced T array
        self.T = np.exp(lnT)  # this array is used for plotting , individual points match self .Y
        '''
        self.sol = sol

    def plot_BBN(self, eq=None):
        print(self.sol.y)
        Y_n_arr, Y_p_arr = self.sol.y

        T_eval = np.linspace(np.log(self.T_init), np.log(self.T_end), self.NR_points)
        T = np.exp(T_eval)

        plt.plot(T, Y_p_arr, label='p')
        plt.plot(T, Y_n_arr, label='n')
        if eq:
            Y_n_eq, Y_p_eq = Ys(T)
            plt.plot(T, Y_n_eq, linestyle=':', color='orange')
            plt.plot(T, Y_p_eq, linestyle=':', color='tab:blue')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.ylim(1e-3, 1.5)
        plt.gca().invert_xaxis()


if __name__ == '__main__':

    bbn = BBN(2)  # Initiate the system including neutrons , protons and deuterium
    bbn.solve_BBN()  # solve the system until end temperature 0.1*10^9 K
    bbn.plot_BBN()


    '''
        fig, ax = plt.subplots()
    for i, y in enumerate(bbn.Y):
        ax.loglog(bbn.T, bbn.mass_number[i] * y, label=bbn.species_labels[i])
    '''

