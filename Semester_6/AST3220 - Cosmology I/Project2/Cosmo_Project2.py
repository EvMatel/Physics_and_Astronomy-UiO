import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import astropy.constants as ac
import astropy.units as au
from scipy import constants as sc


class BBN:

    def __init__(self, N_species):
        self.c = ac.c.cgs.value  # cgs
        self.G = ac.G.cgs.value  # cgs
        self.k_b = ac.k_B.cgs.value  # cgs
        self.pi = sc.pi
        self.m_n = ac.m_n.cgs.value  # g
        self.m_p = ac.m_p.cgs.value  # g

        self.h_bar = ac.hbar.cgs.value  # cgs

        h = 0.7
        self.H_0 = (100*h*au.km/au.s/au.Mpc).cgs.value   # converting to cgs

        self.T_0 = 2.725
        self.N_species = N_species

        # calculating omega

        self.Omega_r0 = self.get_Omega()
        print(self.Omega_r0)

        self.rho_c0 = (3 * self.H_0**2) / (8*self.pi*self.G)


        self.species_labels = ["n", "p", "D", "T", "He3 ", " He4", "Li7", "Be7"]
        self.mass_number = [1, 1, 2, 3, 3, 4, 7, 7]
        self.colors_list = ['tab:orange', 'tab:blue', 'blue', 'yellow', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

        self.sol = None
        self.Omega_b0 = None

    def get_Omega(self):
        Neff = 3
        part1 = ((8 * (np.pi ** 3) * self.G) * (self.k_b * self.T_0)**4) / (45 * (self.H_0 ** 2) * (self.h_bar ** 3) * (self.c ** 5))
        part2 = (1 + Neff*(7 / 8)*(4 / 11)**(4/3))

        return part1*part2

    def t(self, T):
        return (self.T_0 / T)**2 * (1/(2*self.H_0)) * (1 / np.sqrt(self.Omega_r0))

    def diff_lign(self, lnT, init_vals):
        N_species = self.N_species
        T = np.exp(lnT)
        T9 = T / 1e9
        Y_n, Y_p = init_vals[:2]

        # calculating gammas/lambdas
        def Y(x, T, q):
            T_v = T * (4 / 11) ** (1 / 3)
            T_9 = T / 1e9
            T_9v = T_v / 1e9
            Z = 5.93 / T_9
            Z_v = 5.93 / T_9v
            first = (((x + q) ** 2) * (((x ** 2) - 1) ** (1/2)) * x) / ((1 + np.exp(x * Z)) * (1 + np.exp(-(x + q) * Z_v)))
            second = (((x - q) ** 2) * (((x ** 2) - 1) ** (1/2)) * x) / ((1 + np.exp(-x * Z)) * (1 + np.exp((x - q) * Z_v)))

            return first + second

        q = 2.53
        tau = 1700  # s
        x = np.arange(1, 200)
        y_pn = Y(x, T, -q)  # an array with all the values
        y_np = Y(x, T, q)
        lam_p = simpson(y_pn, x) / tau  # adds the array together and divides by tau
        lam_n = simpson(y_np, x) / tau

        a = self.T_0 / T
        H = self.H_0 * np.sqrt(self.Omega_r0) / a**2

        dY = np.zeros_like(np.array(init_vals))

        dY[0] = Y_p*lam_p - Y_n*lam_n
        dY[1] = Y_n*lam_n - Y_p*lam_p

        Omega_b0 = self.Omega_b0
        rho_b = (self.rho_c0 * Omega_b0) / a**3

        # Deuterium
        if N_species > 2:
            Y_D = init_vals[2]

            # n + p -> D + gamma (1)
            lam_pn = 2.5e4*rho_b
            lam_D = 4.68e9 * (lam_pn / rho_b) * T9**(3/2) * np.exp(-25.82 / T9)

            dY[0] += (-lam_pn*Y_n*Y_p) + (lam_D*Y_D)
            dY[1] += (-lam_pn*Y_n*Y_p) + (lam_D*Y_D)
            dY[2] += (lam_pn*Y_n*Y_p) - (lam_D*Y_D)

        # Tritium
        if N_species > 3:

            # n + D <-> T + gamma (3)
            Y_T = init_vals[3]
            lam_nD = rho_b*(75.5 + 1250*T9)
            lam_T = 1.63e10*(lam_nD/rho_b)*T9**(3/2)*np.exp(-72.62/T9)
            dY[0] += (-lam_nD*Y_n*Y_D) + (lam_T*Y_T)
            dY[2] += (-lam_nD*Y_n*Y_D) + (lam_T*Y_T)
            dY[3] += (lam_nD*Y_n*Y_D) - (lam_T*Y_T)

            # D + D <-> p + T (8)
            lam_DD_p = 3.9e8*(rho_b/(T9**(2/3)))*np.exp(-4.26*T9**(-1/3))*(1 + 0.0979*T9**(1/3) + 0.642*T9**(2/3) + 0.440*T9)
            gam_pT_D = 1.73*lam_DD_p*np.exp(-46.80/T9)
            dY[2] += (-lam_DD_p*Y_D*Y_D) + (2*gam_pT_D*Y_p*Y_T)
            dY[1] += (0.5*lam_DD_p*Y_D*Y_D) - (gam_pT_D*Y_p*Y_T)
            dY[3] += (0.5*lam_DD_p*Y_D*Y_D) - (gam_pT_D*Y_p*Y_T)

        # Helium-3
        if N_species > 4:
            # p + D <-> He3 + gamma (2)
            Y_He3 = init_vals[4]
            gam_pD = 2.23e3 * rho_b * T9**(-2/3)*np.exp(-3.72*T9**(-1/3))*(1 + 0.112*T9**(1/3) + 3.38*T9**(2/3) + 2.65*T9)
            lam_He3 = 1.63e10*(gam_pD / rho_b)*T9**(3/2)*np.exp(-63.75/T9)

            dY[1] += (-Y_p*Y_D*gam_pD) + (Y_He3*lam_He3)
            dY[2] += (-Y_p*Y_D*gam_pD) + (Y_He3*lam_He3)
            dY[4] += (Y_p*Y_D*gam_pD) - (Y_He3*lam_He3)

            # n + He3 <-> p + T (4)
            gam_nHe3_p = 7.06e8*rho_b
            gam_pT = gam_nHe3_p*np.exp(-8.864/T9)

            dY[0] += (-Y_n*Y_He3*gam_nHe3_p) + (Y_p*Y_T*gam_pT)
            dY[4] += (-Y_n*Y_He3*gam_nHe3_p) + (Y_p*Y_T*gam_pT)
            dY[2] += (Y_n*Y_He3*gam_nHe3_p) - (Y_p*Y_T*gam_pT)
            dY[3] += (Y_n*Y_He3*gam_nHe3_p) - (Y_p*Y_T*gam_pT)

            # D + D <-> n + He3 (7)
            lam_DD_n = lam_DD_p
            gam_nHe3_D = 1.73*lam_DD_n*np.exp(-37.94/T9)
            dY[2] += (-Y_D*Y_D*lam_DD_n) + (2*Y_n*Y_He3*gam_nHe3_D)
            dY[0] += (0.5*Y_D*Y_D*lam_DD_n) - (Y_n*Y_He3*gam_nHe3_D)
            dY[4] += (0.5*Y_D*Y_D*lam_DD_n) - (Y_n*Y_He3*gam_nHe3_D)


        # Helium-4
        if N_species > 5:
            Y_He4 = init_vals[5]

            # p + T <-> He4 + gamma (5)
            lam_pT_gam = 2.87e4*rho_b*T9**(-2/3)*np.exp(-3.87*T9**(-1/3))*(1 + 0.108*T9**(1/3) + 0.466*T9**(2/3) + 0.352*T9 + 0.3*T9**(4/3) + 0.576*T9**(5/3))
            lam_He4_p =2.59e10 * (lam_pT_gam/rho_b) * T9**(3/2) * np.exp(-229.9/T9)
            dY[1] += (-Y_p*Y_T*lam_pT_gam) + (Y_He4 * lam_He4_p)
            dY[3] += (-Y_p*Y_T*lam_pT_gam) + (Y_He4 * lam_He4_p)
            dY[5] += (Y_p*Y_T*lam_pT_gam) - (Y_He4 * lam_He4_p)

            # n + He3 <-> He4 + gamma (6)
            lam_nHe3_gam = 6e3*rho_b*T9
            lam_He4_n = 2.6e10*(lam_nHe3_gam/rho_b)*T9**(3/2)*np.exp(-238.8/T9)
            dY[0] += (-Y_n*Y_He3*lam_nHe3_gam) + (Y_He4*lam_He4_n)
            dY[4] += (-Y_n*Y_He3*lam_nHe3_gam) + (Y_He4*lam_He4_n)
            dY[5] += (Y_n*Y_He3*lam_nHe3_gam) - (Y_He4*lam_He4_n)

            # D + D <-> He4 + gamma (9)
            lam_DD_gam = 24.1*rho_b*T9**(-2/3)*np.exp(-4.26*T9**(-1/3))*(T9**(2/3) + 0.685*T9 + 0.152*T9**(4/3) + 0.265*T9**(5/3))
            lam_He4_D = 4.50e10*(lam_DD_gam/rho_b)*T9**(3/2)*np.exp(-276.7/T9)
            dY[2] += (-Y_D*Y_D*lam_DD_gam) + (2*Y_He4*lam_He4_D)
            dY[5] += (0.5*Y_D*Y_D*lam_DD_gam) - (Y_He4*lam_He4_D)

            # D + He3 <-> He4 + p (10)
            lam_DHe3 = 2.6e9*rho_b*T9**(-3/2)*np.exp(-2.99/T9)
            lam_He4p = 5.50*lam_DHe3*np.exp(-213.0/T9)
            dY[2] += (-Y_D*Y_He3*lam_DHe3) + (Y_He4*Y_p*lam_He4p)
            dY[4] += (-Y_D*Y_He3*lam_DHe3) + (Y_He4*Y_p*lam_He4p)
            dY[5] += (Y_D*Y_He3*lam_DHe3) - (Y_He4*Y_p*lam_He4p)
            dY[1] += (Y_D*Y_He3*lam_DHe3) - (Y_He4*Y_p*lam_He4p)

            # D + T <-> He4 + n (11)
            lam_DT = 1.38e9*rho_b*T9**(-3/2)*np.exp(-0.745/T9)
            lam_He4n = 5.50*lam_DT*np.exp(-204.1/T9)
            dY[2] += (-Y_D*Y_T*lam_DT) + (Y_He4*Y_n*lam_He4n)
            dY[3] += (-Y_D*Y_T*lam_DT) + (Y_He4*Y_n*lam_He4n)
            dY[5] += (Y_D*Y_T*lam_DT) - (Y_He4*Y_n*lam_He4n)
            dY[0] += (Y_D*Y_T*lam_DT) - (Y_He4*Y_n*lam_He4n)

            # He3 + T <-> He4 + D (15)
            lam_He3T_D = 3.88e9*rho_b*T9**(-2/3)*np.exp(-7.72*T9**(-1/3))*(1+0.0540*T9**(1/3))
            lam_He4D = 1.59*lam_He3T_D*np.exp(-166.2/T9)
            dY[4] += (-Y_He3*Y_T*lam_He3T_D) + (Y_He4*Y_D*lam_He4D)
            dY[3] += (-Y_He3*Y_T*lam_He3T_D) + (Y_He4*Y_D*lam_He4D)
            dY[5] += (Y_He3*Y_T*lam_He3T_D) - (Y_He4*Y_D*lam_He4D)
            dY[2] += (Y_He3*Y_T*lam_He3T_D) - (Y_He4*Y_D*lam_He4D)

        # Lithium-7
        if N_species > 6:
            Y_Li7 = init_vals[6]

            # T + He4 <-> Li7 + gamma (17)
            lam_THe4 = 5.28e5*rho_b*T9**(-2/3)*np.exp(-8.08*T9**(-1/3))*(1+0.0516*T9**(1/3))
            lam_Li7 = 1.12e10*(lam_THe4/rho_b)*T9**(3/2)*np.exp(-28.63/T9)
            dY[3] += (-Y_T*Y_He4*lam_THe4) + (Y_Li7*lam_Li7)
            dY[5] += (-Y_T*Y_He4*lam_THe4) + (Y_Li7*lam_Li7)
            dY[6] += (Y_T*Y_He4*lam_THe4) - (Y_Li7*lam_Li7)

            # p + Li7 <-> He4 + He4 (20)
            lam_pLi7_he4 = 1.42e9*rho_b*T9**(-2/3)*np.exp(-8.47*T9**(-1/3))*(1+0.0493*T9**(1/3))
            lam_He4He4_p = 4.64*lam_pLi7_he4*np.exp(-201.3/T9)
            dY[1] += (-Y_p*Y_Li7*lam_pLi7_he4) + (0.5*Y_He4*Y_He4*lam_He4He4_p)
            dY[6] += (-Y_p*Y_Li7*lam_pLi7_he4) + (0.5*Y_He4*Y_He4*lam_He4He4_p)
            dY[5] += (2*Y_p*Y_Li7*lam_pLi7_he4) - (Y_He4*Y_He4*lam_He4He4_p)

        # Berilium-7
        if N_species > 7:
            Y_Be7 = init_vals[7]

            # He3 + He4 <-> Be7 + gamma (16)
            lam_He3He4 = 4.8e6*rho_b*T9**(-2/3)*np.exp(-12.8*T9**(-1/3))*(1+0.0326*T9**(1/3) - 0.219*T9**(2/3) - 0.0499*T9 + 0.0258*T9**(4/3) + 0.0150*T9**(5/3))
            lam_Be7 = 1.12e10*(lam_He3He4/rho_b)*T9**(3/2)*np.exp(-18.42/T9)
            dY[4] += (-Y_He3*Y_He4*lam_He3He4) + (Y_Be7*lam_Be7)
            dY[5] += (-Y_He3*Y_He4*lam_He3He4) + (Y_Be7*lam_Be7)
            dY[7] += (Y_He3*Y_He4*lam_He3He4) - (Y_Be7*lam_Be7)

            # n + Be7 <-> p + Li7 (18)
            lam_nBe7_p = 6.74e9*rho_b
            lam_pLi7_n = lam_nBe7_p*np.exp(-19.07/T9)
            dY[0] += (-Y_n*Y_Be7*lam_nBe7_p) + (Y_p*Y_Li7*lam_pLi7_n)
            dY[7] += (-Y_n*Y_Be7*lam_nBe7_p) + (Y_p*Y_Li7*lam_pLi7_n)
            dY[1] += (Y_n*Y_Be7*lam_nBe7_p) - (Y_p*Y_Li7*lam_pLi7_n)
            dY[6] += (Y_n*Y_Be7*lam_nBe7_p) - (Y_p*Y_Li7*lam_pLi7_n)

            # n + Be7 <-> He4 + He4 (21)
            lam_nBe7_He4 = 1.2e7*rho_b*T9
            lam_He4He4_n = 4.64*lam_nBe7_He4*np.exp(-220.4/T9)
            dY[0] += (-Y_n*Y_Be7*lam_nBe7_He4) + (0.5*Y_He4*Y_He4*lam_He4He4_n)
            dY[7] += (-Y_n*Y_Be7*lam_nBe7_He4) + (0.5*Y_He4*Y_He4*lam_He4He4_n)
            dY[5] += (2*Y_n*Y_Be7*lam_nBe7_He4) - (Y_He4*Y_He4*lam_He4He4_n)




        return - dY / H

    def np_init_vals_BBN(self, T_i):
        Y_n = (1 + np.exp(((self.m_n - self.m_p) * self.c ** 2) / (self.k_b * T_i))) ** -1
        Y_p = 1 - Y_n

        return Y_n, Y_p

    def solve_BBN(self, Omega_b0=0.05, T_init=100e9, T_end=0.01e9):

        self.Omega_b0 = Omega_b0

        self.T_init = T_init
        self.T_end = T_end

        init_vals = list(self.np_init_vals_BBN(T_init))

        if self.N_species > 2:
            init_vals = list(self.np_init_vals_BBN(T_init)) + [0] * (self.N_species - 2)

        T_span = [np.log(T_init), np.log(T_end)]
        T_eval = np.linspace(np.log(T_init), np.log(T_end), 1000)

        sol = solve_ivp(self.diff_lign, t_span=T_span, t_eval=T_eval, y0=init_vals, method='Radau', rtol=1e-12, atol=1e-12)
        self.sol = sol

        return sol

    def plot_BBN(self):
        Y = self.sol.y
        print(Y.shape)

        T_eval = np.linspace(np.log(self.T_init), np.log(self.T_end), len(self.sol.t))
        T = np.exp(T_eval)

        for i in range(self.N_species):
            plt.plot(T, self.mass_number[i]*Y[i], label=self.species_labels[i], color=self.colors_list[i])

        Y_n_eq, Y_p_eq = bbn.np_init_vals_BBN(T)
        if self.N_species <= 3:
            plt.plot(T, Y_p_eq, linestyle=':', color='tab:blue')
            plt.plot(T, Y_n_eq, linestyle=':', color='orange')

        sum = np.sum(Y, axis=0)
        if self.N_species > 3:
            plt.plot(T, sum, linestyle='--', color='black', label=r'$\sum A \cdot Y$')


        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.ylim(1e-3, 1.5)
        plt.xlim(1e11, 1e8)
        plt.ylabel(r'$Y_i$')
        if self.N_species > 2:
            plt.ylabel(r'Mass Fraction $A_i Y_i$')
        plt.xlabel('T [K]')
        if self.N_species > 3:
            plt.xlim(1e11, 0.01e9)
            plt.ylim(1e-11, 1.5)

        plt.show()

    def plot_relic(self):
        if self.N_species < 8:
            assert AssertionError('N_species must be 8 to use this function.')

        rough_omega_b0 = np.logspace(np.log(0.01), np.log(1), 10)
        smooth_omega_b0 = np.logspace(np.log(0.01), np.log(1), 1000)

        new_Y = []
        for i in rough_omega_b0:
            sol = self.solve_BBN(Omega_b0=i)
            Y = sol.y[:, -1]
            new_Y.append(Y)

        new_Y_arr = np.asarray(new_Y).T
        print(new_Y_arr)
        interp_f = interp1d(rough_omega_b0, new_Y_arr, kind='cubic')
        interp_Y = interp_f(smooth_omega_b0)
        Y_p = np.exp(interp_Y[1, :])
        Y_He3 = np.exp(interp_Y[4, :])
        Y_He4 = np.exp(interp_Y[5, :])
        Y_D = np.exp(interp_Y[2, :])
        Y_Li7 = np.exp(interp_Y[6, :])

        plt.fill_between(smooth_omega_b0, 0.254 - 0.003, 0.254 + 0.003, alpha=0.2, color="C2")
        plt.plot(smooth_omega_b0, 4*Y_He4, color='r', label='He4')
        #plt.ylim(0.20, 0.30)
        #plt.xlim(0.01, 1)
        plt.xscale('log')
        plt.legend()
        plt.show()

        '''
        plt.fill_between(smooth_omega_b0, 0.254 - 0.003, 0.254 + 0.003, alpha=0.2, color="C2")
        plt.plot(smooth_omega_b0, 4 * Y_He4, color='r', label='He4')
        plt.ylim(0.20, 0.30)
        plt.xlim(0.01, 1)
        plt.xscale('log')
        plt.legend()
        plt.show()
        '''
        plt.fill_between(smooth_omega_b0, 2.57e-5 - 0.03e-5, 2.57e-5 + 0.003e-5, alpha=0.2, color="blue")
        plt.fill_between(smooth_omega_b0, 1.6e-10 - 0.3e-10, 1.6e-10 + 0.3e-10, alpha=0.2, color="r")
        plt.plot(smooth_omega_b0, Y_D/Y_p, label='D', color='blue')
        plt.plot(smooth_omega_b0, Y_He3/Y_p, label='He3', color='orange')
        plt.plot(smooth_omega_b0, Y_Li7/Y_p, label='Li7', color='r')
        plt.loglog()
        #plt.ylim(1e-11, 1e-3)
        #plt.xlim(1e-2, 1)
        plt.legend()
        plt.show()



if __name__ == '__main__':

    bbn = BBN(8)
    bbn.solve_BBN()
    #bbn.plot_BBN()
    bbn.plot_relic()
    T_values = [1e10, 1e9, 1e8]
    for i in T_values:
        print(f'The age of the universe at temperature {i:.2g}K is {bbn.t(i):.2f}s.')


