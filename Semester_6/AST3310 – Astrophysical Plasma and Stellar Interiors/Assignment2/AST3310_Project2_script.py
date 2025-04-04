import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import isclose

class ConvectionZone:

    def __init__(self):
        self.mu = self.mean_mol_weight_calc()
        self.opacity_func = None
        self.epsilon_func = None
        self.kappa_T_vals = None
        self.kappa_R_vals = None
        self.eps_T_vals = None
        self.eps_R_vals = None
        self.p = None
        self.R_0 = None
        self.P_0 = None
        self.L_0 = None
        self.T_0 = None

    def read_opacity(self):  # inputs must be given in SI units
        infile = open('opacity.txt', 'r')  # import data
        data = []
        for line in infile:
            l = np.array(line.split())
            data.append(l)
        data.pop(1)             # get rid of the blank line in the data
        data = np.array(data)   # 2D array of the data
        T_vals = np.array(data[:, 0][1:], dtype=float)  # y-axis
        R_vals = np.array(data[0, :][1:], dtype=float)  # x-axis
        k_vals = np.array(data[1:, 1:], dtype=float)  # values

        self.opacity_func = sp.interpolate.RectBivariateSpline(T_vals, R_vals, k_vals)  # interpolating data

        self.kappa_T_vals, self.kappa_R_vals = T_vals, R_vals

    def kappa(self, T, rho, Sanity=0):  # takes T and rho in SI

        if Sanity == 1:  # for the sanity check
            R_log = rho
            T_log = T
        else:
            T_log = np.log10(T)
            rho_cgs = rho / 1000  # converted from SI to CGS
            R = (rho_cgs / ((T / 1e6)**3))
            R_log = np.log10(R)

            '''
            if T_log < self.kappa_T_vals[0] or T_log > self.kappa_T_vals[-1]:
                print('T is outside of the data set: answer is extrapolated.')

            if R_log < self.kappa_R_vals[0] or R_log > self.kappa_R_vals[-1]:
                print('R is outside of the data set: answer is extrapolated.')
            '''

        kappa_log_cgs = float(self.opacity_func(T_log, R_log))
        kappa_SI = (10**kappa_log_cgs) * 0.1  # getting rid of the logorithm and then converting from CGS to SI

        if Sanity == 1:  # for the sanity check
            return kappa_log_cgs, kappa_SI
        else:
            return kappa_SI

    def read_epsilon(self):  # inputs are in SI units
        infile = open('epsilon.txt', 'r')  # import data
        data = []
        for line in infile:
            l = np.array(line.split())
            data.append(l)
        data.pop(1)             # get rid of the blank line in the data
        data = np.array(data)   # 2D array of the data
        T_vals = np.array(data[:, 0][1:], dtype=float)
        R_vals = np.array(data[0, :][1:], dtype=float)
        eps_vals = np.array(data[1:, 1:], dtype=float)


        self.epsilon_func = sp.interpolate.RectBivariateSpline(T_vals, R_vals, eps_vals)

    def epsilon(self, T, rho, Sanity=0):

        if Sanity == 1:  # for the sanity check
            R_log = rho
            T_log = T
        else:
            T_log = np.log10(T)
            rho_cgs = rho / 1000  # converted from SI to CGS
            R = rho_cgs / (T / 1e6)**3
            R_log = np.log10(R)

        eps_log_cgs = float(self.epsilon_func(T_log, R_log))
        eps_SI = (10**eps_log_cgs) * 0.0001  # converted from CGS to SI

        if Sanity == 1:  # for the sanity check
            return eps_log_cgs, eps_SI
        else:
            return eps_SI

    def opacity_sanity_check(self):  # all checks passed with a tolerance of 0.02
        T = np.array([3.75, 3.755, 3.755, 3.755, 3.755, 3.770, 3.780, 3.795, 3.770, 3.775, 3.780, 3.795, 3.800])
        R = np.array([-6, -5.95, -5.8, -5.7, -5.55, -5.95, -5.95, -5.95, -5.80, -5.75, -5.70, -5.55, -5.50])
        k_expected = np.array([-1.55, -1.51, -1.57, -1.61, -1.67, -1.33, -1.20, -1.02, -1.39, -1.35, -1.31, -1.16, -1.11])
        k_expected_SI = np.array([2.84e-3, 3.11e-3, 2.68e-3, 2.46e-3, 2.12e-3, 4.70e-3, 6.25e-3, 9.45e-3, 4.05e-3, 4.43e-3, 4.94e-3, 6.89e-3, 7.69e-3])

        for i in range(len(T)):
            tol = 0.02
            pass_or_fail_cgs = isclose(self.kappa(T[i], R[i], Sanity=1)[0], k_expected[i], rel_tol=tol)
            assert (pass_or_fail_cgs == True)

            tol = 0.0003
            pass_or_fail_SI = isclose(round(self.kappa(T[i], R[i], Sanity=1)[1], 5), k_expected_SI[i], abs_tol=tol)
            assert (pass_or_fail_SI == True)

        print(f'*** Opacity sanity check passed ***')

    def epsilon_sanity_check(self):
        T = np.array([3.750, 3.755])
        R = np.array([-6.00, -5.95])
        eps_expected_cgs = np.array([-87.995, -87.267])
        eps_expected_SI = np.array([1.012e-92, 5.401e-92])

        for i in range(len(T)):
            tol = 0.35
            pass_or_fail_cgs = isclose(self.epsilon(T[i], R[i], Sanity=1)[0], eps_expected_cgs[i], rel_tol=tol)
            assert (pass_or_fail_cgs == True)

            print(self.epsilon(T[i], R[i], Sanity=1)[1], eps_expected_SI[i])
            tol = 2.0e-94
            pass_or_fail_SI = isclose(self.epsilon(T[i], R[i], Sanity=1)[1], eps_expected_SI[i], abs_tol=tol)
            assert (pass_or_fail_SI == True)

    def mean_mol_weight_calc(self):
        X = 0.7
        Y_3_2 = 1e-10
        Y = 0.29
        Z_7_3 = 1e-7
        Z_7_4 = 1e-7
        Z_14_7 = 1e-11

        mean_mol_w = 1 / (2*X + Y_3_2 + (3/4)*Y + (4/7)*Z_7_3 + (5/7)*Z_7_4 + (8/14)*Z_14_7)

        return mean_mol_w

    def P(self, rho, T):
        c = 299792458  # m/s
        k = 1.380649e-23  # m^2 kg / s^2 K
        m_u = 1.6605e-27  # kg
        sigma = 5.670374419e-8  # stefan-boltzmann
        a = 4 * sigma / c  # radiation density constant

        P_g = (rho * k * T) / (self.mu * m_u)
        P_rad = (a / 3) * T ** 4

        P = P_g + P_rad

        return P

    def rho(self, P, T):
        c = 299792458  # m/s
        sigma = 5.670374419e-8  # stefan-boltzmann in SI
        a = 4 * sigma / c  # radiation density constant
        k = 1.380649e-23  # m^2 kg / s^2 K
        m_u = 1.6605e-27  # kg

        P_rad = (a / 3) * T ** 4
        P_g = P - P_rad
        rho = (P_g * self.mu * m_u) / (k * T)

        return rho

    def diff_solver(self, N, init_vals, p):

            def ODEs(m, vals):  # all initial values must be in SI units
                r, P, L, T = vals

                rho = self.rho(P, T)

                m_u = 1.6605e-27  # kg
                k = 1.380649e-23  # m^2 kg / s^2 K
                sigma = 5.670374419e-8  # stefan-boltzmann
                c_p = (k * 5) / (2 * self.mu * m_u)
                G = 6.6743e-11  # SI
                g = (G * m) / (r**2)

                kappa = self.kappa(T, rho)
                eps = self.epsilon(T, rho)

                #print(T, rho)
                #print(kappa)

                dr_dm = 1 / (4*np.pi*(r**2)*rho)
                dP_dm = - (G * m) / (4 * np.pi * (r ** 4))
                dL_dm = eps

                H_p = - P * (dr_dm / dP_dm)
                l_m = H_p  # since alpha_lm is assumed to be 1
                U = ((64 * sigma * (T ** 3)) / (3 * kappa * (rho ** 2) * c_p)) * np.sqrt(H_p / g)

                S_Qd = 2 / r  # surface area of the sun divided by the diameter and the perimeter of a slice of the sun
                K = (U / l_m) * S_Qd

                nabla_ad = 2 / 5  # for an ideal gas
                nabla_stable = (3 * kappa * rho * H_p * L) / (64*np.pi*sigma*(r**2)*(T**4))


                # using third order polynomial we calculated in Exercises 11-13

                roots = np.roots([1, (U / (l_m**2)), (U*K / (l_m**2)), (- U*(nabla_stable - nabla_ad) / (l_m**2))])
                real_root = [n for n in roots if n.imag == 0]
                xi = real_root[0].real


                # checking if stable or not
                if nabla_stable <= nabla_ad:  # convectively stable (only radiation)
                    dT_dm = - (3 * kappa * L) / (256 * (np.pi ** 2) * sigma * (r ** 4) * (T ** 3))
                    nabla_star = nabla_stable
                    F_con = 0
                else:                  # convectively unstable (convection is happening)
                    nabla_star = xi ** 2 + K * xi + nabla_ad
                    dT_dm = (nabla_star * T * dP_dm) / P
                    F_con = rho * c_p * T * np.sqrt(g) * (H_p ** (-3 / 2)) * (l_m / 2) ** 2 * xi ** 3

                F_rad = ((16 * sigma * T ** 4) / (3 * kappa * rho * H_p)) * nabla_star

                return dr_dm, dP_dm, dL_dm, dT_dm, F_con, F_rad, nabla_star, nabla_stable


            # initial values
            r_0 = init_vals[0]
            P_0 = self.P(init_vals[1], init_vals[3])
            L_0 = init_vals[2]
            T_0 = init_vals[3]
            M_sol = 1.989e30

            self.R_0 = r_0
            self.P_0 = P_0
            self.L_0 = L_0
            self.T_0 = T_0

            # creating arrays to save values in
            r_arr, P_arr, L_arr, T_arr, m_arr, F_c_arr, F_rad_arr, nabla_star_arr, nabla_stable_arr = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
            r_arr[0], P_arr[0], L_arr[0], T_arr[0], m_arr[0] = r_0, P_0, L_0, T_0, M_sol

            # array for f for dynamic step size
            f_arr = np.zeros((N, 4))
            f_arr[0] = np.array(list(ODEs(M_sol, [r_0, P_0, L_0, T_0])[:4]))

            #  array for V for dynamic step size
            V_arr = np.zeros((N, 4))
            V_arr[0] = np.array([r_0, P_0, L_0, T_0])

            self.p = p

            # calculatoing dm
            dm_arr = np.zeros(N)
            dm_list = p * (V_arr[0] / f_arr[0])  # calculating all the dms for each variable
            dm_arr[0] = min(abs(dm_list))  # choosing the smallest one

            # calc F_con and nablas
            F_c_arr[0], F_rad_arr[0], nabla_star_arr[0], nabla_stable_arr[0] = ODEs(M_sol, [r_0, P_0, L_0, T_0])[4:]


            for i in range(N-1):
                r_arr[i + 1] = r_arr[i] + (f_arr[i][0] * -dm_arr[i])
                P_arr[i + 1] = P_arr[i] + (f_arr[i][1] * -dm_arr[i])
                L_arr[i + 1] = L_arr[i] + (f_arr[i][2] * -dm_arr[i])
                T_arr[i + 1] = T_arr[i] + (f_arr[i][3] * -dm_arr[i])
                m_arr[i + 1] = m_arr[i] - dm_arr[i]     # m values

                '''print(i)
                print(f'dm = {dm_arr[i]}')
                print(f'm = {m_arr[i]}')
                print(f'm-dm = {m_arr[i] - dm_arr[i]}\n')
                #print(V_arr[i] / f_arr[i])'''

                if m_arr[i+1] < 0.01 * m_arr[0]:  # putting a stopper at 2.5% of M_0
                    break

                V_arr[i+1] = r_arr[i+1], P_arr[i+1], L_arr[i+1], T_arr[i+1]
                f_arr[i+1] = np.array(list(ODEs(m_arr[i+1], V_arr[i+1])[:4]))

                #print(L_arr[i + 1], f_arr[i + 1][2])

                dm_list = p * (V_arr[i+1] / f_arr[i+1])
                dm_arr[i+1] = np.min(abs(dm_list))

                F_c_arr[i+1], F_rad_arr[i+1], nabla_star_arr[i+1], nabla_stable_arr[i+1] = ODEs(m_arr[i+1], V_arr[i+1])[4:]

            return V_arr, m_arr, F_c_arr, F_rad_arr, nabla_star_arr, nabla_stable_arr

    def gradient_sanity_check(self, m, init_vals, p=0): # set p=1 to see the calculated values

        r, rho, L, T = init_vals

        m_u = 1.6605e-27  # kg
        k = 1.380649e-23  # m^2 kg / s^2 K
        sigma = 5.670374419e-8  # SI

        P = self.P(rho, T)

        c_p = (5 * k) / (2 * self.mu * m_u)
        G = 6.6743e-11  # SI
        g = (G * m) / (r ** 2)

        kappa = 3.98

        dr_dm = 1 / (4 * np.pi * (r ** 2) * rho)
        dP_dm = - (G * m) / (4 * np.pi * (r ** 4))

        H_p = - P * (dr_dm / dP_dm)  # pressure scale height
        l_m = H_p  # since alpha_lm is assumed to be 1

        nabla_ad = 2 / 5  # for an ideal gas
        nabla_stable = (3 * kappa * rho * H_p * L) / (64 * np.pi * sigma * (r ** 2) * (T ** 4))

        U = ((64 * sigma * (T ** 3)) / (3 * kappa * (rho ** 2) * c_p)) * np.sqrt(H_p / g)
        S_Qd = 2 / r   # surface area of the sun divided by the diameter and the perimeter of a slice of the sun
        K = (U / l_m) * S_Qd

        # using third order polynomial we calculated in Exercises 11-13
        roots = np.roots([1, (U / (l_m ** 2)), (- U * K / (l_m ** 2)), (- U * (nabla_stable - nabla_ad) / (l_m ** 2))])
        real_root = [n for n in roots if n.imag == 0]
        xi = real_root[0].real

        nabla = xi**2 + K*xi + nabla_ad

        v = np.sqrt(g / H_p) * (l_m / 2) * xi

        F_con = rho * c_p * T * np.sqrt(g) * (H_p**(-3/2)) * (l_m / 2)**2 * xi**3
        F_rad = (16*sigma*T**4 * nabla) / (3*kappa*rho*H_p)
        F_tot = F_rad + F_con

        nabla_p = nabla - xi**2

        assert nabla_ad < nabla_p < nabla < nabla_stable

        if p==1:
            print(f'mean_mol = {self.mu}')
            print(f'Kappa = {kappa}')
            print(f'H_p = {H_p}')
            print(f'Nabla_stable = {nabla_stable}')
            print(f'U = {U}')
            print(f'Xi = {xi}')
            print(f'Nabla* = {nabla}')
            print(f'v = {v}')
            print(f'F_con % = {F_con / F_tot}')
            print(f'F_rad % = {F_rad / F_tot}')
            print(nabla_ad, nabla_p, nabla, nabla_stable)



        print('*** Gradient sanity check passed ***')

    def plot(self, m, V, nablas, F):
        M_sol = 1.989e30
        R_sol = 6.96e8
        L_sol = 3.846e26
        rho_sol_ave = 1.408e3

        m_av = m / M_sol
        L_av = V[:, 2] / L_sol
        rho_av = self.rho(V[:, 1], V[:, 3]) / rho_sol_ave
        T = V[:, 3]
        R_av = V[:, 0] / R_sol
        P_av = self.P(rho_av, T)

        # plot for luminosity, mass, and temperature

        fig, axs = plt.subplots(3, 1)

        axs[0].plot(R_av, L_av)
        axs[0].invert_xaxis()
        axs[0].set_ylabel('Luminosity [J/kgs]')

        axs[1].plot(R_av, m_av)
        axs[1].invert_xaxis()
        axs[1].set_ylabel('Mass [kg]')

        axs[2].plot(R_av, T)
        axs[2].invert_xaxis()
        axs[2].set_ylabel('T [K]')

        for ax in axs.flat:
            ax.set(xlabel=''r'$R/R_{Sol}$')

        for ax in axs.flat:
            ax.label_outer()

        fig.suptitle(f'R_0={self.R_0:.2g}, P_0={self.P_0:.2g}, L_0={self.L_0:.2g}, T_0={self.T_0:.2g}', fontsize=10)

        plt.show()


        fig, axs = plt.subplots(2, 1)

        axs[0].plot(R_av, rho_av)
        axs[0].invert_xaxis()
        axs[0].set_yscale('log')
        axs[0].set_ylabel(''r'$log(\rho) [kg/m^3]$')

        axs[1].plot(R_av, P_av)
        axs[1].invert_xaxis()
        axs[1].set_yscale('log')
        axs[1].set_ylabel(''r'$log(P) [N/m^2]$')

        for ax in axs.flat:
            ax.set(xlabel=''r'$R/R_{Sol}$')

        for ax in axs.flat:
            ax.label_outer()

        fig.suptitle(f'R_0={self.R_0:.2g}, P_0={self.P_0:.2g}, L_0={self.L_0:.2g}, T_0={self.T_0:.2g}', fontsize=10)

        plt.show()

        nabla_ad = np.ones(len(P)) * (2/5)
        nabla_star = nablas[0]
        nabla_stable = nablas[1]

        plt.plot(R_av, nabla_ad, label=''r'$\nabla_{ad}$')
        plt.plot(R_av, nabla_star, label=''r'$\nabla^*$')
        plt.plot(R_av, nabla_stable, label=''r'$\nabla_{stable}$')
        plt.ylim(1e-2, 1e2)
        plt.gca().invert_xaxis()
        plt.yscale('log')
        plt.legend()
        plt.title(f'Temp. gradients (for p = {self.p}) R_0={self.R_0:.2g}, P_0={self.P_0:.2g}, L_0={self.L_0:.2g}, T_0={self.T_0:.2g}', fontsize=9)
        plt.xlabel(''r'$R/R_{Sol}$')
        plt.ylabel(''r'$log(\nabla) [K/m]$')
        plt.show()

        F_c = F[0]
        F_rad = F[1]
        F_tot = F_c + F_rad

        plt.plot(R_av, F_c/F_tot, label=''r'$F_{con}/F_{tot}$')
        plt.plot(R_av, F_rad/F_tot, label=''r'$F_{rad}/F_{tot}$')
        plt.gca().invert_xaxis()
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.title(f'Flux (for p = {self.p}) R_0={self.R_0:.2g}, P_0={self.P_0:.2g}, L_0={self.L_0:.2g}, T_0={self.T_0:.2g}', fontsize=9)
        plt.xlabel(''r'$R/R_{Sol}$')
        plt.ylabel(''r'$F [W/m^2]$')
        plt.show()

    def cross_section(self, R, L, F_C, show_every=50, sanity=False, savefig=False):
        """
        plot cross section of star
        :param R: radius, array
        :param L: luminosity, array
        :param F_C: convective flux, array
        :param show_every: plot every <show_every> steps
        :param sanity: boolean, True/False
        :param savefig: boolean, True/False
        """

        # R_sun = 6.96E8      # [m]
        R_sun = R[0]
        # L_sun = 3.846E26    # [W]
        L_sun = L[0]

        plt.figure(figsize=(800 / 100, 800 / 100))
        fig = plt.gcf()
        ax = plt.gca()

        r_range = 1.2 * R[0] / R_sun
        rmax = np.max(R)

        ax.set_xlim(-r_range, r_range)
        ax.set_ylim(-r_range, r_range)
        ax.set_aspect('equal')

        core_limit = 0.995 * L_sun

        j = 0
        for k in range(0, len(R) - 1):
            j += 1
            # plot every <show_every> steps
            if j % show_every == 0:
                if L[k] >= core_limit:  # outside core
                    if F_C[k] > 0.0:  # plot convection outside core
                        circle_red = plt.Circle((0, 0), R[k] / rmax, color='red', fill=True)
                        ax.add_artist(circle_red)
                    else:  # plot radiation outside core
                        circle_yellow = plt.Circle((0, 0), R[k] / rmax, color='yellow', fill=True)
                        ax.add_artist(circle_yellow)
                else:  # inside core
                    if F_C[k] > 0.0:  # plot convection inside core
                        circle_blue = plt.Circle((0, 0), R[k] / rmax, color='blue', fill=True)
                        ax.add_artist(circle_blue)
                    else:  # plot radiation inside core
                        circle_cyan = plt.Circle((0, 0), R[k] / rmax, color='cyan', fill=True)
                        ax.add_artist(circle_cyan)
        circle_white = plt.Circle((0, 0), R[-1] / rmax, color='white', fill=True)
        ax.add_artist(circle_white)

        # create legends
        circle_red = plt.Circle((2 * r_range, 2 * r_range), 0.1 * r_range, color='red', fill=True)
        circle_yellow = plt.Circle((2 * r_range, 2 * r_range), 0.1 * r_range, color='yellow', fill=True)
        circle_blue = plt.Circle((2 * r_range, 2 * r_range), 0.1 * r_range, color='blue', fill=True)
        circle_cyan = plt.Circle((2 * r_range, 2 * r_range), 0.1 * r_range, color='cyan', fill=True)

        ax.legend([circle_red, circle_yellow, circle_cyan, circle_blue], \
                  ['Convection outside core', 'Radiation outside core', 'Radiation inside core',
                   'Convection inside core'])
        plt.xlabel('$R$')
        plt.ylabel('$R$')
        plt.title(f'Cross section of star (R_0={self.R_0:.2g}, P_0={self.P_0:.2g}, L_0={self.L_0:.2g}, T_0={self.T_0:.2g})')
        plt.show()

        if savefig:
            if sanity:
                fig.savefig('Figures/sanity_cross_section.png', dpi=300)
            else:
                fig.savefig('Figures/final_cross_section.png', dpi=300)




if __name__ == "__main__":
    instance = ConvectionZone()
    instance.read_opacity()
    instance.read_epsilon()
    instance.opacity_sanity_check()
    #instance.epsilon_sanity_check()  # my epsilon check passes everything except the 2nd SI check, it comes back with half the wanted value


    # making a function to run the gradient santity check just to condense my code somewhat
    def running_grad_san_check(p):
        T = 0.9e6
        rho = 55.9
        R_sol = 6.96e8
        M_sol = 1.989e30
        L_sol = 3.846e26
        L = 1 * L_sol
        r = 0.84 * R_sol
        m = 0.99 * M_sol
        init_vals = [r, rho, L, T]
        instance.gradient_sanity_check(m, init_vals, p)
    running_grad_san_check(p=0)  # make p=1 to see the values i get

    R_sol = 6.96e8
    L_sol = 3.846e26
    T_0 = 5770
    rho_sol_ave = 1.408e3
    rho_0 = 1.42e-7 * rho_sol_ave

    init_vals = [R_sol, rho_0, L_sol, T_0]
    p = 0.1
    V, m, F_c, F_rad, nab_star, nab_stable = instance.diff_solver(3_000, init_vals, p)

    R, P, L, T = V[:, 0], V[:, 1], V[:, 2], V[:, 3]

    F = [F_c, F_rad]
    nablas = [nab_star, nab_stable]
    instance.plot(m, V, nablas, F)

    instance.cross_section(R, L, F_c)