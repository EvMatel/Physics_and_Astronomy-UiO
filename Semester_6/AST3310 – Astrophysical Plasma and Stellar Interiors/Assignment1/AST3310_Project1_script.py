import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class StellarCore:

    def __init__(self, T, rho):  # T = temperature, rho = density
        self.T = T*1e-9  # [10^9]
        self.rho = rho

        X = 0.7                                 # Hydrogen1-1
        self.mp = 1.67262192e-27                    # [kg]
        self.n_p = (rho * X) / self.mp          # [1/m^3]

        self.m_D = 3.344496969e-27              # [kg] Deuterium

        Y_3_2 = 10**-10                         # Helium3-2
        self.m_3_2 = 5.008237897e-27                      # [kg]
        self.n_3_2 = (rho * Y_3_2) / self.m_3_2      # [1/m^3]

        Y = 0.29                                # Helium4-2
        self.m_4_2 = 6.6464e-27                # [kg]
        self.n_4_2 = (rho * Y) / self.m_4_2          # [1/m^3]

        Z_7_3 = 10**-7                          # Lithium7-3
        self.m_7_3 = 1.1650341e-26                   # [kg]
        self.n_7_3 = (rho * Z_7_3) / self.m_7_3      # [1/m^3]

        self.m_7_4 = self.m_7_3                     # Berilium7-4
        self.n_7_4 = self.n_7_3

        self.m_8_4 = 1.332518301E-26                # [kg] Berilium8-4

        self.m_8_5 = self.m_8_4                     # [kg] Boron8-5

        self.m_12_6 = 1.992648239E-26               # [kg] Carbon12-6

        Z_14_7 = 10**-11                            # Nitrogen14-7
        self.m_14_7 = 2.32526673e-26                 # [kg]
        self.n_14_7 = (rho * Z_14_7) / self.m_14_7   # [1/m^3]

        self.m_15_8 = 2.491308462E-26               # [kg] Oxygen15-8

        self.m_e = 9.10938e-31                      # [kg] electron
        self.n_e = self.n_p + (2*self.n_3_2) + (2*self.n_4_2)


        #  reaction rates
        self.r_pp = self.r_ik(self.n_p, self.n_p, self.lam_ik(1))
        self.r_33 = self.r_ik(self.n_3_2, self.n_3_2, self.lam_ik(2))
        self.r_34 = self.r_ik(self.n_3_2, self.n_4_2, self.lam_ik(3))
        self.r_e7 = self.r_ik(self.n_e, self.n_7_4, self.lam_ik(4))
        self.r_17merk = self.r_ik(self.n_p, self.n_7_3, self.lam_ik(5))
        self.r_17 = self.r_ik(self.n_p, self.n_7_4, self.lam_ik(6))
        self.r_p14 = self.r_ik(self.n_p, self.n_14_7, self.lam_ik(7))

        # calculating normalisation factor
        self.norm_He3 = self.r_pp / (2*self.r_33 + self.r_34)

        # making sure He3 isn't being consumed more then is being produced
        if 2*self.r_33 + self.r_34 > self.r_pp:
            self.r_33 = self.r_33 * self.norm_He3
            self.r_34 = self.r_34 * self.norm_He3

        # calculating normalisation factor
        self.norm_Be7 = self.r_34 / (self.r_e7 + self.r_17)

        # making sure more Be isn't being consumed then is being produced
        if self.r_e7 + self.r_17 > self.r_34:
            self.r_e7 = self.r_e7 * self.norm_Be7
            self.r_17 = self.r_17 * self.norm_Be7

        # calculating normalisation factor
        self.norm_Li7 = self.r_e7 / self.r_17merk

        # making sure He4 isn't being consumed more then is being produced
        if self.r_17merk > self.r_e7:
            self.r_17merk = self.r_17merk * self.norm_Li7

    def lam_ik(self, key):  # key can be 1-7 based on Table 3.1

        T_merk = self.T / (1 + (self.T*4.95e-2))
        T_merk2 = self.T / (1 + 0.759*self.T)
        Na = 6.0221408e23  # [1/mol]

        lam_dict = {

            1: 4.01e-15 * (self.T**(-2/3)) * np.exp(-3.380 * (self.T**(-1/3)))
               * (1 + 0.123*(self.T**(1/3)) + 1.09*(self.T**(2/3)) + 0.938*self.T),

            2: 6.04e10 * (self.T**(-2/3)) * np.exp(-12.276 * (self.T**(-1/3))) * (1 + 0.034*(self.T**(1/3)) - 0.522*(self.T**(2/3))
                - 0.124*self.T + 0.353*(self.T**(4/3)) + 0.213*(self.T**(5/3)) ),

            3: 5.61e6 * (T_merk**(5/6)) * (self.T**(-3/2)) * np.exp(-12.826 * (T_merk**(-1/3))),
            4: 1.34e-10 * (self.T**(-1/2)) * ( 1 - 0.537*(self.T**(1/3)) + 3.86*(self.T**(2/3))
                + (0.0027 * (self.T**-1) * np.exp(2.515e-3 * (self.T**-1))) ),

            5: 1.096e9*(self.T**(-2/3))*np.exp(-8.472*(self.T**(-1/3)))
                - 4.83e8*(T_merk2**(5/6))*(self.T**(-3/2))*np.exp(-8.472*(T_merk2**(-1/3)))
                + 1.06e10*(self.T**(-3/2))*np.exp(-30.442*(self.T**-1)),

            6: 3.11e5*(self.T**(-2/3))*np.exp(-10.262*(self.T**(-1/3))) + 2.53e3*(self.T**(-3/2))*np.exp(-7.306*(self.T**-1)),
            7: 4.9e7*(self.T**(-2/3))*np.exp(-15.228*(self.T**(-1/3)) - 0.092*(self.T**2)) * ( 1 + 0.027*(self.T**(1/3)) - 0.778*(self.T**(2/3))
                - 0.149*self.T + 0.261*(self.T**(4/3)) + 0.127*(self.T**(5/3)) ) + 2.37e3*(self.T**(-3/2))*np.exp(-3.011*(self.T**-1))
                + 2.19e4*np.exp(-12.53*(self.T**-1))
        }


        # write the upper limit thing
        if self.T < 1e-3:
            lam_dict[4] = 1.57e-7 / self.n_e

        lam_ik = float(lam_dict[key] / (Na * 1e6))  # converting from cm to m and taking out moles

        return lam_ik

    def r_ik(self, n_i, n_k, lam_ik):
        # Takes number densities and lambda and returns the reaction rate

        if n_i == n_k:  # kronecker delta
            kr = 1
        else:
            kr = 0

        r_ik = ((n_i * n_k) * lam_ik) / (self.rho * (1 + kr))

        return r_ik

    def MeV_to_J(self, MeV):
        return MeV * 1.6022E-13     # [J]

    def E(self, m):  # takes mass in kg and returns E in J
        E = m*8.98e16  # [J]
        return E

    def PP0(self):
        Q_11 = self.MeV_to_J(1.177)    # [J]   step 1
        Q_21 = self.MeV_to_J(5.494)   # [J]  step 2
        Qtot = Q_11 + Q_21


        eps = Qtot * self.r_pp

        q_loss = self.MeV_to_J(0.265)

        return self.r_pp, Qtot, eps, q_loss

    def PPI(self):   # PP0 has a much higher reaction rate than PP1, and therefore, there will always be enough He_3_2
        Q_33 = self.MeV_to_J(12.860)  # [J]
        eps = Q_33 * self.r_33

        return self.r_33, Q_33, eps

    def PPIIandPPIII(self):
        Q_34 = self.MeV_to_J(1.586)  # [J]  step 1
        eps = Q_34 * self.r_34

        return self.r_34, Q_34, eps

    def PPII(self):
        Q_e7 = self.MeV_to_J(0.049)  # [J]  step 2
        Q_17merk = self.MeV_to_J(17.346)  # [J]  step 3
        Q = np.array([Q_e7, Q_17merk])

        r = np.array([self.r_e7, self.r_17merk])

        eps = Q*r
        eps_tot = np.sum(eps)

        q_loss = self.MeV_to_J(0.815)

        return r, Q, eps, eps_tot, q_loss

    def PPIII(self):
        Q_17 = self.MeV_to_J(0.137)  # [J]  step 2

        # Decay happens so much quicker that we can ignore the reaction rate
        Q_8b = self.MeV_to_J(8.367)  # [J]  step 3/ beta
        Q_8a = self.MeV_to_J(2.995)  # [J]  step 4 / alpha
        Q_decay = Q_8b + Q_8a

        Q = np.array([Q_17, Q_decay])

        eps = np.sum(Q) * self.r_17    # not taking the decay into account

        q_loss = self.MeV_to_J(6.711)

        return self.r_17, Q, eps, q_loss

    def CNO(self):
        Q = self.MeV_to_J(25.028)  # [J]  all steps combined
        eps = Q * self.r_p14

        q_loss = self.MeV_to_J(0.707 + 0.997)

        return self.r_p14, Q, eps, q_loss

    def SanityCheckSol(self):

        def tol(ans):   # tolerance of 2%
            return (ans / 100) * 2

        zero = self.PP0()
        one = self.PPI()
        twoandthree = self.PPIIandPPIII()
        two = self.PPII()
        three = self.PPIII()
        CNO = self.CNO()

        ans = 404
        eps_pp = zero[2]
        assert (ans - tol(ans) < eps_pp*self.rho < ans + tol(ans))

        ans = 8.68e-9
        eps_33 = one[2]
        assert (ans - tol(ans) < eps_33 * self.rho < ans + tol(ans))

        ans = 4.86e-5
        eps_34 = twoandthree[2]
        assert (ans - tol(ans) < eps_34 * self.rho < ans + tol(ans))

        ans = 1.49e-6
        eps_e7 = two[2][0]
        assert (ans - tol(ans) < eps_e7 * self.rho < ans + tol(ans))

        ans = 5.29e-4
        eps_17 = two[2][1]
        assert (ans - tol(ans) < eps_17 * self.rho < ans + tol(ans))

        ans = 1.63e-6
        r17 = three[0]
        Q = np.sum(three[1])
        assert (ans - tol(ans) < r17 * Q * self.rho < ans + tol(ans))

        ans = 9.18e-8
        epsCNO = CNO[2]
        assert (ans - tol(ans) < epsCNO * self.rho < ans + tol(ans))

        print('Sanity check #1 passed. \n')

    def SanityCheck2(self):

        def tol(ans):   # tolerance of 2%
            return (ans / 100) * 2

        zero = self.PP0()
        one = self.PPI()
        twoandthree = self.PPIIandPPIII()
        two = self.PPII()
        three = self.PPIII()
        CNO = self.CNO()

        ans = 7.34e4
        eps_pp = zero[2]
        assert (ans - tol(ans) < eps_pp*self.rho < ans + tol(ans))

        ans = 1.09
        eps_33 = one[2]
        assert (ans - tol(ans) < eps_33 * self.rho < ans + tol(ans))

        ans = 1.74e4
        eps_34 = twoandthree[2]
        assert (ans - tol(ans) < eps_34 * self.rho < ans + tol(ans))

        ans = 1.22e-3
        eps_e7 = two[2][0]
        assert (ans - tol(ans) < eps_e7 * self.rho < ans + tol(ans))

        ans = 4.35e-1
        eps_17 = two[2][1]
        assert (ans - tol(ans) < eps_17 * self.rho < ans + tol(ans))

        ans = 1.26e5
        r17 = three[0]
        Q = np.sum(three[1])
        assert (ans - tol(ans) < r17 * Q * self.rho < ans + tol(ans))

        ans = 3.45e4
        epsCNO = CNO[2]
        assert (ans - tol(ans) < epsCNO * self.rho < ans + tol(ans))

        print('Sanity check #2 passed. \n')

    def EnergyProd(self, pr=None):
        eps_pp = self.rho * self.PP0()[2]  # [J / m^3s]
        eps_33 = self.rho * self.PPI()[2]  # [J / m^3s]
        eps_34 = self.rho * self.PPIIandPPIII()[2]  # [J / m^3s]
        eps_e7 = self.rho * self.PPII()[2][0]  # [J / m^3s]
        eps_17merk = self.rho * self.PPII()[2][1]  # [J / m^3s]
        eps_17 = self.rho * self.PPIII()[2]  # [J / m^3s]


        eps_PP1 = (2*eps_pp + eps_33) * (self.r_33 / (2*self.r_pp))  # [J / m^3s]
        eps_PP2 = (eps_pp + eps_34 + eps_e7 + eps_17merk) * ((self.r_34 + self.r_e7 + self.r_17merk) / self.r_pp)  # [J / m^3s]
        eps_PP3 = (eps_pp + eps_34 + eps_17) * ((self.r_34 + self.r_17) / self.r_pp)  # [J / m^3s]
        eps_CNO = self.rho * self.CNO()[2]  # [J / m^3s]

        E_tot = eps_PP1 + eps_PP2 + eps_PP3 + eps_CNO

        eps_PP1_rel = eps_PP1 / E_tot
        eps_PP2_rel = eps_PP2 / E_tot
        eps_PP3_rel = eps_PP3 / E_tot
        eps_CNO_rel = eps_CNO / E_tot

        # total energy released due to loss of mass
        Q_tot_PP1 = 2*Sanity.PP0()[1] + Sanity.PPI()[1]  # [J]
        Q_tot_PP2andPP3 = Sanity.PPIIandPPIII()[1]  # [J]
        Q_tot_PP2 = np.sum(Sanity.PPII()[1]) + Q_tot_PP1 + Q_tot_PP2andPP3  # [J]
        Q_tot_PP3 = np.sum(Sanity.PPIII()[1]) + Q_tot_PP1 + Q_tot_PP2andPP3  # [J]
        Q_tot_CNO = Sanity.CNO()[1]  # [J]

        # energy loss due to neutrinos
        Q_loss_PP1 = 2 * Sanity.PP0()[3]  # [J]
        Q_loss_PP2 = Sanity.PPII()[4] + Q_loss_PP1  # [J]
        Q_loss_PP3 = Sanity.PPIII()[3] + Q_loss_PP1  # [J]
        Q_loss_CNO = Sanity.CNO()[3]  # [J]


        if pr is None:
            print(f'Energy Produced [J / m^3s] - PP1: {2*eps_pp + eps_33:.3g}, PP2: {eps_PP2:.3g}, PP3: {eps_PP3:.3g}, CNO: {eps_CNO:.3g}')
            print(f'Energy Loss [J / m^3s]- PP1: {Q_loss_PP1:.3g}, PP2: {Q_loss_PP2:.3g}, PP3: {Q_loss_PP3:.3g}, CNO: {Q_loss_CNO:.3g}')
            print(f'PP1: {(Q_loss_PP1 / Q_tot_PP1) * 100:.2f}%, PP2: {(Q_loss_PP2 / Q_tot_PP2) * 100:.2f}%, PP3: {(Q_loss_PP3 / Q_tot_PP3) * 100:.2f}%, CNO: {(Q_loss_CNO / Q_tot_CNO) * 100:.2f}%')


        return np.array([eps_PP1_rel, eps_PP2_rel, eps_PP3_rel, eps_CNO_rel])

    def GamowPeaks(self, m_i, m_k, Zi, Zk, E):  # Z = atomic number
        m = (m_i * m_k) / (m_i + m_k)
        T = self.T * 1e9  # descaling the temperature
        k_b = 1.380649e-23  # [m^2kg / s^2K]
        epsilon_0 = 8.8541878128e-12  # [F/m]
        h = 6.62607015e-34  # [m^2kg/s]
        e = 1.60217663e-19  # [C]

        exp1 = np.exp(- E / (k_b * T))
        exp2 = np.exp(-np.sqrt(m / (2 * E)) * ((Zi*Zk*(e**2)*np.pi) / (epsilon_0 * h)))

        Peak = exp1 * exp2

        return Peak

    def PlotGamowPeaks(self):
        E = np.linspace(1e-17, 1e-13, 100_000)

        gam_pp = self.GamowPeaks(self.mp, self.mp, 1, 1, E)
        gam_33 = self.GamowPeaks(self.m_3_2, self.m_3_2, 2, 2, E)
        gam_34 = self.GamowPeaks(self.m_3_2, self.m_4_2, 2, 2, E)
        gam_e7 = self.GamowPeaks(self.m_e, self.m_7_4, 4, 1, E)
        gam_17merk = self.GamowPeaks(self.mp, self.m_7_3, 1, 3, E)
        gam_17 = self.GamowPeaks(self.mp, self.m_7_4, 1, 4, E)
        gam_p14 = self.GamowPeaks(self.mp, self.m_14_7, 1, 7, E)

        gamow = [gam_pp, gam_33, gam_34, gam_e7, gam_17merk, gam_17, gam_p14]
        gamow = np.asarray(gamow)
        gamow_strings = ['pp', '33', '34', 'e7', '17\'', '17', 'p14']

        for i in range(len(gamow)):
            plt.plot(E, gamow[i] / np.linalg.norm(gamow[i]), label=f'{gamow_strings[i]}')


        plt.xlabel('Energy [J]')
        plt.ylabel('Relative Probability')
        plt.title('Gamow Peaks for each reaction')
        plt.xscale('log')
        plt.legend()
        plt.show()


if __name__ == "__main__":

    '''
    In order to turn the sanity check 'on or off' just delete the lines where I call them, line 380, and 388
    '''
    # Sanity check #1

    T_sol = 1.57e7  # [10^9 K]
    rho_sol = 1.62e5  # [kg/m^3]

    Sanity = StellarCore(T_sol, rho_sol)
    Sanity.SanityCheckSol()
    Sanity.EnergyProd()
    Sanity.PlotGamowPeaks()


    # Sanity check #2
    T = 1e8  # [K]
    San2 = StellarCore(T, rho_sol)
    San2.SanityCheck2()


    T_arr = np.linspace(1e4, 1e9, 1000)
    E = []
    for i in T_arr:
        E.append(StellarCore(i, rho_sol).EnergyProd(pr=0))

    E = np.asarray(E)


    plt.plot(T_arr, E[:, 0], label='PP1')
    plt.plot(T_arr, E[:, 1], label='PP2')
    plt.plot(T_arr, E[:, 2], label='PP3')
    plt.plot(T_arr, E[:, 3], label='CNO')
    #plt.xlim(1e7, 1e9)
    plt.xlabel('Temperature [K]')
    plt.ylabel('Relative Energy')
    plt.title('Relative Energy Produced from the PP chains and CNO cycle')
    plt.legend()
    plt.xscale('log')
    plt.show()