import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Oppgave 9

G = 6.6743e-11
k = np.sqrt(8 * np.pi * G)

def integrate_inv_plp_pot(N, init_values):

    x1, x2, x3, lam = init_values

    gamma = 2

    dx1_dN = (-3*x1) + (np.sqrt(6) / 2) * lam * x2**2 + (1/2) * x1 * (3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx2_dt = (-(np.sqrt(6) / 2) * lam * x1 * x2) + (1/2) * x2 * (3 + 3*x1**2 - 3*x2**2 + x3**2)
    dx3_dt = -2*x3 + ((1/2) * x3 * (3 + 3*x1**2 - 3*x2**2 + x3**2))
    dlam_dt = -np.sqrt(6) * lam**2 * (gamma - 1) * x1

    return dx1_dN, dx2_dt, dx3_dt, dlam_dt

def integrate_exp_pot(N, init_values):

    x1, x2, x3, lam= init_values

    dx1_dN = (-3 * x1) + (np.sqrt(6) / 2) * lam * x2 ** 2 + (1 / 2) * x1 * (3 + 3 * x1 ** 2 - 3 * x2 ** 2 + x3 ** 2)
    dx2_dN = (-(np.sqrt(6) / 2) * lam * x1 * x2) + (1 / 2) * x2 * (3 + 3 * x1 ** 2 - 3 * x2 ** 2 + x3 ** 2)
    dx3_dN = -2 * x3 + ((1 / 2) * x3 * (3 + 3 * x1 ** 2 - 3 * x2 ** 2 + x3 ** 2))

    dlam_dN = 0  # since gamma=1

    return dx1_dN, dx2_dN, dx3_dN, dlam_dN


power_law_init_values = [5e-5, 1e-8, 0.9999, 1e9]
exp_init_values = [0.01, 5e-13, 0.9999, 3/2]

def N_to_Z(N):   # converting N to redshift z
    return (1 / np.exp(N)) - 1

def Z_to_N(z):
    return np.log(1 / (1+z))

m = 10_000
N_start = Z_to_N(2e7)
N_fin = 0
N = np.linspace(N_start, N_fin, m)
z = N_to_Z(N)

exp_sol = solve_ivp(integrate_exp_pot, y0=exp_init_values, t_span=[N_start, N_fin], t_eval=N)
power_sol = solve_ivp(integrate_inv_plp_pot, y0=power_law_init_values, t_span=[N_start, N_fin], t_eval=N)


def solve_omega(sol):
    x1, x2, x3, lam = sol.y
    omega_phi = (x1**2 + x2**2)
    omega_m = (x3**2)
    omega_r = (1 - x1**2 - x2**2 - x3**2)

    w_phi = (x1**2 - x2**2) / (x1**2 + x2**2)

    return omega_phi, omega_r, omega_m, w_phi

def plot_oppgave9(sol, title):
    omega_phi, omega_r, omega_m, w_phi = solve_omega(sol)

    plt.plot(z, omega_phi, label='phi')
    plt.plot(z, omega_r, label='r')
    plt.plot(z, omega_m, label='m')
    plt.legend()
    plt.xlabel('Redshift [z]')
    plt.ylabel('Omega []')
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.title(f'Omega for the {title} method')
    plt.show()

    plt.plot(z, w_phi, label='w_phi')
    plt.legend()
    plt.xlabel('Redshift [z]')
    plt.ylabel('Omega []')
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.title(f'EoS equation for {title}')
    plt.show()


plot_oppgave9(exp_sol, 'exponential')
plot_oppgave9(power_sol, 'inverse power law')


# Oppgave 10
def calc_Hubble(omega_m0, omega_r0, omega_phi, z):
    m_part = omega_m0 * (1+z)**3
    r_part = omega_r0 * (1+z)**4
    return np.sqrt(m_part + r_part + omega_phi)


def calc_Hubble_ACDM(z):
    omega_m = 0.3
    return np.sqrt(omega_m * (1+z)**3 + (1 - omega_m))


omega_phi, omega_r, omega_m_exp, w_phi = solve_omega(exp_sol)
Hubble_exp = calc_Hubble(omega_m_exp[-1], omega_r[-1], omega_phi, z)

omega_phi, omega_r, omega_m_power, w_phi = solve_omega(power_sol)
Hubble_power = calc_Hubble(omega_m_power[-1], omega_r[-1], omega_phi, z)

Hubble_ACDM = calc_Hubble_ACDM(z)

plt.plot(z, Hubble_exp, label='Exp')
plt.plot(z, Hubble_power, label='Power Law')
plt.plot(z, Hubble_ACDM, label='ACDM')
plt.xlabel('Redshift [z]')
plt.ylabel('Omega []')
plt.xscale('log')
plt.title('Hubble parameters')
plt.legend()
plt.gca().invert_xaxis()
plt.show()



# Problem 11

def calc_time_uni(omega_m):
    return (2 / (3 * np.sqrt(1 - omega_m))) * np.sinh((1 - omega_m) / omega_m)


print(calc_time_uni(omega_m_exp[-1]))
print(calc_time_uni(omega_m_power[-1]))