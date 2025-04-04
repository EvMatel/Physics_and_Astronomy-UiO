import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

# Task e

def inflation(Y, t):
    psi, dpsi_dtau, a = Y
    psi_i = np.sqrt(1001/(4*np.pi))
    v = (3/(8*np.pi)) * (psi/psi_i)**2
    h = np.sqrt((8*np.pi/3) * (0.5*dpsi_dtau**2 + v))
    dv_dpsi = 2*v/psi
    dY_dtau = [dpsi_dtau, -3*h*dpsi_dtau - dv_dpsi, h]
    return dY_dtau


t_end = 2000
N = 10000
t = np.linspace(0, t_end, N)
y0 = [np.sqrt(1001/(4*np.pi)), 0, 0]

sol = sci.odeint(inflation, y0, t)
psi, dpsi_dtau, a = sol[:, 0], sol[:, 1], sol[:, 2]

# oppgave f

slow_psi_i = np.sqrt(1001/(4*np.pi)) # 11 / (2*np.sqrt(np.pi))
slow_roll = slow_psi_i - (t / (4*np.pi*slow_psi_i))

plt.plot(t, psi, 'b', label=r'$\psi$')
plt.plot(t, slow_roll, 'g', label='Slow')
plt.plot(t, dpsi_dtau, linestyle='--', color='orange', label=r'$\frac{\partial \psi}{\partial \tau}$')
plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.grid()
plt.title(r'Scalar Field of the $\psi^2$ Potential')
plt.savefig('task_e_1')
plt.show()

plt.plot(t, psi, 'b', label=r'$\psi$')
plt.plot(t, slow_roll, 'g', label='Slow')
plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.xlim(750, 1500)
plt.ylim(-2.5, 2.5)
plt.grid()
plt.title(r'Scalar Field of the $\psi^2$ Potential')
plt.savefig('task_e_3')
plt.show()

plt.plot(t, a, 'b', label=r'$ln(\frac{a}{a_i})$')
plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.grid()
plt.title(r'Scale Factor of the $\psi^2$ Potential')
plt.savefig('task_e_2')
plt.show()


# Task g

eps = 1 / (4*np.pi*(psi**2))
plt.plot(t, eps, 'b', label=r'$\epsilon$')
plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.yscale('log')
plt.ylim(10**-3, 1)
plt.xlim(0, 1000)
plt.grid()
plt.title(r'$\epsilon$ vs. $\tau$ of the $\psi^2$ Potential')
plt.savefig('task_g')
plt.show()


idx = int(np.argwhere(eps >= 1)[0])
N_t = a[idx]
print(f'The total number of e-folds during inflation is {N_t:.2f}.')

# Task i
psi_i = np.sqrt(1001/(4*np.pi))
v = (3/(8*np.pi)) * (sol[:, 0] / psi_i)**2
ratio = (0.5 * sol[:, 1]**2 - v) / (0.5 * sol[:, 1]**2 + v)
plt.plot(t, ratio, 'b', label=r'$p_{\phi} / \rho_{\phi}c^2$')
plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.title('Pressure to Density Ratio vs. Time')
plt.grid()
plt.savefig('task_i')
plt.show()


# Task j
eta = eps  # given this potential, both epsilon and eta are equal
N = N_t - a
plt.plot(N[:idx], eps[:idx], label=r'$\epsilon$')
plt.plot(N[:idx], eta[:idx], label=r'$\eta$')
plt.legend(loc='best')
plt.xlabel('N')
plt.title(r'Slow Roll Conditions During Inflation vs. $\tau$')
plt.grid()
plt.xlim(0, 150)
plt.gca().invert_xaxis()
plt.savefig('task_j')
plt.show()



# Task k
idx_begin = int(np.argwhere(N <= 60)[0])
idx_end = int(np.argwhere(N <= 50)[0])

eps_50_60 = eps[idx_begin:idx_end]
eta_50_60 = eta[idx_begin:idx_end]

r = 16*eps_50_60
n = 1 - 6*eps_50_60 + 2*eta_50_60

plt.plot(n, r)
plt.xlabel('n')
plt.ylabel('r')
plt.grid()
plt.title(r'n-r Plane for the $\psi^2$ PotentiaL')
plt.savefig('task_k')
plt.show()


# Task m

def Starobinsky(Y, t):
    psi, dpsi_dtau, a = Y
    X = -np.sqrt(16*np.pi / 3)
    v = (3 / (8*np.pi)) * ((1 - np.exp(X*psi)) / (1 - np.exp(2*X)))**2
    h = np.sqrt((8*np.pi/3) * (0.5*dpsi_dtau**2 + v))
    dv_dpsi = (3 / (4*np.pi*(1 - np.exp(2*X))**2)) * (1 - np.exp(X*psi)) * (-X*np.exp(X*psi))
    dY_dtau = [dpsi_dtau, -3*h*dpsi_dtau - dv_dpsi, h]
    return dY_dtau

t_end = 3500
N = 100_000
t = np.linspace(0, t_end, N)
y0 = [2, 0, 0]

star_sol = sci.odeint(Starobinsky, y0, t)
star_psi, star_dpsi_dtau, star_a = star_sol[:, 0], star_sol[:, 1], star_sol[:, 2]

plt.plot(t, star_psi, label=r'$\psi$')
plt.plot(t, star_dpsi_dtau, linestyle='--', label=r'$\frac{\partial \psi}{\partial \tau}$')
plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.grid()
plt.title('Starobinsky Scalar Field')
plt.savefig('task_m_1')
plt.show()

plt.plot(t, star_psi, label=r'$\psi$')
plt.plot(t, star_dpsi_dtau, linestyle='--', label=r'$\frac{\partial \psi}{\partial \tau}$')
plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.xlim(2650, 2800)
plt.ylim(-0.2, 0.2)
plt.grid()
plt.title(r'Starobinsky Scalar Field')
plt.savefig('task_m_3')
plt.show()


plt.plot(t, star_a, label=r'$ln(\frac{a}{a_i})$')
plt.legend(loc='best')
plt.xlabel(r'$\tau$')
plt.grid()
plt.title('Scale Factor of the Starobinsky Model')
plt.savefig('task_m_2')
plt.show()


# Task n

X = -np.sqrt((16*np.pi) / 3)*star_psi
star_eps = (4 * np.exp(2*X)) / (3*(1 - np.exp(X))**2)
star_eta = (4*(2*np.exp(2*X) - np.exp(X))) / (3*(1 - np.exp(X))**2)



star_idx = int(np.argwhere(star_eps >= 1)[0])
N_t = star_a[star_idx]
print(f'The total number of e-folds during inflation for the Starobinsky model is {N_t:.2f}.')

N = N_t - star_a


plt.plot(N[:star_idx], star_eps[:star_idx], label=r'$\epsilon$')
plt.plot(N[:star_idx], abs(star_eta[:star_idx]), label=r'|$\eta$|')
'''plt.xlim(1000, -1)
plt.ylim(-0.01, 0.15)'''
plt.xlabel('N')
plt.gca().invert_xaxis()
plt.xscale('log')
plt.title('Slow-Roll Conditions for Starobinsky Model')
plt.grid()
plt.legend()

plt.savefig('task_n_1')
plt.show()


star_idx_begin = int(np.argwhere(N <= 60)[0])
star_idx_end = int(np.argwhere(N <= 50)[0])

star_eps_50_60 = star_eps[star_idx_begin:star_idx_end]
star_eta_50_60 = star_eta[star_idx_begin:star_idx_end]

star_r = 16*star_eps_50_60
star_n = 1 - 6*star_eps_50_60 + 2*star_eta_50_60

plt.plot(star_n, star_r)
plt.legend(loc='best')
plt.xlabel('n')
plt.ylabel('r')
plt.title('n-r Plane of the Starobinsky Model')
plt.grid()
plt.savefig('task_n_2')
plt.show()


# Task p

#n_planck = 0.9649 +- 0.0042
# r_planck < 0.056


plt.plot(n, r)
plt.fill_betweenx(r, 0.9649 - 0.0042, 0.9649 + 0.0042, alpha=0.2, color="C2", label='Planck n-values')
plt.fill_between(star_n, 0, 0.056, alpha=0.2, color="C4", label='Planck r-values')
plt.xlabel('n')
plt.ylabel('r')
plt.grid()
'''plt.xlim(0.961, 0.9665)
plt.ylim(0.134, 0.160)'''
plt.title(r'n-r Plane of the $\psi^2$ Model')
plt.legend()
plt.savefig('task_p_1')
plt.show()


plt.plot(star_n, star_r)
plt.fill_betweenx(star_r, 0.9649 - 0.0042, 0.9649 + 0.0042, alpha=0.2, color="C2", label='Planck n-values')
plt.fill_between(star_n, 0, 0.056, alpha=0.2, color="C4", label='Planck r-values')
plt.xlabel('n')
plt.ylabel('r')
'''plt.xlim(0.961, 0.966)
plt.ylim(0.00335, 0.0043)'''
plt.title('n-r Plane of the Starobinsky Model')
plt.grid()
plt.legend()
plt.savefig('task_p_2')
plt.show()

