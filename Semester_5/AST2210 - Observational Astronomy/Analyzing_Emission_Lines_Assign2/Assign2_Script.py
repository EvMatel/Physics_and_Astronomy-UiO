import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# EXTRACTING DATA
def obs_values(file):
    infile = open(file, "r")
    velocity = []  # [K]
    flux_density = []  # [something]
    for line in infile:  # Looping over the lines in the file
        vel_in, flux_in = line.split()  # Splitting the line and putting the values in there one array
        velocity.append(float(vel_in))  # add to list
        flux_density.append(float(flux_in))

    return velocity, flux_density


velocity_spec, attenae_temp_spec = obs_values('/Users/evanmatel/Desktop/School/5.Semester/AST2210/Assignment2')  # N = 234

velocity = np.asarray(velocity_spec)
attenae_temp_spec = np.asarray(attenae_temp_spec)


# converting antennae_temp into flux density
K_to_Jy = 37.5  # [Jy/K] (with error of +-2,5, given in the lecture notes)
K_to_Jy_max = 40  # [Jy/K]
K_to_Jy_min = 35  # [Jy/K]
flux_density = attenae_temp_spec * K_to_Jy  # [Jy]
flux_density_max = attenae_temp_spec * K_to_Jy_max  # [Jy]
flux_density_min = attenae_temp_spec * K_to_Jy_min  # [Jy]


# fitting a gaussian curve to the data

def gauss_func(x, a, b, c, d):
    exp = np.exp(-((x - b) ** 2) / (2 * (c ** 2)))
    return (a * exp) + d

#the following values were given from GILDAS
a = 6.44815e-2  # amplitude of gaussian
b = 82.957  # mean of the gaussian
c = 332.968 / 2.3548  # standard deviation
d = 0.0  # y-value of the baseline
Values = [a, b, c, d]


popt, pcov = curve_fit(gauss_func, velocity, flux_density, Values)
popt_max, pcov = curve_fit(gauss_func, velocity, flux_density_max, Values)
popt_min, pcov = curve_fit(gauss_func, velocity, flux_density_min, Values)
gauss_val = gauss_func(velocity, *popt)
gauss_val_max = gauss_func(velocity, *popt_max)
gauss_val_min = gauss_func(velocity, *popt_min)

# oppgave 4
LFtot = 22.854 * K_to_Jy  # total line flux (area taken from gildas and converted to Jy)[Jy km/s]
error0 = 22.854 * 2.5  # 2.5 is the error from the K_to_Jy conversion


# oppgave 5
D_l = 139.4  # [Mpc]
ny_obs = 223.6  # GHz
z = 0.0308
x = 3.25e7 * (D_l**2 / (ny_obs**2 * (1 + z)**3))  # coefficiants
L_co = x * LFtot  # [K km pc^2/ s ]
print(f'The luminosity is {L_co:.4g} [K km pc^2/ s ]')
L_error = x * error0
print(f'The total luminosity error is +- {L_error:.3g} [K km pc^2/ s ], with an error procentage of {L_error / L_co:.3f}%.')


# oppgave 6
alpha_co = 1.7   # with error of 0.4 [M_sol / (K km pc^2/ s)]
M_H2He = L_co * alpha_co  # [Solar masses]
print(f'The mass is {M_H2He:.4g} solar masses')


# random useable coefficiants and variables

Sv_peak = np.max(gauss_val)  # flux at the peak of the gaussian [Jy]
FWHM = c * 2.3548  # [km/s]
ny_rest = 230.538
RMS_base = 1.03e-2 * K_to_Jy
S_to_N = Sv_peak / RMS_base  # ['sigmas']
print(f'The S to N ratio is  {S_to_N:.3g} sigma.')
ang_res_mb = 28  # [arcseconds] error room of +-2
ang_res_peak = ang_res_mb / S_to_N
print(f'The peak resolution is {ang_res_peak:.3g}" .')

# calculating the error

dalpha = L_co
e1 = 0.4
dlo = alpha_co * x
e2 = error0
M_error = np.sqrt((dalpha*e1)**2 + (dlo*e2)**2)
print(f'The total mass error is +- {M_error:.3g} solar masses, with an error procentage of {M_error / M_H2He:.3f}%.')



# plotting the data
plt.plot(velocity, flux_density)
plt.plot(velocity, gauss_val, '--', label='gauss')
plt.plot(velocity, gauss_val_max, '--', label='max gauss')
plt.plot(velocity, gauss_val_min, '--', label='min gauss')
plt.legend()
plt.xlabel('Velocity [km/s]')
plt.ylabel('Flux Density [Jy]')
plt.title('CO(2-1) Emission line')
plt.show()