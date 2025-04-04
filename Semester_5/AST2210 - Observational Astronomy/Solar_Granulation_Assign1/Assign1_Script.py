import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.optimize import curve_fit

sns.color_palette("bright")
sns.set_theme()
sns.set_style("darkgrid")
plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "gist_heat"


# importing our data
idata = np.load("idata_square.npy")
spect_pos = np.load("spect_pos.npy")


# the class that takes the data and plots it
class SunData:

    def __init__(self, wav_idx=0):
        self.wav_idx = wav_idx  # wav_idx can be between 0 and 7


    def plot_intensity(self, Points, PointNames):  # plots the intensity data at a given wavelength
        wav_idx = self.wav_idx
        intensity_data = idata[:, :, wav_idx]
        fig, ax = plt.subplots()
        ax.grid(False)
        im = ax.imshow(intensity_data)
        for i in range(len(Points)):
            x = Points[i][0]
            y = Points[i][1]
            ax.plot(x, y, marker="o", label=f"{PointNames[i]}")
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel(r"Intensity")
        ax.set_title(r"$\lambda$ = " + f"{spect_pos[wav_idx]:.3f} [Å]")
        ax.set_xlabel("x [idx]")
        ax.set_ylabel("y [idx]")
        ax.legend()
        fig.tight_layout()

    def plot_subfield(self, x1, y1, h, w):  # lower left corner is at (x1, y1), top right corner is at (x2, y2).
        wav_idx = self.wav_idx
        fig, ax = plt.subplots()
        ax.grid(False)

        corner = (x1 - 1, y1 - 1)
        height = h
        width = w
        rect = Rectangle(corner, width, height, linewidth=2, edgecolor='yellow', facecolor='none')
        x1, y1 = rect.get_xy()
        x2 = x1 + rect.get_width()
        y2 = y1 + rect.get_height()
        slice_x = slice(x1, x2)
        slice_y = slice(y1, y2)
        idata_cut = idata[slice_y, slice_x, wav_idx]
        axis = (x1, x2, y1, y2)
        im = ax.imshow(idata_cut, extent=axis)
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel(r"Intensity")

        ax.set_title(r"Subfield at $\lambda$ = " + f"{wav_idx}")
        ax.set_xlabel("x [idx]")
        ax.set_ylabel("y [idx]")
        fig.tight_layout()
        plt.show()

    def plot_spectrum(self, Points, PointNames):  # plots the spectrum of a given point over 8 different wavelengths, and their average
        fig, ax = plt.subplots()
        PointsData = []
        for i in range(len(Points)):
            x = Points[i][0]
            y = Points[i][1]
            wavelength_spectrum = idata[y - 1, x - 1, :]
            PointsData.append(wavelength_spectrum)
            ax.plot(wavelength_spectrum, ls="--", lw=1, marker="x", label=f"{PointNames[i]}")
        PointsData = np.asarray(PointsData)

        Average = []
        for i in range(len(spect_pos)):
            wavelength_spectrum = idata[:, :, i]
            ave = np.mean(wavelength_spectrum)
            Average.append(ave)
        Average = np.asarray(Average)
        ax.plot(Average, ls="--", lw=1, marker="x", label='Average')

        # finding the gaussian curves

        def gauss_func(x, a, b, c, d):
            exp = np.exp(-((x - b) ** 2) / (2 * (c ** 2)))
            return (a * exp) + d


        ABCDA = []
        for i in range(len(Points)):
            ABCDA.append(PointsData[i])
        ABCDA.append(Average)


        # finding parameters to fit a gauss curve to each of the plots

        b_index_list = [1, 4, 4, 2, 3] # i had to brute force this and not generalize it
        x = np.linspace(0, 7, 8)
        N = len(x)
        PointNamesList = PointNames.tolist()
        PointNamesList.append('Average')
        LineColors = ['blue', 'orange', 'green', 'red', 'purple']

        for i in range(len(ABCDA)):
            d = np.max(ABCDA[i])
            a = np.min(ABCDA[i]) - d
            b = b_index_list[i]

            ABCDA[i] = np.asarray(ABCDA[i])

            sum = 0
            for j in range(N):
                sum += (j - b)**2
            c = np.sqrt(sum / N)


            Values = [a, b, c, d]
            popt, pcov = curve_fit(gauss_func, x, ABCDA[i], Values)

            #smoothing out of the gaussian
            smoothY_f = interp1d(x, gauss_func(x, *popt), kind="cubic")
            smoothX = np.linspace(x[0], x[-1], 500)
            smoothY = smoothY_f(smoothX)
            ax.plot(smoothX, smoothY, color=f'{LineColors[i]}', label=f" {PointNamesList[i]} Gauss fit ")

        ax.set_title("Spectral Lines of Points and Average of the Region")
        ax.set_ylabel("Intensity")
        ax.set_xlabel(r"$\lambda$ [idx]")
        ax.legend(prop={'size': 6})
        fig.tight_layout()
        plt.savefig('spectra_plot')
        plt.show()

    def doppler(self, Points, PointNames):
        lam = 6173  # Å
        c = 2.997e8  # m/s
        dlam = np.zeros((np.shape(idata[:, :, 0])))

        for i in range(len(dlam) - 1):
            for j in range(len(dlam[0]) - 1):
                min = np.min(idata[i, j, :])
                index, = np.where(idata[i, j, :] == min)
                dlam[i][j] = spect_pos[index] - lam


        v = dlam * c / lam  # m/s
        fig, ax = plt.subplots()
        ax.grid(False)
        im = ax.imshow(v)
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel(r"Velocity [m/s]")
        ax.set_title("Doppler Velocities of the Region")
        ax.set_ylabel("y [idx]")
        ax.set_xlabel('x [idx]')
        plt.savefig('Doppler')
        plt.show()
        for i in range(len(Points)):
            print(f'Point {PointNames[i]} at (x,y) = ({Points[i][0], Points[i][1]}) has a doppler velocity of '
                  f'{v[Points[i][1]][Points[i][0]]} [m/s].')

    def doppler_subfield(self, x1, y1, h, w):
        corner = (x1 - 1, y1 - 1)
        height = h
        width = w
        rect = Rectangle(corner, width, height, linewidth=2, edgecolor='yellow', facecolor='none')
        x1, y1 = rect.get_xy()
        x2 = x1 + rect.get_width()
        y2 = y1 + rect.get_height()

        lam = 6173  # Å
        c = 2.997e8  # m/s
        dlam = np.zeros((np.shape(idata[:, :, 0])))

        for i in range(len(dlam) - 1):
            for j in range(len(dlam[0]) - 1):
                min = np.min(idata[i, j, :])
                index, = np.where(idata[i, j, :] == min)
                dlam[i][j] = spect_pos[index] - lam


        v = dlam * c / lam  # m/s
        fig, ax = plt.subplots()
        ax.grid(False)
        im = ax.imshow(v)
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel(r"Velocity [m/s]")
        plt.xlim(x1, x2)
        plt.ylim(y1, y2)
        ax.set_title("Doppler Velocities of the Subregion")
        ax.set_ylabel("y [idx]")
        ax.set_xlabel('x [idx]')
        plt.savefig('Doppler_subfield')
        plt.show()






# interesting points [x, y] in the data for us to look closer at
A = [49, 197]
B = [238, 443]
C = [397, 213]
D = [466, 52]
PointCoor = np.array([A, B, C, D])
PointNames = np.array(['A', 'B', 'C', 'D'])



# plotting the intensity data on all spectrum
def plot_intens():
    for i in range(len(spect_pos)):
        Intens_plot = SunData(i)
        Intens_plot.plot_intensity(PointCoor, PointNames)
        plt.savefig(f'Intensity_lam{i}')
#plot_intens()




# plotting the spectrum of our random points
Spec = SunData()
#Spec.plot_spectrum(PointCoor, PointNames)



# Calculating the size of the photospheric PoV
def FoV_Size():

    Ylen_of_idata = np.shape(idata)[0]
    Xlen_of_idata = np.shape(idata)[1]

    ac_per_px = 0.058  # arcsec per pixel
    km_per_ac = 740  # kilometer per arcsecond

    Height_of_FoV = Ylen_of_idata * ac_per_px * km_per_ac
    Width_of_FoV = Xlen_of_idata * ac_per_px * km_per_ac
    print(f'The width of the photospheric FoV is {Width_of_FoV:.2f}km, the height is {Height_of_FoV:.2f}km, and the total area is {Width_of_FoV * Height_of_FoV:.2e}km^2.')
FoV_Size()

# plotting the sub-field we are interested in
def subfield():
    h = 100
    w = 150
    x1, y1 = 525, 325
    SubPlot = SunData()
    SubPlot.plot_subfield(x1, y1, h, w)



# plotting doppler velocities
Doppler = SunData()
Doppler.doppler(PointCoor, PointNames)


# plotting the doppler velocities of the subfield
def doppler_subfield():
    h = 100
    w = 150
    x1, y1 = 525, 325
    Doppler = SunData()
    Doppler.doppler_subfield(x1, y1, h, w)
#doppler_subfield()

print(spect_pos)