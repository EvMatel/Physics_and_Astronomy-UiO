import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from math import isnan


# calculates the standard deviation
def sigma(data):
    mean = np.mean(data)
    sum = np.sum((data - mean)**2)
    N = len(data)
    std = np.sqrt(sum / N)

    return std

def RMS(filename):
    hdu = fits.open(filename)
    data = hdu[1].data
    sigmas = []
    y = np.arange(317)
    x = np.arange(320)
    for i in y:
        for j in x:
            sigmas.append(sigma(data[1623:2300, i, j]))
    sigmas2 = []
    for i in sigmas:
        if not isnan(i):
            sigmas2.append(i)
    RMS = np.mean(sigmas2)
    print(RMS)
    RMS *= 10  # multiplying by 10 because i believe that to be correct
    print(RMS)
    return RMS


# reads entire data cube
def readfits(filename):
    hdu = fits.open(filename)  #reads the file
    #hdu.info()  # prints general info about the file, i.e. number of extensions and their dimensions.
    data = hdu[1].data
    hdr = hdu[1].header

    return data, hdr


# reads spatially collapsed spectra
def readPointfits(filename):
    hdu = fits.open(filename)  #reads the file
    #hdu.info()  # prints general info about the file, i.e. number of extensions and their dimensions.
    data = hdu[0].data
    hdr = hdu[0].header

    return data, hdr


# reads and plots spectrally collapsed pictures
def readPic(filename, output_name=0, title=0):
    hdu = fits.open(filename)  # reads the file
    data = hdu[0].data
    plt.contourf(np.log(data), 100, cmap='gist_heat')
    cbar = plt.colorbar()
    cbar.set_label('Flux Density [10^(-20)*erg/s/cm^2/Å]')
    x1 = [32, 21.3, 10.6, 0, -10.6, -21.3, -32]
    xi = [0, 53, 106, 159, 212, 265, 320]
    y1 = [-31.7, -21.3, -10.6, 0, 10.6, 21.3, 31.7]
    yi = [0, 53, 106, 159, 212, 265, 317]
    plt.xticks(xi, x1)
    plt.yticks(yi, y1)
    plt.xlabel('RA [Arcseconds]')
    plt.ylabel('Declination [Arcseconds]')
    if title == 0:
        plt.title('Total Integrated Map of NGC 1365')
    else:
        plt.title(f'{title}')
    if output_name != 0:
        plt.savefig(output_name)
    plt.show()


# reads and plots spectrally collapsed pictures, and marks certain points
def readPic_withpoints(filename, x, y, output_name=0):
    hdu = fits.open(filename)  # reads the file
    data = hdu[0].data
    plt.contourf(np.log(data), 100, cmap='gist_heat')
    cbar = plt.colorbar()
    cbar.set_label('Flux Density [10^(-20)*erg/s/cm^2/Å]')
    plt.scatter(x, y, s=10, marker='x', c='black')
    txt = ['A', 'B', 'C']
    for i in range(len(x)):
        plt.annotate(txt[i], (x[i], y[i]))
    x1 = [32, 21.3, 10.6, 0, -10.6, -21.3, -32]
    xi = [0, 53, 106, 159, 212, 265, 320]
    y1 = [-31.7, -21.3, -10.6, 0, 10.6, 21.3, 31.7]
    yi = [0, 53, 106, 159, 212, 265, 317]
    plt.xticks(xi, x1)
    plt.yticks(yi, y1)
    plt.xlabel('RA [Arcseconds]')
    plt.ylabel('Declination [Arcseconds]')
    plt.title('Total Integrated Map of NGC 1365')
    if output_name != 0:
        plt.savefig(output_name)
    plt.show()


# oppgave 5
def readPicWithSigma(filename, title, RMS, output_name=0):
    hdu = fits.open(filename)
    data = hdu[0].data


    levels = [3*RMS, 10*RMS, 20*RMS]

    fig, ax = plt.subplots()
    cs1 = ax.contourf(np.log(data), 100, cmap='gist_heat')
    cs2 = ax.contour(data, cmap='gist_heat', levels=levels)
    fmt = {}
    strs = ['3\u03C3', '10\u03C3', '20\u03C3']
    for l, s in zip(levels, strs):
        fmt[l] = s
    ax.clabel(cs2, cs2.levels, inline=True, fmt=fmt, fontsize=10)
    cbar = fig.colorbar(cs1)
    cbar.set_label('Flux Density [10^(-20)*erg/s/cm^2/Å]')
    x = [32, 21.3, 10.6, 0, -10.6, -21.3, -32]
    xi = [0, 53, 106, 159, 212, 265, 320]
    y = [-31.7, -21.3, -10.6, 0, 10.6, 21.3, 31.7]
    yi = [0, 53, 106, 159, 212, 265, 317]
    plt.xticks(xi, x)
    plt.yticks(yi, y)
    plt.xlabel('RA [Arcseconds]')
    plt.ylabel('Declination [Arcseconds]')
    plt.title(f'{title} with \u03C3 = {rms:.3f}')
    if output_name != 0:
        plt.savefig(output_name)
    plt.show()


# reads and plots spatially collapsed spectra
def plotSpectrum(filename, title, xlim=0, ylim=0, x=0, output_name=0):

    # loading data
    if x != 0:
        data, hdr = readfits(filename)  # reading full data cube
    else:
        data, hdr = readPointfits(filename)  # reading spacially collapsed spoectra

    # treating data

    data = data[:-1]  # removing the last element


    # plotting data
    plt.figure(figsize=(10, 5))
    Spectrum = np.linspace(4750.0795898438, 9351.3295898438, len(data))  # the spectral range of MUSE

    plt.plot(Spectrum, data)

    if xlim != 0:
        plt.xlim(xlim)
    if ylim != 0:
        plt.ylim(ylim)
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Flux Density [10^(-20)*erg/s/cm^2/Å]')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(title)
    if output_name != 0:
        plt.savefig(output_name)
    plt.show()


def subplot(y1, y2, y3, title, output_name=0):
    x = np.linspace(4750.0795898438, 9351.3295898438, len(y1))

    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle('Sharing both axes')
    axs[0].plot(x, y1)
    axs[1].plot(x, y2)
    axs[2].plot(x, y3)

    for ax in axs.flat:
        ax.set(xlabel='Wavelength [Å]', ylabel='Flux Density [10^(-20)*erg/s/cm^2/Å]')
    for ax in axs.flat:
        ax.label_outer()


    if output_name != 0:
        plt.savefig(output_name)
    plt.show()


def redshift(wavelengths):
    wavelengths2 = np.asarray(wavelengths)
    constant = [4887.9, 4986.05, 5034.2, 6335.0, 6583.4, 6598.7, 6619.5, 6753.8, 6768.2]
    constantARRAY = np.asarray(constant)
    for i in range(len(wavelengths2)):
        ans = constantARRAY / wavelengths2
    REALans = np.mean(ans)
    REALans = REALans - 1
    z = 0.005476
    print(f'{REALans + z:.7f}, {REALans:.7f}')


def redshift2(wavelengths):
    wavelengths2 = np.asarray(wavelengths)
    constant = [4887.9, 5034.2, 6335.0, 6583.4, 6598.7, 6619.5, 6753.8, 6768.2]
    constantARRAY = np.asarray(constant)
    for i in range(len(wavelengths2)):
        ans = wavelengths2 / constantARRAY
    REALans = np.mean(ans)
    REALans = 1 - REALans
    z = 0.005476
    print(f'{REALans + z:.7f}, {REALans:.7f}')


# main #






#plotSpectrum('AGN_ap4.fits', 'Point A', output_name='PointA')
#plotSpectrum('O3_ap5.fits', 'Point B', output_name='PointB')
#plotSpectrum('pointC_west.fits', 'Point C', output_name='PointC')

#readPic('OIII_N2.fits', output_name='OIII', title='Isolated [OIII] Line')
#readPic('Halpha.fits', output_name='Halpha', title='Isolated H\u03B1 Line')

#rms = RMS('ADP.2017-03-27T12_08_50.541.fits')
#readPicWithSigma('OIII_N2.fits', 'OIII', rms, output_name='OIII_sigma')
#readPicWithSigma('Halpha.fits', 'H\u03B1', rms, output_name='Halpha_sigma')

#readPic('integrated_over_spectra.fits', 'integrated')
#readPic_withpoints('integrated_over_spectra.fits', [166, 109, 261], [156, 156, 153], output_name='integrated_with_points')




wavesA = [4887.54, 4985.64, 5033.83, 6334.09, 6584.2, 6597.7, 6618.5, 6752.7, 6767.5]
wavesB = [4885.1, 4983.9, 5031.4, 6334.09, 6581.6, 6595.0, 6616.2, 6750.2, 6765.2]
wavesC = [4887.2, 5035.08, 6335.6, 6584.9, 6599.17, 6620.3, 6753.85, 6768.4]

redshift(wavesA)
redshift(wavesB)
redshift2(wavesC)

