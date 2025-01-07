import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value='extrapolate')
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

if "__main__" == __name__ :

    ag = 0
    age_range = np.linspace(0,30,31)
    for ag in age_range:
        dir_name = "recoil_halite/SN20pc/Halite_muon_recoil_SN20pc_%ikyr.dat" %ag
        data_tmp = np.loadtxt(dir_name)
        data_tmp = data_tmp.T
        if ag == 0:
            data = data_tmp
        else:
            data += data_tmp
    
    drde = data[1]
    for i in range(2,15):
        drde += data[i]

    ee = np.loadtxt(dir_name, usecols=(0), unpack=True)

    plt.figure()
    func = log_interp1d(ee, drde)
    xx = np.logspace(np.log10(min(ee)), np.log10(max(ee)), 200)
    plt.loglog(xx, func(xx))
    plt.xlabel("$E/\\text{keV}$")
    plt.ylabel("$\\text{d}R/\\text{d}E / (\\text{keV}^{-1}\,\\text{Myr}^{-1}\,\\text{kg}^{-1})$")
    plt.savefig("plot_recoil_SN20pc.png")
