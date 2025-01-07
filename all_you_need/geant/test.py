import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate

def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interpolate.interp1d(logx, logy, kind=kind, fill_value='extrapolate')
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

if "__main__" == __name__ :

    ee, nn = np.loadtxt("MuonFluxes/SN100pcGCR.txt", usecols=(0,1), unpack=True)
    func = log_interp1d(ee[:-1], nn[:-1]/np.diff(ee)*1e-4)

    energy0, flux0 = np.loadtxt("../Resconi.txt", usecols = (0,1), delimiter=' ',  unpack='true')
    func_resc = log_interp1d(energy0, flux0/(energy0**3))

    plt.figure()
    xx = np.logspace(0, 3, 200)
    plt.loglog(xx, func(xx), color="k", linestyle="--")
    plt.loglog(xx, func_resc(xx), color="darkred", linestyle="--")
    plt.savefig("test.png")

    name = ["20pc", "50pc", "100pc"]
    age_name = ["100yr", "300yr", "1kyr", "3kyr", "10kyr", "30kyr", "100kyr", "300kyr"]
    age = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
    color = ["steelblue", "darkred", "darkorange"]

    for j,ag in enumerate(age_name):

        plt.figure()
        for i,nn in enumerate(name):

            name_dir = "MuonFluxes/SN%s%s.txt" %(nn, ag)
            ee, ff = np.loadtxt(name_dir, usecols=(0,1), unpack=True)
            func = log_interp1d(ee[:-1], ff[:-1]/np.diff(ee)*1e-4)
            xx = np.logspace(0, 3, 200)
            plt.loglog(xx, func(xx), color=color[i], linestyle="--", label=nn)
        plt.loglog(xx, func_resc(xx), color="k", linestyle="--", label="Resconi")
        plt.legend()
        plt.savefig("plot_snfluxes/plot_sn_%.1fkyr.png" %age[j])

    

