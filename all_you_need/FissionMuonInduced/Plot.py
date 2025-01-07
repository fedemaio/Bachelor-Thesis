import numpy as np
import math

from numpy import exp
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz, quad
from scipy.special import erf

from WIMpy import DMUtils as DMU

from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator
import paleopy as paleopy

import os.path
from scipy.optimize import curve_fit
import scipy as sp
import scipy.interpolate
import random as rd

# -----------------------------------------------------------------------------------------------------------------------------------

def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value='extrapolate')
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def Asimm(A):
    A0 = 232
    A1 = 134
    A2 = 142
    sigma2 = 5.6
    sigma1 = sigma2/2
    cas = 0.5
    return np.exp(-(A-A2)**2/(2*sigma2**2))+np.exp(-(A-(A0-A2))**2/(2*sigma2**2))+cas*np.exp(-(A-A1)**2/(2*sigma1**2))+np.exp(-(A-(A0-A1))**2/(2*sigma1**2))

def Extr(n):
    s = np.zeros(n)
    B = Prob*n
    Arange = np.linspace(68, 165, 98)
    for i in range(n):
        a = rd.random()*n
        for j in range(len(B)):
            if (a<B[j]): 
                s[i] = Arange[j]
                break
    return(s)

def energySFF(A0,Z0,B0,A1,Z1,B1,A2,Z2,B2):

    mp = 938.3 #MeV
    mn = 939.6 #MeV

    M  = Z0*mp + (A0-Z0)*mn - B0*A0
    m1 = Z1*mp + (A1-Z1)*mn - B1*A1
    m2 = Z2*mp + (A2-Z2)*mn - B2*A2

    Ek1 = (M**2 + m1**2 - m2**2)/(2*M) - m1
    Ek2 = (M**2 + m2**2 - m1**2)/(2*M) - m2
    
    return Ek1,Ek2

def func(x, a, b, c, d, e):
    return a + b * x + c * x**2 + d * x**3 + e * x**4

def errortotal(N1,N2,f,C,sigmaf):
    rho = N1/N2
    den1 = N1**2
    den2 = N2**2
    num1 = 0.
    num2 = 0.
    for j in range(2):
        for k in range(2):
            num1 += (C[j][k]*sigmaf[0][j])**2/den1
            num2 += (C[j][k]*sigmaf[1][j])**2/den2
    #return rho*np.sqrt(1./N1+1./N2)
    return rho*np.sqrt(1./N1+1./N2+num1+num2)

def num(f,C,time,i):
    N = 0
    for j in range(2):
        for k in range(2):
            N += f[i][j]*C[j][k]*time
    return N

# -----------------------------------------------------------------------------------------------------------------------------------

# Plotting a distribution od decays from 232Th

print(' ---------------------------------------------------------------------------------------------- ')
print('')
print(' Plotting the A distribution ')

Arange = np.linspace(68, 165, 98)
Prob = []
Tot = 0
Prob.append(Asimm(68))
Tot = Asimm(68)
for i in range(97):
    Prob.append(Prob[i]+Asimm(Arange[i+1]))
    Tot += Asimm(Arange[i+1])
Prob = Prob/Tot
A232 = Extr(200000)
print(' Completed first extraction ')

Z0,A0 = np.loadtxt("U238.dat", usecols = (1,2), unpack='true')
A = []
for i in range(int(len(A0)/3)):
    A.append(A0[i*3+1])
    A.append(A0[i*3+2])
print(' Completed second extraction ')

print(' lunghezze: ',len(A232),len(A))

plt.figure(figsize=(12,8))
plt.hist(A232, 97, range = (68,165),color='darkgreen', label='$^{232}$Th', histtype='step')
plt.hist(A, 97, range = (68,165),color='darkblue', label='$^{238}$U', histtype='step')
plt.xlabel("Atomic mass", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.legend()
plt.savefig("./plot/FissionFrag.jpg",bbox_inches="tight", dpi=200)

# Plottting range in mineral

print(' Completed the first plot ')
print('')
print(' Plotting the range of muons inside the mineral ')

R = []
E = []
nuclei = ('Zr', 'Si', 'O')
fractions = [0.1667, 0.1667, 0.6667]

rho = 4.7 # g/cm^3

for i in range(len(nuclei)):
    dirname = './Range/'+nuclei[i]+'.txt'
    e, r = np.loadtxt(dirname, usecols = (0, 2), unpack = 'true')
    unit = np.loadtxt(dirname, usecols = (1), dtype = str, unpack = 'true')
    for k in range(len(unit)):
        if (unit[i] == 'GeV'): e[i] = e[i]*1000
    E.append(e)
    R.append(r)

Rtot = np.zeros(len(R[0]))
for i in range(len(R[0])):
    for k in range(len(nuclei)):
        Rtot[i] += R[k][i]*fractions[k]

Rtot = Rtot/rho
Energy = E[0]
func2 = log_interp1d(Energy, Rtot)
En = np.linspace(10, 1000, 100)
l1 = pow(10./rho,1./3.)
l2 = pow(100./rho,1./3.)

plt.figure(figsize=(12,8))
plt.plot(En, func2(En), color = 'steelblue', label = '$\\upmu$ range in zircon')
plt.axhline(l1, xmin=0, xmax=1, color='darkred', linestyle='--', label='Linear size for $m=10\,\\mathrm{g}$')
plt.axhline(l2, xmin=0, xmax=1, color='darkorange', linestyle='--', label='Linear size for $m=100\,\\mathrm{g}$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel("E [$\\mathrm{MeV}$]")
plt.ylabel("$\\upmu$ range [$\\mathrm{cm}$]")
plt.savefig('./plot/Muonrange.jpg', bbox_inches='tight', dpi=200)

print(' Completed the plot ')
print(' ')
print(' Plotting the dR/dx')

# plotting muon flux 

dirname = '../Resconi.txt'
e, flux = np.loadtxt(dirname, usecols = (0, 1), delimiter = ' ', unpack = 'true')
functionflux = log_interp1d(e, flux)
Energia = np.logspace(0,4,200)
plt.figure(figsize=(12,8))
plt.loglog(Energia, functionflux(Energia), color='k', linestyle='-.', label='$\\upmu$ flux at sea level')
plt.legend()
plt.xlabel('E [$\\mathrm{GeV}$]')
plt.ylabel('E$^3\\times\\frac{\\text{d}\\Phi}{\\text{dE}_\\upmu}$ [$\\mathrm{cm}^{-2}\,\\mathrm{sr}^{-1}\,\\mathrm{s}^{-1}\,\\mathrm{GeV}^{2}$]')
plt.savefig('./plot/Flux.png', bbox_inches='tight', dpi=200)

# plotting dR/dx

dirname = '../Resconi.txt'
e, flux = np.loadtxt(dirname, usecols = (0, 1), delimiter = ' ', unpack = 'true')
n = 20
E = np.zeros(n)
F = np.zeros(n)
for i in range(n):
    E[i] = e[i]*1e3 #MeV
    F[i] = flux[i]/((e[i]**3)*1e3) #cm^-2 sr^-1 s^-1 MeV^-1

popt, pcov = curve_fit(func, E, F)

En = np.linspace(10, 100, 9)
Ezr = 15
f = lambda x : func(x, *popt)
Number1 = quad(f, 0, Ezr)[0]*np.pi*1*3600*24*365*1e6 #flux in Myr^-1
Number100 = quad(f, 0, 100)[0]*np.pi*100*3600*24*365*1e6

count_U = np.loadtxt('U238histo.dat', usecols=(2), unpack = 'true')
count_Th = np.loadtxt('Th232histo.dat', usecols=(2), unpack='true')

n_bins = 100
x_min = 3 #log scale in nm
x_max = 5 #log scale in nm

lenght = np.logspace(x_min,x_max,n_bins)
lenght_width = np.diff(lenght)
lenght_mid   = lenght[:-1] + lenght_width/2

Na  = 6.022e23
rho_Zirc = 4.7
mass = 10**3*rho_Zirc*1e-3

# decay time for U and Th
tau_U = 6.45e3 #Myr 
tau_Th = 1.405e4/np.log(2) # Myr
# branching ratio of the spotaneous fission
BR_U = 5.4e-7
BR_Th = 1.1e-11
# fractions considered in the mineral (very high)
fraction_U = 0.001
fraction_Th = 0.001

# factors to evaluate the track rate 
# it should be noted that the spontaneous fission has to consider the avogadro number, since from the tracks per 1e5 decays we have to find the tracks per mass
# the induced fission has just the fraction since 0.14 and 0.02 are already the number of decays per kg of material
factor_U = 1e-5*(BR_U)*((fraction_U)*Na*1e3/(238))/tau_U
factor_Uind = 1e-5*(0.14*Number100)*(fraction_U)/mass*(np.exp(-1/tau_U))
factor_Th = 1e-5*(BR_Th)*((fraction_Th)*Na*1e3/(232))/tau_Th
factor_Thind = 1e-5*(0.02*Number100)*(fraction_Th)/mass*(np.exp(-1/tau_Th))
# 1e-5 perché conto il numero di tracce per fissione, 0.14*Number il numero di fissioni se fosse tutto U

plt.figure(figsize=(12,8))
plt.loglog(lenght_mid,count_U*factor_U/lenght_width, label='SF $^{238}\\text{U}$', color='steelblue')
plt.loglog(lenght_mid,count_Th*factor_Th/lenght_width, label='SF $^{232}\\text{Th}$', color='darkred')
plt.loglog(lenght_mid,count_U*factor_Uind/lenght_width, label='MIF $^{238}\\text{U}$', color='darkorange')
plt.loglog(lenght_mid,count_Th*factor_Thind/lenght_width, label='MIF $^{232}\\text{Th}$', color='darkgreen')
ax = plt.gca()

plt.ylabel("dR/dx [$\\mathrm{Myr}^{-1}\,\\mathrm{nm}^{-1}\,\\mathrm{kg}^{-1}$]")
plt.xlabel("x [nm]")
plt.legend()
plt.xlim(5e3,1e5)
plt.text(0.05, 0.88, "$f^{\\text{U}}=%.3f\,\\mathrm{g}/\\mathrm{g}$" %(fraction_U),fontsize=23.0, transform=ax.transAxes)
plt.text(0.05, 0.81, "$f^{\\text{Th}}=%.3f\,\\mathrm{g}/\\mathrm{g}$" %(fraction_Th),fontsize=23.0, transform=ax.transAxes)
plt.savefig("./plot/MuonFission.jpg",bbox_inches="tight", dpi=200)

print(' Plot completed')
print('')

mass = 10.0
Ez = {'10.0':30.0, '100.0':50.0}
f = lambda x : func(x, *popt)
l = pow(mass/rho_Zirc,1./3.)
Ezr = Ez[str(mass)]

# 100 g ---> 2.77 cm ---> 50 MeV
# 10 g  ---> 1.28 cm ---> 30 MeV

Number1 = quad(f, 0, Ezr)[0]*np.pi* l*l* 3600*24*365*1e6 #flux integrated in sr time and surface
mass = mass*1e-3
number = Number1

CU = (BR_U)*(Na*1e3/(238))/tau_U*mass # [Myr^-1]
CUind = (0.14*number) # [Myr^-1]
CTh = (BR_Th)*(Na*1e3/(232))/tau_Th*mass # [Myr^-1]
CThind = (0.02*number) # [Myr^-1]

f1U = 0.003
errorU = 0.001*1e-3
f1Th = 0.007
errorTh = 0.001*1e-3
f2U = 0.003
f2Th = np.linspace(0.007,0.08,100)

sigmaf = [[errorU,errorTh],[errorU,errorTh]]

time = 5000/1e6
timeind = 5000/1e6

print(' Plotting the muon signal inside the mineral considering ')
print(' m = ', mass*1e3, ' g')
print(' age = ',time*1e6,' yr')
print(' time of exposure = ',timeind*1e6,' yr')

#fig, axe = plt.subplots(figsize=(12,8))
plt.figure(figsize=(12,8))

y1 = []
y2 = []
sigma1up = []
sigma1down = []
sigma2up = []
sigma2down = []

C = [[CU*time,CUind*timeind],[CTh*time,CThind*timeind]]
f = [[f2U,f2Th[99]],[f1U,f1Th]]

N1 = num(f,C,1,0)
N2 = num(f,C,1,1)

#print(mass,errortotal(N1,N2,f,C,sigmaf)/(N1/N2),N1/N2-1.)

for i in range(len(f2Th)):
    C = [[CU*time,CUind*timeind],[CTh*time,CThind*timeind]]
    f = [[f2U,f2Th[i]],[f1U,f1Th]]
    N1 = num(f,C,1,0)
    N2 = num(f,C,1,1)
    y1.append(N1/N2)
    sigma1 = errortotal(N1,N2,f,C,sigmaf)
    sigma1up.append(N1/N2+sigma1)
    sigma1down.append(N1/N2-sigma1)
    
    C = [[CU*time,0.*CUind*timeind],[CTh*time,0.*CThind*timeind]]
    N1 = num(f,C,1,0)
    N2 = num(f,C,1,1)
    y2.append(N1/N2)
    sigma2 = errortotal(N1,N2,f,C,sigmaf)
    sigma2up.append(N1/N2+sigma2)
    sigma2down.append(N1/N2-sigma2)
    
plt.plot(f2Th,y1, label='Induced', color = 'darkred', linestyle = '-.')
plt.fill_between(f2Th,sigma1down,sigma1up, color = 'darkred', alpha = 0.25)
plt.plot(f2Th, y2, label='No induced', color = 'steelblue', linestyle = '-.')
plt.fill_between(f2Th,sigma2down,sigma2up, color = 'steelblue', alpha = 0.25)
#plt.axvline(x=0.059, ymin=0.0, ymax=1, color = 'black', label='Expected $f_2^\\text{Th}$')
plt.ylabel("$\\uprho$", fontsize=23.0)
plt.xlabel("$f_2^{\\text{Th}}$ $[\\mathrm{g}/\\mathrm{g}]$", fontsize=23.0)
#plt.ylim(0.9994,1.0027)
plt.legend(loc=2, fontsize = 23.0)

txt = "Time period = "+str(time*1e6)+" $\\mathrm{yr}$"
ax = plt.gca()
plt.text(0.37, 0.94, "$m=%i\,\\mathrm{g}$" %(mass*1e3),fontsize=23.0, transform=ax.transAxes)
plt.text(0.37, 0.88, "$f_1^{\\text{U}}=(%.3f\\pm0.001)\\times 10^{-3}\,\\mathrm{g}/\\mathrm{g}$" %(f1U*1e3),fontsize=23.0, transform=ax.transAxes)
#plt.text(0.37, 0.81, "$f_1^{\\text{Th}}=0\,\\mathrm{g}/\\mathrm{g}$" ,fontsize=23.0, transform=ax.transAxes)
plt.text(0.37, 0.81, "$f_2^{\\text{U}}=(%.3f\\pm0.001)\\times 10^{-3}\,\\mathrm{g}/\\mathrm{g}$" %(f2U*1e3),fontsize=23.0, transform=ax.transAxes)
plt.text(0.03, 0.77, "$t_{\\text{age}}=%i\,\\mathrm{kyr}$" %(time*1e3),fontsize=23.0, transform=ax.transAxes)
plt.text(0.03, 0.70, "$t_{\\text{exp}}=%i\,\\mathrm{kyr}$" %(timeind*1e3),fontsize=23.0, transform=ax.transAxes)
#plt.text(0.03, 0.70, "$t_{\\text{exp}}=0.2\,\\mathrm{kyr}$" ,fontsize=23.0, transform=ax.transAxes)

filename = 'Muonsignal%ig%ikyr%ikyr_NoInduced.png' % (mass*1e3,time*1e3,timeind*1e3)

plt.ylim(0.999,1.007)

plt.savefig("./plot/"+filename, bbox_inches="tight", transparent=True)
#plt.savefig("./plot/Prova.png",bbox_inches="tight")

print(' Plot completed :) ')
print('')
print(' ---------------------------------------------------------------------------------------------- ')


