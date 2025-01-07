from cProfile import label
import numpy as np #matematica
import math

from scipy.interpolate import interp1d,InterpolatedUnivariateSpline #interpolazioni
from scipy.optimize import curve_fit #fit
from scipy.integrate import cumtrapz, quad, romberg, quadrature #integrazione
from matplotlib import pyplot as plt #grafici
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors

import matplotlib as mpl 
from matplotlib import rc

import os.path
import pandas as pd #file e dati
import csv
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt

import sys #file e dati

namedir = 'Halite_QGSP' #trova la directory della cartella con questo nome
# namedir = 'Halite_final'

#ho ristampato i valori prima
EnergyName = ["1.166000", "1.555000", "2.074000", "2.766000", "3.689000", "4.920000", "6.560000", "8.740000", "11.190000", "13.880000", "17.210000", "21.330000",
    "26.440000", "32.780000", "40.640000", "50.120000", "61.530000", "75.540000", "92.740000", "113.860000", "139.780000", "171.610000", "210.600000", "258.600000",
    "317.500000", "392.700000", "488.600000", "607.900000", "756.300000", "941.000000", "1170.700000", "1456.500000",
    "1812.200000", "2254.600000"]

length = 1000 #cm
rho = 2.16*1e-3 #kg/cm^3

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

#interpolazione logaritmica di dati
def log_interp1d(xx, yy, kind='linear'): #xx e yy sono due array, il kind inidica un'interpolazione lineare
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value='extrapolate') #fill_value='extrapolate' permette di estrapolate oltre i limiti dei dati originali
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz))) #riporta in scala lineare
    return log_interp

# ritorna histogram * weight
#conta i dati e li separa in base all'energia e poi moltiplica per un peso
def DataCounter(Er, data, weight): #Er=array di limiti degli intervalli energetici, data=valori dei dati da contare
    Count = np.zeros(len(Er)-1) #crea array di zeri
    for i,e in enumerate(data):
        index = np.digitize(data[i],Er) #trova in quale intervallo ricade data[i]
        Count[index] += 1.*weight #conta con il peso
    return Count

#separa i conteggi per diversi tipi di nuclei (frammenti) (come sopra)
def TotalCounter(Er, data, weight, namefrag, frag): #namefrag= nome dei frammenti, frag=lista dei frammenti da considerare
    Count = np.zeros((len(frag),len(Er)-1)) #Crea un array Count bidimensionale di zeri (prima dim=numero di tipi di frammenti, seconda dim=numeri di intervalli energetici)
    for i,e in enumerate(data):
        name = namefrag[i] #estrae il nome del frammento e toglie numeri, parentesi e punti
        name = name.replace('[','')
        name = name.replace(']','')
        name = name.replace('0','')
        name = name.replace('1','')
        name = name.replace('2','')
        name = name.replace('3','')
        name = name.replace('4','')
        name = name.replace('5','')
        name = name.replace('6','')
        name = name.replace('7','')
        name = name.replace('8','')
        name = name.replace('9','')
        name = name.replace('.','')
        if name != 'He' and name != 'alpha' and name != 'proton': #eclude He, alpha e proton
            for j in range(len(frag)):
                if name == frag[j]: #cerca corrispondenza con i frammenti in frag
                    index = np.digitize(data[i],Er) #becca a quale intervallo corrisponde
                    Count[j][index] += 1.*weight #aumenta il count pesandolo
    return Count

#legge dati da file e conta gli eventi per ogni nucleo per ogni energia
def Count(weight,Er): 
    nuclei = []
    for j in range(34): #perchè in range 34?
        name = np.loadtxt(namedir + "/Nuclei/outNuclei_"+EnergyName[j]+".txt", usecols = 0, dtype=str) 
        for i in range(len(name)): #legge la prima colonna da un file e salva in un array togliendo parentesi, numeri e punti alla stringa che legge
            name[i] = name[i].replace("[","")
            name[i] = name[i].replace("]","")
            name[i] = name[i].replace("1","")
            name[i] = name[i].replace("2","")
            name[i] = name[i].replace("3","")
            name[i] = name[i].replace("4","")
            name[i] = name[i].replace("5","")
            name[i] = name[i].replace("6","")
            name[i] = name[i].replace("7","")
            name[i] = name[i].replace("8","")
            name[i] = name[i].replace("9","")
            name[i] = name[i].replace("0","")
            name[i] = name[i].replace(".","")
            if name[i] not in nuclei:
                nuclei.append(name[i]) #se quello che legge nella stringa non è in nuclei lo aggiunge
    countNuclei = np.zeros((len(nuclei),len(Er)-1)) #crea array bidimensionale come sopra
    countCl35 = np.zeros(len(Er)-1) #crea array specifici per gli atomi dell'alite
    countCl37 = np.zeros(len(Er)-1)
    countNa23 = np.zeros(len(Er)-1)
    for i in range(34): #per ogni tipo di nucleo legge i dati (colonna 2) e usa DataCounter per contare i dati dei singoli nuclei
        data = np.loadtxt(namedir + "/Cl35/outCl35_"+EnergyName[i]+".txt", usecols = 2)
        countCl35 += DataCounter(Er, data, weight[i])
        data = np.loadtxt(namedir + "/Cl37/outCl37_"+EnergyName[i]+".txt", usecols = 2)
        countCl37 += DataCounter(Er, data, weight[i])
        data = np.loadtxt(namedir + "/Na23/outNa23_"+EnergyName[i]+".txt", usecols = 2)
        countNa23 += DataCounter(Er, data, weight[i])
        data = np.loadtxt(namedir + "/Nuclei/outNuclei_"+EnergyName[i]+".txt", usecols = 2)
        tag = np.loadtxt(namedir + "/Nuclei/outNuclei_"+EnergyName[i]+".txt", usecols = 0, dtype=str)
        countNuclei += TotalCounter(Er, data, weight[i], tag, nuclei)
    return [countCl35, countCl37, countNa23, countNuclei, nuclei] #restituisce gli array dei vari nuclei con i rispettivi conteggi
        
#crea un'immagine con il grafico (scala log) dei vari conteggi
def PlotCount(Er, Count, name): 
    plt.figure(figsize=(12,8))
    
    Er_width = np.diff(Er) #differenza tra un valore dell'array e quella precedente = larghezza del bin
    Er_mid = Er[:-1] + Er_width/2 #punto medio del bin

    Frag = np.zeros(len(Er_mid)) #crea array con tutti i punti medi dei bin di energia
    for k in range (len(Count[4])): #Count[4]=lunghezza di nuclei
        for i in range(len(Er_mid)):
            Frag[i] += Count[3][k][i] #somma i bin dei vari nuclei

    F35 = interp1d(Er_mid, Count[0]/Er_width, fill_value='extrapolate') #interpola in modo log i frammenti dei nuclei di alite
    F37 = interp1d(Er_mid, Count[1]/Er_width, fill_value='extrapolate')
    F23 = interp1d(Er_mid, Count[2]/Er_width, fill_value='extrapolate')

    plt.loglog(Er_mid*1e3, F35(Er_mid)*1e-3/(length*rho), label = "$^{35}$Cl", linewidth = 2, color='steelblue') #1000*punto medio del bin, count di quel frammento in quel bin di E (dR/dE)
    plt.loglog(Er_mid*1e3, F37(Er_mid)*1e-3/(length*rho), label = "$^{37}$Cl", linewidth = 2, color='darkorange')
    plt.loglog(Er_mid*1e3, F23(Er_mid)*1e-3/(length*rho), label = "$^{23}$Na", linewidth = 2, color='darkgreen')        
    plt.loglog(Er_mid*1e3, Frag*1e-3/Er_width/(length*rho), label = 'Fragments', linewidth = 2, color='darkred')

    plt.xlabel("E [$\\mathrm{keV}$]")
    plt.ylabel("dR/dE [$\\mathrm{keV}^{-1}\,\\mathrm{kg}^{-1}\,\\mathrm{Myr}^{-1}\,\\mathrm{sr}^{-1}$]")
    plt.xlim(1e1,1e6)
    #ax.set_ylim([1e0, 1e10])
    plt.legend()
    plt.savefig("recoil_halite/plot/nuclear_recoil_"+name+".png", bbox_inches="tight")

#scrive il dR/dE su un file .dat
def Stampa(name, Conta): 
    f = open("recoil_halite/Halite_muon_recoil_"+name+".dat", "w") #apre un file e ci stampa le intestazioni
    print("# dR/dEr [1/keV/kg/Myr]", file = f)
    # ['S', 'proton', 'F', 'Ne', 'Cl', 'Na', 'Si', 'O', 'alpha', 'P', 'N', 'C', 'Al', 'Mg', 'Be', 'He', 'B', 'Ar', 'Li']
    print("# Er [keV], S, F, Ne, Cl, Na, Si, O, P, N, C, Al, Mg, Be, B, Ar, Li", file = f)

    for i in range(n-1): #n, lenght e rho definiti sotto
        energy = "{:e}".format(Er_mid[i]*1e3) # {:e} annuncia che metterà un float e poi gli da con .format il valore della stringa normalizzata con tutte le costanti del caso
        Na = "{:e}".format((Conta[2][i] + Conta[3][5][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        #print(Conta[2][i])
        Cl = "{:e}".format((Conta[0][i] + Conta[1][i] + Conta[3][4][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)   
        S = "{:e}".format((Conta[3][0][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        Ne  = "{:e}".format((Conta[3][3][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        O  = "{:e}".format((Conta[3][7][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        Si  = "{:e}".format((Conta[3][6][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        Al  = "{:e}".format((Conta[3][12][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        P = "{:e}".format((Conta[3][9][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        F  = "{:e}".format((Conta[3][2][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        N = "{:e}".format((Conta[3][10][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        C  = "{:e}".format((Conta[3][11][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        Mg = "{:e}".format((Conta[3][13][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        B = "{:e}".format((Conta[3][16][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        Li = "{:e}".format((Conta[3][18][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)   
        Be = "{:e}".format((Conta[3][14][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        Ar  = "{:e}".format((Conta[3][17][i])*1e-3/Er_width[i]/(length*rho)*np.pi*2)
        print(energy, S, Ne, Si, O, P, C, F, N, Al, Mg, Na, Be, Cl, B, Ar, Li, sep = "  ", file = f) #printa gli array di ciascun nucleo sul file
    f.close()

#integra il flusso di muoni su energie diverse per il tempo di esposizione
#informazioni su come funzia quad() sparpagliate ad ogni sua chiamata
def Integration(arr_func, arr_time, tot_time): #arr_func= array di funzioni di flusso, arr_time= tempi di esposizione per ciascuna funzione, tot_time=tempo totale di esposizione
    E = np.logspace(0, np.log10(10), 9) #array di 10 bin in scala log
    time = 60*60*24*365*1e6 #un milione di anni
    En = np.zeros(34) 
    Num = np.zeros(34)
    for i in range(8): #divide l'integrazione in 4 intervalli energetici di energia. Qui da 0 a 10 Gev in 8 bin
        for j,func in enumerate(arr_func):
            E_mid = (E[i]+E[i+1])/2. #mezzo bin
            En[i] = E_mid #array con i valori medi dei bin di energia
            Num[i] += quad(func, E[i], E[i+1], epsrel=1e-8)[0]*time*arr_time[j] #quad per integrare la funzione di flusso nell'intervallo, moltiplica per il tempo e accumula i risultati in Num

    E = np.logspace(np.log10(10), np.log10(45), 8) #da 10 a 45 Gev in 7 bin
    for i in range(7):
        for j,func in enumerate(arr_func):
            E_mid = (E[i]+E[i+1])/2.
            En[i+8] = E_mid
            Num[i+8] += quad(func, E[i], E[i+1], epsrel=1e-8)[0]*time*arr_time[j] # quad permette di calcolare l'integrale definito di una funzione in un intervallo specificato

    E = np.logspace(np.log10(45), np.log10(350), 11) #da 45 a 350 Gev in 10 bin
    for i in range(10):
        for j,func in enumerate(arr_func): #ciclo per ogni funzione flusso
            E_mid = (E[i]+E[i+1])/2.
            En[i+15] = E_mid
            Num[i+15] += quad(func, E[i], E[i+1], epsrel=1e-8)[0]*time*arr_time[j] #quad restituisce il risultato dell'integrale e l'errore

    E = np.logspace(np.log10(350), np.log10(2500), 10) #da 350 a 2500 Gev in 9 bin
    for i in range(9):
        for j,func in enumerate(arr_func):
            E_mid = (E[i]+E[i+1])/2.
            En[i+25] = E_mid
            Num[i+25] += quad(func, E[i], E[i+1], epsrel=1e-8)[0]*time*arr_time[j] #epsrel è un parametro che indica la grandezza della tolleranza dell'errore (errore più basso di quello)
    
    return np.array(Num)/tot_time #Num è la somma delle energie di tutti i flussi normalizzato per il tempo totale di esposizione

# taken from eq. 24.6 of PDG chapter on cosmic rays
#come si riduce il flusso in base alla profondità
def par(X): 
    psi = 2.5 #km.w.e.
    return np.exp(X/psi) #formula empirica che dice quanta energia viene assorbita sotto 2.5 km di acqua


def flux_deep(E,X): #E=energia della particella, x=spessore
    E0 = par(X)*(E+510)-510 #510 ?, par(x) fattore correttivo che dipende dalla profondità
    return func(E0)*par(X) #flux in cm^-2 s^-1 GeV^-1

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------



if "__main__" == __name__ :

    energy0, flux0 = np.loadtxt("../Resconi.txt", usecols = (0,1), delimiter=' ',  unpack='true')
    func = log_interp1d(energy0, flux0/(energy0**3)) #prende i flussi da Resconi

    Num_Resconi = Integration([func], [1.], 1.) #integra le energie dei flussi da Resconi

    name = ['100yr', '300yr', '1kyr', '3kyr', '10kyr', '30kyr', '100kyr', '300kyr'] #intestazioni
    arr_time = [200*1e-6, (650-200)*1e-6, (2000-650)*1e-6, (6500-2000)*1e-6, (0.02-0.0065), (0.065-0.02), (0.2-0.065), (0.27-0.2)] #ho aggiunto le parentesi all'ultimo elemento

    listfunc = []
    for i in name:
        x,y = np.loadtxt('MuonFluxes/SN20pc'+i+'.txt', usecols = (0,1), unpack = 'true')
        y = y[:-1]/np.diff(x)
        x = x[:-1]
        listfunc.append(log_interp1d(x,1e-4*y)) #crea listfunc con i flussi da una supernova a 20 pc per diversi tempi di esposizione

    if "deep" in sys.argv: #normalizza i flussi di energia se qualcosa ha una profondità
        for i,_ in enumerate(listfunc):
            xx = np.logspace(0, 3, 200)
            ee = par(1.5)*(xx+510)-510
            x = 1.5
            yy = listfunc[i](ee)*par(x)
            listfunc[i] = log_interp1d(xx, yy)


    Num_SN50pc = Integration(listfunc, arr_time, 1.) #integra le energie dei flussi da una SN a 50pc

    plt.figure() #plotta i flussi di SN a varie distanze, fino ad una certa profondità
    age = ["20pc", "50pc", "100pc"]
    color = ["darkred", "steelblue", "darkorange"]
    for ii, a in enumerate(age):
        xx, yy = np.loadtxt("MuonFluxes/SN%sGCR.txt" %a, usecols=(0,1), unpack=True)
        XX_fit = xx[:-1]
        yy_fit = yy[:-1]/np.diff(xx)*1e-4
        ff = log_interp1d(XX_fit, yy_fit)
        x = np.logspace(0,3,200)
        if "deep" in sys.argv:
            xx = np.logspace(0, 3, 200)
            ee = par(1.5)*(xx+510)-510
            x = 1.5
            yy = ff(ee)*par(x)
            ff = log_interp1d(xx, yy)
            plt.loglog(xx, ff(xx), color=color[ii], linestyle="-.", label="SN%s - $1.5\,\\text{km}$ water" %a)
        else:
            plt.loglog(xx, ff(xx), color=color[ii], linestyle="-.", label="SN%s" %a)

    if "deep" in sys.argv:
        xx = np.logspace(0, 3, 200)
        ee = par(1.5)*(xx+510)-510
        x = 1.5
        yy = func(ee)*par(x)
        func = log_interp1d(xx, yy)
        plt.loglog(xx, func(xx), color="k", linestyle="--", label="GCR - $1.5\,\\text{km}$ water")
    else:
        plt.loglog(xx, func(xx), color="k", linestyle="--", label="GCR")
    plt.xlabel("$E/\\text{GeV}$")
    plt.ylabel("$\\text{d}N/\\text{d}E$")
    plt.legend()
    if "deep" in sys.argv:
        plt.savefig("MuonFluxes/plot/plot_GCR_deep.png")
    else:
        plt.savefig("MuonFluxes/plot/plot_GCR.png")

    EnergyName = ["1.166000", "1.555000", "2.074000", "2.766000", "3.689000", "4.920000", "6.560000", "8.740000", "11.190000", "13.880000", "17.210000", "21.330000",
    "26.440000", "32.780000", "40.640000", "50.120000", "61.530000", "75.540000", "92.740000", "113.860000", "139.780000", "171.610000", "210.600000", "258.600000",
    "317.500000", "392.700000", "488.600000", "607.900000", "756.300000", "941.000000", "1170.700000", "1456.500000",
    "1812.200000", "2254.600000"]

    weightResconi = Num_Resconi/1e4
    weightSN50pc = Num_SN50pc/1e4
    n = 101 #cosa sono?
    Er = np.logspace(-2, 3, n) #MeV
    length = 1000 #cm
    rho = 2.16*1e-3 #kg/cm^3
    Er_width = np.diff(Er)
    Er_mid = Er[:-1] + Er_width/2

    Conta = Count(weightResconi, Er)
    PlotCount(Er, Conta, 'Resconi')
    Stampa('Resconi', Conta)

    # evaluating halite shield for Resconi flux

    if "skip_resc_over" not in sys.argv: #stessa cosa di prima ma più in profondità
        print("doing resconi with overburden!")
        xx = np.logspace(0, 3, 200)
        ee = par(2.0)*(xx+510)-510
        yy = func(ee)*par(2.0)
        func_resc = log_interp1d(xx, yy)
        Num_Resconi = Integration([func_resc], [1.], 1.)
        weight_tmp = Num_Resconi/1e4
        Conta = Count(weight_tmp, Er)
        PlotCount(Er, Conta, "resconi_overburden")
        Stampa("deep_lowedgeedge", Conta)
        print()
        print("done!")
    else: 
        print("slipping resconi with overburden!")
    print()


    age_range = np.linspace(1,270,270) # age range in kyr
    
    if "skip_resc" not in sys.argv:
        for ag in age_range:

            print("doing overburden for %i kyr!" %ag, flush=True, end="\r")
            
            dd = 30*1e-3*(ag-1)*2 # deposit of halite, halite depth = 2 * water depth
            xx = np.logspace(0, 3, 200)
            ee = par(dd)*(xx+510)-510
            yy = func(ee)*par(dd)
            func_resc = log_interp1d(xx, yy)
            Num_Resconi = Integration([func_resc], [0.001], 1.)
            weight_tmp = Num_Resconi/1e4
            Conta = Count(weight_tmp, Er)
            PlotCount(Er, Conta, "resconi_%ikyr" %(ag))
            Stampa("resconi_%ikyr" %(ag), Conta)
        print()
        print("done!")
    else: 
        print("slipping resconi flux")
    print()
    
    


    # evaluating halite shield
    # Rate 2.5 m al kyr

    name_sn = ["20pc", "50pc", "100pc"]
    name = ['100yr', '300yr', '1kyr', '3kyr', '10kyr', '30kyr', '100kyr', '300kyr']
    arr_time = [200*1e-6, (650-200)*1e-6, (2000-650)*1e-6, (6500-2000)*1e-6, (0.02-0.0065), (0.065-0.02), (0.2-0.065), 0.27-0.2]
    tt = [200e-6, 650e-6, 2000e-6, 6500e-6, 0.02, 0.065, 0.2, 0.27] #tempi assoluti

    for nn in name_sn:

        print("evaluating overburden for sn at %s!" %nn)

        listfunc = []
        for i in name: #per ogni distanza carica i flussi e li interpola
            x,y = np.loadtxt("MuonFluxes/SN%s%s.txt" %(nn, i), usecols=(0,1), unpack=True)
            y = y[:-1]/np.diff(x)
            x = x[:-1]
            listfunc.append(log_interp1d(x,1e-4*y))

        age_pre = 1
        delay = 1

        age_range = np.linspace(1,50-delay,50-delay) # age range in kyr

        for ag in age_range:

            print("doing overburden for %i kyr!" %ag, flush=True, end="\r")

            listfunc_tmp = []
            dd = 2.5*1e-3*(ag-1)*2 # deposit of halite, halite depth = 2 * water depth

            listfunc_use = []
            arr_time_use = []
            for i,_ in enumerate(tt):
                if (ag+delay)*1e-3 >= tt[i] and age_pre*1e-3 < tt[i]:
                    if (ag+delay)*1e-3 == tt[i]:
                        arr_time_use.append((ag+delay)*1e-3-age_pre*1e-3)
                        listfunc_use.append(listfunc[i])
                    elif tt[i] > age_pre*1e-3 and tt[i] < (ag+delay)*1e-3 and age_pre!=0:
                        arr_time_use.append(tt[i]-age_pre*1e-3)
                        listfunc_use.append(listfunc[i-1])
                    else:
                        arr_time_use.append(arr_time[i])
                        listfunc_use.append(listfunc[i])
                elif (ag+delay)*1e-3 < tt[i] and (ag+delay)*1e-3 > tt[i-1] :
                    listfunc_use.append(listfunc[i])
                    if age_pre*1e-3 > tt[i-1]:
                        arr_time_use.append((ag+delay)*1e-3-age_pre*1e-3)
                    else:
                        arr_time_use.append((ag+delay)*1e-3-tt[i-1])

            for i,_ in enumerate(listfunc_use):
                xx = np.logspace(0, 3, 200)
                ee = par(dd)*(xx+510)-510
                yy = listfunc_use[i](ee)*par(dd)
                listfunc_tmp.append(log_interp1d(xx, yy))

            Num_tmp = Integration(listfunc_tmp, arr_time_use, 1.)
            weight_tmp = Num_tmp/1e4

            Conta = Count(weight_tmp, Er)
            PlotCount(Er, Conta, "SN%s_%ikyr" %(nn,ag))
            Stampa("SN%s_%ikyr" %(nn,ag), Conta)

            age_pre = ag+delay

        print()
        print()
    
    print("done!")
    exit()
