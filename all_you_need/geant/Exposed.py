
#importo tutte le funzioni di Halite.py
from Halite import *

#parametri iniziali
#depths=[100, 200, 300, 400] #profindità del blocco di alite in m

#leggi l'energia del flusso di muoni provenienti da una SN distante 20 pc
#energy0, flux0 = np.loadtxt("/MuonFluxes/SN20pc1kyr.txt", usecols = (0,1), delimiter=' ',  unpack='true')
#interpola in un array di funzioni i flussi e le energie appena letti
#func = log_interp1d(energy0, flux0/(energy0**3)) 


#calcola il dR/dE del flusso di questi muoni in uno strato di alite
#Count(pesi, energia)
#stampa questi dati in un file di output
#Stampa("fluxes_SN20pc_exposed.dat", Conta)


#plotta dR/dE in funzione di x per diversi spessori dello strato di alite
#for valore in depths:
    #plt.figure(figsize=(12,8))
    #x,y,=np.loadtxt("", usecols = (0,1), dtype=str)

    #plt.xlabel("E")
    #plt.ylabel("dR/dE [1/keV/kg/Myr]")
    #plt.legend()
    #plt.savefig("flux_muon_SN20pc.png", bbox_inches="tight")

################################################################
################################################################
