import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, quad
from scipy.special import erf

import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d
import configparser

from tqdm import tqdm
import swordfish as sf
from WIMpy import DMUtils as DMU
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

class Mineral:
    def __init__(self, mineral, muon = "false", water = "false"):
        
        #mineral in config
        
        self.name = mineral
        
        #legge i dati dalla Mineral list e li salva per uno specifico minerale
        config = configparser.ConfigParser()
        config.read(dir_path + "/Data/MineralList.txt") 
        data = config[mineral]
    
        nuclist = data["nuclei"].split(",")
        self.nuclei = [x.strip(' ') for x in nuclist]
        
        self.N_nuclei = len(self.nuclei)
        
        self.stoich = np.asarray(data["stoich"].split(","), dtype=float)
        
        #self.abun = np.asarray(data["abundances"].split(","), dtype=float)
        self.N_p = np.asarray(data["N_p"].split(","), dtype=float)
        self.N_n = np.asarray(data["N_n"].split(","), dtype=float)
        self.shortname = data["shortname"]
        if(self.shortname=="Hal" or self.shortname=="Obs" or self.shortname=="Cer" or self.shortname=="Pyr" or self.shortname == "Zir"):
            fraglist = data["fragments"].split(",")
            self.frag = [x.strip(' ') for x in fraglist]
        
        #Check that there's the right number of everything
        if (len(self.stoich) != self.N_nuclei):
            raise ValueError("Number of stoich. ratio entries doesn't match number of nuclei for mineral <" + self.name + ">...")
        if (len(self.N_p) != self.N_nuclei):
            raise ValueError("Number of N_p entries doesn't match number of nuclei for mineral <" + self.name + ">...")
        if (len(self.N_p) != self.N_nuclei):
            raise ValueError("Number of N_n entries doesn't match number of nuclei for mineral <" + self.name + ">...")
        
        self.U_frac = float(data["U_frac"]) #Uranium fraction by weight
        
        #Calculate some derived stuff
        self.molarmass = np.sum(self.stoich*(self.N_p + self.N_n))
        self.abun = self.stoich*(self.N_p + self.N_n)/self.molarmass

        
        self.dEdx_interp = []
        self.Etox_interp = []
        self.xtoE_interp = []
        
        self.Etox_interp_Th = None
        
        if (self.shortname == "Zab"):
            self.loadSRIMdata(modifier="CC2338")
        elif (self.shortname == "Syl"):
            self.loadSRIMdata(modifier="CC1")
        else:
            self.loadSRIMdata()
        
        self.NeutronBkg_interp = []
        
        self.loadNeutronBkg()
        
        if(muon == "true"):
            #self.loadMuon()
            #self.loadMuonModified()
            A = self.loadMuon_num(0)
            B = self.loadMuon_num(1)
            C = self.loadMuon_num(2)
            D = self.loadMuon_num(3)
            E = self.loadMuon_num(4)
            F = self.loadMuon_num(5)
            G = self.loadMuon_num(6)
            H = self.loadMuon_num(7)
            I = self.loadMuon_num(8)
            J = self.loadMuon_num(9)
            self.Muon_interpArray = [A, B, C, D, E, F, G, H, I, J]
        
        if(muon == "true"):
            A = self.loadMuonWater(0)
            B = self.loadMuonWater(1)
            C = self.loadMuonWater(2)
            D = self.loadMuonWater(3)
            E = self.loadMuonWater(4)
            F = self.loadMuonWater(5)
            self.MuonWater_interpArray = [A, B, C, D, E, F]
            
        if(self.shortname == "Zir"):
            A = self.loadMuon(0)
            self.Muon_interpArray = [A]
            
        
        #self.loadFissionBkg()
        
        #Do we need these cumbersome dictionaries...?
        self.dEdx_nuclei = dict(zip(self.nuclei, self.dEdx_interp))
        self.Etox_nuclei = dict(zip(self.nuclei, self.Etox_interp))
        self.xtoE_nuclei = dict(zip(self.nuclei, self.xtoE_interp))
        self.ratio_nuclei = dict(zip(self.nuclei, self.abun))

    #--------------------------------   
    def showProperties(self):
        print("Mineral name:", self.name)
        print("    N_nuclei:", self.N_nuclei)
        print("    Molar mass:", self.molarmass, " g/mol")
        print("    nucleus \t*\t abun.  *\t (N_p, N_n)")
        print(" **************************************************")
        for i in range(self.N_nuclei):
            print("    " + self.nuclei[i] + "\t\t*\t" + str(self.abun[i]) + "\t*\t(" +str(self.N_p[i]) + ", " + str(self.N_n[i]) + ")")
         
    #--------------------------------   
    def loadSRIMdata(self, modifier=None):
        #The modifier can be used to identify a particular version of the SRIM
        #track length files (e.g. modifier="CC2338")
        
        SRIMfolder = dir_path + "/Data/dRdESRIM/"

        self.Etox_interp = []
        self.xtoE_interp = []
        self.dEdx_interp = []
    
        for nuc in self.nuclei:
            #Construct the SRIM output filename
            infile = SRIMfolder + nuc + "-" + self.shortname
            if not(modifier == None):
                infile += "-" + modifier
            infile += ".txt"
        
            E, dEedx, dEndx = np.loadtxt(infile, usecols=(0,1,2), unpack=True)
            dEdx = dEedx + dEndx    #Add electronic stopping to nuclear stopping
            dEdx *= 1.e-3           # Convert keV/micro_m to keV/nm
            x = cumtrapz(1./dEdx,x=E, initial=0)    #Calculate integrated track lengths
        
            #Generate interpolation function (x(E), E(x), dEdx(x))
            self.Etox_interp.append(interp1d(E, x, bounds_error=False, fill_value='extrapolate'))
            self.xtoE_interp.append(interp1d(x, E, bounds_error=False, fill_value='extrapolate'))
            self.dEdx_interp.append(interp1d(x, dEdx, bounds_error=False, fill_value='extrapolate'))    
    
        #Load in the Thorium track lengths...
        #Construct the SRIM output filename
        infile = SRIMfolder + "Th-" + self.shortname
        if not(modifier == None):
            infile += "-" + modifier
        infile += ".txt"
        
        E, dEedx, dEndx = np.loadtxt(infile, usecols=(0,1,2), unpack=True)
        dEdx = dEedx + dEndx    #Add electronic stopping to nuclear stopping
        dEdx *= 1.e-3           # Convert keV/micro_m to keV/nm
        x = cumtrapz(1./dEdx,x=E, initial=0)    #Calculate integrated track lengths
        self.Etox_interp_Th = interp1d(E, x, bounds_error=False, fill_value='extrapolate')
    
    

    
    #--------------------------------
    def showSRIM(self):
        print("Plotting SRIM data for " + self.name + ":")
        x_list = np.logspace(0,4,100)

        fig, axarr = plt.subplots(figsize=(10,4),nrows=1, ncols=2)
        ax1, ax2 = axarr
        for i in range(self.N_nuclei):
            ax1.loglog(x_list, self.dEdx_interp[i](x_list),label=self.nuclei[i])
        ax1.set_ylabel("dE/dx [keV/nm]")
        ax1.set_xlabel("x [nm]")
        ax1.legend()
                
        E_list = np.logspace(-3, 3, 500) # keV    
        
        for i in range(self.N_nuclei):
            ax2.loglog(E_list, self.Etox_interp[i](E_list),label=self.nuclei[i])
        ax2.set_ylabel("x [nm]")
        ax2.set_xlabel("E [keV]")
        ax2.legend()
        
        plt.savefig(self.name + 'SRIM.pdf', bbox_inches='tight')
        plt.show()
        
        
    #--------------------------------
    def dRdx(self, x_bins, sigma, m, gaussian=False):
        x_width = np.diff(x_bins)
        x = x_bins[:-1] + x_width/2
        #Returns in events/kg/Myr/nm

        
        dRdx = np.zeros_like(x)
        for i, nuc in enumerate(self.nuclei):
            # Ignore recoiling hydrogen nuclei
            if (nuc != "H"):
                Etemp = self.xtoE_nuclei[nuc](x)
                dRdx_nuc = (DMU.dRdE_standard(Etemp, self.N_p[i], self.N_n[i], m, sigma, \
                                        vlag=248.0, sigmav=166.0, vesc=550.0)*self.dEdx_nuclei[nuc](x))
                dRdx += self.ratio_nuclei[nuc]*dRdx_nuc
            
        if gaussian:
            dRdx = gaussian_filter1d(dRdx,1)+1e-20
        return dRdx*1e6*365

    def dRdx_generic_vel(self, x_bins, sigma, m, eta, gaussian=False):
        x_width = np.diff(x_bins)
        x = x_bins[:-1] + x_width/2
        #Returns in events/kg/Myr/nm

        
        dRdx = np.zeros_like(x)
        for i, nuc in enumerate(self.nuclei):
            # Ignore recoiling hydrogen nuclei
            if (nuc != "H"):
                Etemp = self.xtoE_nuclei[nuc](x)
                dRdx_nuc = (DMU.dRdE_generic(Etemp, self.N_p[i], self.N_n[i], m, sigma, eta)*self.dEdx_nuclei[nuc](x))
                dRdx += self.ratio_nuclei[nuc]*dRdx_nuc
            
        if gaussian:
            dRdx = gaussian_filter1d(dRdx,1)+1e-20
        return dRdx*1e6*365
    
    #--------------------------------
    def dRdx_nu(self,x_bins, components=False, gaussian=False):
        x_width = np.diff(x_bins)
        x = x_bins[:-1] + x_width/2
        #Returns in events/kg/Myr/nm
        nu_list = ['DSNB', 'atm', 'hep', '8B', '15O', '17F', '13N', 'pep','pp','7Be-384','7Be-861']
    
        E_list = np.logspace(-3, 3, 5000) # keV
    
        if components:
            dRdx = []
            for j, nu_source in enumerate(nu_list):
                dRdx_temp = np.zeros_like(x)
                for i, nuc in enumerate(self.nuclei):
                    if (nuc != "H"):
                        xtemp = self.Etox_nuclei[nuc](E_list)
                        dRdx_nuc = (np.vectorize(DMU.dRdE_CEvNS)(E_list, self.N_p[i], self.N_n[i], flux_name=nu_source)
                                                            *self.dEdx_nuclei[nuc](xtemp))
                        temp_interp = interp1d(xtemp, dRdx_nuc, fill_value='extrapolate')
                        dRdx_temp += self.ratio_nuclei[nuc]*temp_interp(x)
                    
                if gaussian:
                    dRdx.append(gaussian_filter1d(dRdx_temp*1e6*365,1)+1e-20)
                else:
                    dRdx.append(dRdx_temp*1e6*365+1e-20)
        else:
            dRdx = np.zeros_like(x)
            for i, nuc in enumerate(self.nuclei):
                if (nuc != "H"):
                    xtemp = self.Etox_nuclei[nuc](E_list)
                    dRdx_nuc = (np.vectorize(DMU.dRdE_CEvNS)(E_list, self.N_p[i], self.N_n[i], flux_name='all')
                                                        *self.dEdx_nuclei[nuc](xtemp))
                    temp_interp = interp1d(xtemp, dRdx_nuc, fill_value='extrapolate')
                    dRdx += self.ratio_nuclei[nuc]*temp_interp(x)*1e6*365
            if gaussian:
                dRdx = gaussian_filter1d(dRdx*1e6*365,1)+1e-20
                
        return dRdx
    
    def xT_Thorium(self):
        E_Thorium = 72. #keV
        return self.Etox_interp_Th(E_Thorium)
    
    def norm_Thorium(self, T):
        #T is in years. Returns events/kg/Myr
        T_half_238 = 4.468e9
        T_half_234 = 2.455e5
        
        lam_238 = np.log(2)/T_half_238
        lam_234 = np.log(2)/T_half_234
        
        #Avocado's constant
        N_A = 6.022140857e23
        

        n238_permass = self.U_frac*N_A*1e3/238.0 #Number of U238 atoms *per kg*
        Nalpha = n238_permass*(lam_238/(lam_234 - lam_238))*(np.exp(-lam_238*T) - np.exp(-lam_234*T))
        return Nalpha/(T*1e-6)
        
    def loadNeutronBkg(self):
        
        fname = dir_path + "/Data/" + self.name + "_ninduced_wan.dat"

        #Read in the column headings so you know which element is which
        f = open(fname)
        head = f.readlines()[1]
        columns = head.split(",")
        columns = [c.strip() for c in columns]
        ncols = len(columns)
        f.close()
        
        data = np.loadtxt(fname)
        E_list = data[:,0]
        
        self.NeutronBkg_interp = []
        
        for i, nuc in enumerate(self.nuclei):
            dRdE_list = 0.0*E_list
            #How many characters is the length of the element name you're looking for
            nchars = len(nuc)
            for j in range(ncols):
                #Check if this is the correct element
                if (columns[j][0:nchars] == nuc):
                    dRdE_list += data[:,j]
            
            (self.NeutronBkg_interp).append(interp1d(E_list, dRdE_list,bounds_error=False,fill_value=0.0))
            
    def dRdx_neutrons(self, x_bins):
        x_width = np.diff(x_bins)
        x = x_bins[:-1] + x_width/2
        #Returns in events/kg/Myr/nm
        
        
        dRdx = np.zeros_like(x)
        for i, nuc in enumerate(self.nuclei):
            if (nuc != "H"):
                E_list = self.xtoE_nuclei[nuc](x) 
                dRdx_nuc = self.NeutronBkg_interp[i](E_list)*self.dEdx_nuclei[nuc](x)
                dRdx += dRdx_nuc #Isotope fractions are already included in the tabulated neutron spectra
                
        return dRdx*self.U_frac/0.1e-9 #Tables were generated for a Uranium fraction of 0.1 ppb

    def loadMuon_num(self, k):
        tag = ["Resconi", "Modified1", "Modified2", "Modified3", "Modified4", "Modified5", 
               "deep", "SN20pc", "SN100pc", 'SN50pc']
        Muon_interp = []
        #for k in range(5):
        fname = dir_path + "/Data/" + self.name + "_muon_recoil_"+tag[k]+".dat"

        #Read in the column headings so you know which element is which
        f = open(fname)
        head = f.readlines()[1]
        columns = head.split(",")
        columns = [c.strip() for c in columns]
        
        ncols = len(columns)
        f.close()
        
        data = np.loadtxt(fname)
        E_list = data[:,0]
        #print(data[:,0])
        #print(len(data[:,0]))
        
        for i, nuc in enumerate(self.frag):
            #print(nuc)
            dRdE_list = 0.0*E_list
            
            #How many characters is the length of the element name you're looking for
            nchars = len(nuc)
            for j in range(ncols):
                #print(columns[j])
                #if(k==4): print(j)
                #Check if this is the correct element
                if (columns[j] == nuc):
                    
                    dRdE_list += data[:,j]
            #print(dRdE_list )
            (Muon_interp).append(interp1d(E_list, dRdE_list,bounds_error=False,fill_value=0.0))
        #print(k)
        return Muon_interp
    
    def loadMuon(self, tag_name, complete=False):
        
        Muon_interp = []
        #for k in range(5):
        if complete==True:
            fname = tag_name
        else:
            fname = dir_path + "/Data/" + self.name + "_muon_recoil_"+tag_name+".dat"

            #Read in the column headings so you know which element is which
        f = open(fname)
        head = f.readlines()[1]
        columns = head.split(",")
        columns = [c.strip() for c in columns]
        
        ncols = len(columns)
        f.close()
        
        data = np.loadtxt(fname)
        E_list = data[:,0]
        #print(data[:,0])
        #print(len(data[:,0]))
        
        for i, nuc in enumerate(self.frag):
            #print(nuc)
            dRdE_list = 0.0*E_list
            
            #How many characters is the length of the element name you're looking for
            nchars = len(nuc)
            for j in range(ncols):
                #print(columns[j])
                #if(k==4): print(j)
                #Check if this is the correct element
                if (columns[j] == nuc):
                    
                    dRdE_list += data[:,j]
            #print(dRdE_list )
            (Muon_interp).append(interp1d(E_list, dRdE_list,bounds_error=False,fill_value=0.0))
        #print(k)
        return Muon_interp
    
    #carica le tracce dei muoni da file in base ai diversi tipi d esposizione
    def loadMuonSNoverburden(self, name, delay=0, fast=False, thin=False):

        step = 1
        if thin == True:
            step = 10

        for ii in range(1,(50-delay)*step):

            name_file = dir_path + "/Data/%s_%ikyrdelay"% (name, delay)
            if fast == True:
                name_file = name_file + "_fast"
            name_file = name_file + "_0.1kyrstep"

            if thin == True:
                name_file = name_file +  "/Halite_muon_recoil_%s_%.1fkyr.dat" %(name, ii/10.)
            else:
                name_file = name_file +  "/Halite_muon_recoil_%s_%.1fkyr.dat" %(name, ii)
            # print(name_file)

            if name == "resconi":
                name_file = dir_path + "/Data/%s" %(name)
                if fast == True:
                    name_file = name_file + "_fast"
                name_file = name_file + "_0.1kyrstep"

                if thin == True:
                    name_file = name_file +  "/Halite_muon_recoil_%s_%.1fkyr.dat" %(name, ii/10.)
                else:
                    name_file = name_file +  "/Halite_muon_recoil_%s_%.1fkyr.dat" %(name, ii)
            
            if ii == 1:
                muon_int = self.loadMuon(name_file, complete=True)
            else: 
                muon_int += self.loadMuon(name_file, complete=True)
        
        return muon_int

    
    
    def loadMuonWater(self, k):
        tag = ["Resconi", "Modified1", "Modified2", "Modified3", "Modified4", "Modified5"]
        Muon_interp = []
        #for k in range(5):
        fname = dir_path + "/Data/" + self.name + "_muon_recoil_Water_"+tag[k]+".dat"

            #Read in the column headings so you know which element is which
        f = open(fname)
        head = f.readlines()[1]
        columns = head.split(",")
        columns = [c.strip() for c in columns]
        ncols = len(columns)
        f.close()
        
        data = np.loadtxt(fname)
        E_list = data[:,0]
        #print(data[:,1])
        
        for i, nuc in enumerate(self.frag):
            #print(nuc)
            dRdE_list = 0.0*E_list
            #How many characters is the length of the element name you're looking for
            nchars = len(nuc)
            for j in range(ncols):
                #print(j)
                #Check if this is the correct element
                if (columns[j] == nuc):
                    #print("QUA")
                    dRdE_list += data[:,j]
            (Muon_interp).append(interp1d(E_list, dRdE_list,bounds_error=False,fill_value=0.0))
        #print(k)
        return Muon_interp

            
    def dRdx_muons(self, y_bins, n, N = 1, water = "false"):
        y_width = np.diff(y_bins)
        y = y_bins[:-1] + y_width/2.
        #Returns in events/kg/Myr/nm
        
        # self.frag = ['S', 'Ne', 'O', 'Si', 'Al', 'P', 'F', 'N', 'C', 'Mg', 'Na', 'He', 'Li', 'Be', 'Cl', 'Ar']
        # B ha problemi per halite
        dRdx_mu = np.zeros(len(y))
        for i, nuc in enumerate(self.frag):
            
            SRIMfolder = dir_path + "/Data/dRdESRIM/"
            if (nuc not in self.nuclei):
                SRIMfolder += "Fragment/"
                
            infile = SRIMfolder + nuc + "-" + self.shortname
            infile += ".txt"
        
            E, dEedx, dEndx = np.loadtxt(infile, usecols=(0,1,2), unpack=True)
            dEdx = dEedx + dEndx    #Add electronic stopping to nuclear stopping
            dEdx *= 1.e-3           # Convert keV/micro_m to keV/nm
            x = cumtrapz(1./dEdx,x=E, initial=0)    #Calculate integrated track lengths
        
            #Generate interpolation function (x(E), E(x), dEdx(x))
            Etox_interp = interp1d(E, x, bounds_error=False, fill_value='extrapolate')
            xtoE_interp = interp1d(x, E, bounds_error=False, fill_value='extrapolate')
            dEdx_interp = interp1d(x, dEdx, bounds_error=False, fill_value='extrapolate')
            
            E_list = xtoE_interp(y)
            if(water == "true"): dRdx_nuc = N*self.MuonWater_interpArray[n][i](E_list)*dEdx_interp(y)
            else: dRdx_nuc = N*self.Muon_interpArray[n][i](E_list)*dEdx_interp(y)
            #print(len(x), len(dRdx_mu))
            dRdx_mu += dRdx_nuc #Isotope fractions are already included in the tabulated muon spectra
            #print("QUA")
        
        return dRdx_mu

    def dRdx_muons_new(self, y_bins, tag_name, N = 1, water = "false", compl=False, delay=0, fast=False, thin=False):
        y_width = np.diff(y_bins)
        y = y_bins[:-1] + y_width/2.
        #Returns in events/kg/Myr/nm
        
        self.frag = ['S', 'Ne', 'O', 'Si', 'Al', 'P', 'F', 'N', 'C', 'Mg', 'Na', 'He', 'Li', 'Be', 'Cl', 'Ar']
        # B ha problemi per halite
        dRdx_mu = np.zeros(len(y))
        for i, nuc in enumerate(self.frag):
            
            SRIMfolder = dir_path + "/Data/dRdESRIM/"
            if (nuc not in self.nuclei):
                SRIMfolder += "Fragment/"
                
            infile = SRIMfolder + nuc + "-" + self.shortname
            infile += ".txt"
        
            E, dEedx, dEndx = np.loadtxt(infile, usecols=(0,1,2), unpack=True)
            dEdx = dEedx + dEndx    #Add electronic stopping to nuclear stopping
            dEdx *= 1.e-3           # Convert keV/micro_m to keV/nm
            x = cumtrapz(1./dEdx,x=E, initial=0)    #Calculate integrated track lengths
        
            #Generate interpolation function (x(E), E(x), dEdx(x))
            Etox_interp = interp1d(E, x, bounds_error=False, fill_value='extrapolate')
            xtoE_interp = interp1d(x, E, bounds_error=False, fill_value='extrapolate')
            dEdx_interp = interp1d(x, dEdx, bounds_error=False, fill_value='extrapolate')
            
            E_list = xtoE_interp(y)
            if(water == "true"): 
                dRdx_nuc = N*self.loadMuon(tag_name)[i](E_list)*dEdx_interp(y)
            elif compl == True:
                dRdx_nuc = N*self.loadMuonSNoverburden(tag_name, delay, fast, thin)[i](E_list)*dEdx_interp(y)
            else: 
                dRdx_nuc = N*self.loadMuon(tag_name)[i](E_list)*dEdx_interp(y)
            #print(len(x), len(dRdx_mu))
            dRdx_mu += dRdx_nuc #Isotope fractions are already included in the tabulated muon spectra
            #print("QUA")
        
        return dRdx_mu


    def loadMuonModified(self):
        
        fname = dir_path + "/Data/" + self.name + "_muon_recoil_Claudio1.dat"

        #Read in the column headings so you know which element is which
        f = open(fname)
        head = f.readlines()[1]
        columns = head.split(",")
        columns = [c.strip() for c in columns]
        ncols = len(columns)
        f.close()
        
        data = np.loadtxt(fname)
        E_list = data[:,0]
        
        self.Muon_interp1 = []
        
        for i, nuc in enumerate(self.frag):
            dRdE_list = 0.0*E_list
            #How many characters is the length of the element name you're looking for
            nchars = len(nuc)
            for j in range(ncols):
                #Check if this is the correct element
                if (columns[j][0:nchars] == nuc):
                    dRdE_list += data[:,j]
            
            (self.Muon_interp1).append(interp1d(E_list, dRdE_list,bounds_error=False,fill_value=0.0))
            
        fname = dir_path + "/Data/" + self.name + "_muon_recoil_Claudio2.dat"

        #Read in the column headings so you know which element is which
        f = open(fname)
        head = f.readlines()[1]
        columns = head.split(",")
        columns = [c.strip() for c in columns]
        ncols = len(columns)
        f.close()
        
        data = np.loadtxt(fname)
        E_list = data[:,0]
        
        self.Muon_interp2 = []
        
        
        for i, nuc in enumerate(self.frag):
            dRdE_list = 0.0*E_list
            #How many characters is the length of the element name you're looking for
            nchars = len(nuc)
            for j in range(ncols):
                #Check if this is the correct element
                if (columns[j][0:nchars] == nuc):
                    dRdE_list += data[:,j]
            
            (self.Muon_interp2).append(interp1d(E_list, dRdE_list,bounds_error=False,fill_value=0.0))

    ###############################################FUNZIONE AGGIUNTA DA ME###########################################################################################
    #funzione per il calcolo delle tracce adattata
    def dRdx_modified(self, x_bins, Er, dRdE_vals, gaussian=False):
        x_width = np.diff(x_bins)
        x = x_bins[:-1] + x_width / 2 
        
        dRdE_interp = interp1d(Er, dRdE_vals, bounds_error=False, fill_value=0.0)
        
        dRdx = np.zeros_like(x)

        for i, nuc in enumerate(self.nuclei):
            # Ignore recoiling hydrogen nuclei
            if nuc != "H":
                Etemp = self.xtoE_nuclei[nuc](x)
                dRdx_nuc = dRdE_interp(Etemp) * self.dEdx_nuclei[nuc](x)
                #aggiungi il contributo ponderato del nucleo che sto considerando
                dRdx += self.ratio_nuclei[nuc] * dRdx_nuc

        if gaussian:
            dRdx = gaussian_filter1d(dRdx, 1) + 1e-20

        return dRdx * 1e6 * 365
            
#--------------------------------------------

def Max(x, x_a, x_b, sigma, N):
    m = (x_b+x_a)/2.0
    if(x<=m): return 2*N/(N+1)*1/np.sqrt(2*3.14*sigma*sigma)*np.exp(-(x-m)*(x-m)/(2*sigma*sigma))
    else: return 2*N/(N+1)*1/np.sqrt(2*3.14*sigma*sigma)*np.exp(-(x-m)*(x-m)/(2*(sigma/N)*sigma/N))

def window(x, x_a, x_b, sigma):
    return 0.5*(erf((x - x_a)/(np.sqrt(2.0)*sigma)) - erf((x - x_b)/(np.sqrt(2.0)*sigma)))

def calcBins(sigma):
    x0 = sigma/2.0
    x_bins = np.logspace(np.log10(x0), 3, 70)
    return x_bins
    
def calcBins_1nm():
    x0 = 0.5
    x_bins = np.arange(x0, 100.5)
    x_bins = np.append(x_bins, np.arange(100.5, 1001,10))
    return x_bins
    
def GetBackground(mineral, sigma, x_bins=None):
    x0 = sigma/2.0
    
    x_bins_all = np.logspace(-1, 3,200)
    x_width_all = np.diff(x_bins_all)
    x_c_all = x_bins_all[:-1] + x_width_all/2

    if (x_bins is None):
        x_bins = calcBins(sigma)
    N_bins = len(x_bins) - 1
    
    Nevents_BG = []
    
    T_exp = 1e7 #Set exposure time for the Thorium background
    
    dRdx_BG = mineral.dRdx_nu(x_bins_all, components=True, gaussian=False)
    dRdx_BG.append(mineral.dRdx_neutrons(x_bins_all))
    
    for dRdx in dRdx_BG:
        #dRdx_smooth = gaussian_filter1d(dRdx, sigma, mode='constant',cval = 1e-30)
        dRdx_interp = interp1d(x_c_all, dRdx, bounds_error=False, fill_value=0.0)
        
        N_events_ind = np.zeros(N_bins)
        for i in range(N_bins):
            # xmean = 0.5*(x_bins[i] + x_bins[i+1])
            # x1 = xmean - 5.0*sigma
            # x2 = xmean + 5.0*sigma
            x1 = x_bins[i] - 5.0*sigma
            x2 = x_bins[i+1] + 5.0*sigma
            # x1 = np.clip(x1, 0.1, 1e5)
            #print(xmean, x1, x2)
            integ = lambda y: dRdx_interp(y)*window(y, x_bins[i], x_bins[i+1], sigma)
            #print(integ(xmean))
            N_events_ind[i] = quad(integ, x1, x2, epsrel=1e-4)[0] + 1e-30
        
        Nevents_BG.append(N_events_ind)
        
        #dRdx *= x_width
        
    
    #Add the (smeared) delta-function for Thorium recoils
    Nevents_Th = np.zeros(N_bins)
    x_Th = mineral.xT_Thorium()
    for i in range(N_bins):
        Nevents_Th[i] = window(x_Th, x_bins[i], x_bins[i+1], sigma)
    
    Nevents_BG.append(mineral.norm_Thorium(T=T_exp)*Nevents_Th)
    
    return Nevents_BG

def GetSignal(mineral, sigma, m_DM, xsec, x_bins=None, eta=None, vel_dis=False):
    x0 = sigma/2.0
    
    x_bins_all = np.logspace(-1, 3,200)
    x_width_all = np.diff(x_bins_all)
    x_c_all = x_bins_all[:-1] + x_width_all/2
    
    if (x_bins is None):
        x_bins = calcBins(sigma)
    N_bins = len(x_bins) - 1
    
    if vel_dis:
        dRdx_sig = mineral.dRdx_generic_vel(x_bins_all, xsec, m_DM, eta, gaussian=False)
    else:
        dRdx_sig = mineral.dRdx(x_bins_all, xsec, m_DM, gaussian=False)
    #dRdx_smooth = gaussian_filter1d(dRdx_sig,sigma, mode='constant',cval = 1e-30)
    dRdx_interp = interp1d(x_c_all, dRdx_sig, bounds_error=False, fill_value=0.0)
    
    Nevents_sig = np.zeros(N_bins)
    
    for i in range(N_bins):
        # xmean = 0.5*(x_bins[i] + x_bins[i+1])
        x1 = x_bins[i] - 5.0*sigma
        x2 = x_bins[i+1] + 5.0*sigma
        # x1 = np.clip(x1, 0.1, 1e5)
        #x1 = x_bins[i]
        #x2 = x_bins[i+1]
        integ = lambda y: dRdx_interp(y)*window(y, x_bins[i], x_bins[i+1], sigma)

        
        
        Nevents_sig[i] = quad(integ, x1, x2,epsrel=1e-4)[0]
    
    #Nevents_sig = np.array([quad(dRdx_interp, x_bins[i], x_bins[i+1])[0] for i in range(N_bins)])
    
    return Nevents_sig + 1e-30
    
