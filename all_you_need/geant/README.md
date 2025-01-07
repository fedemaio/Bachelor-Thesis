# GEANT directory
# Directory to convert the Geant output in paleopy input


* Halite_QGSP: 
    
    Directory containing the Geant output, obtained simulating 10000 muons passing through a Halite block at different energies.
    The simulation were completed using QGSP_BERT_HP as Physics List.
    The standard files contain the recoiling particle information:
        - name
        - mass
        - energy
        - parent (parent id, only particle with parent id = 1 are considered)
        - z (vertical depth)
        - id
    The recoiling energy is in MeV


* MuonFluxes:

    Directory containing the muon fluxes from a SN at a given distance from Earth after fixed kyr.
    The files have the muon fluxes (in m^-2 s^-1 sr^-1 GeV^-1) as a function of the muon energy (in Gev)


* plot_snfluxes

    Directory containing the plots of the sn fluxes 


* recoil_halite:

    output directory for the recoil in halite, the output is dR/dEr as a function of the energy for all the different nuclei found in Geant simulations.
    dR/dEr in keV^-1 kg^-1 yr^-1.
    Er in keV.


## Halite.py ##

* Halite.py:

    Code where all the magic happens.
    Starting from the observed muon flux at the Earth (options could be standard muon flux, i.e. Resconi, or sn muon fluxes),the code integrates the number of muon expected at an halite block and count the number of recoil.
    The output is a differential recoil (dR/dEr measured in keV^-1 kg^-1 yr^-1) as a function od the energy, as requested by paleopy.
    Options available: 
        - evaluate the differential recoil at a fixed depth, for example in the bottom of the sea. IMPORTANT: always consider the depth in km of water equivalent (km.w.e)
        - evaluate the differential recoil as a function of time, when for example the block is covered by some layers.
