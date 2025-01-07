import ROOT
from array import array
from math import *

gr_reso  = ROOT.TGraph()
gr_shmax = ROOT.TGraph() 
hist2  = ROOT.TH1F()
hist12 = ROOT.TH1F() 

shower = ROOT.TF1("shower","[2]*[1]* ( pow([1]*x,[0]-1)*exp(-1*[1]*x))/(TMath::Gamma([0])) ",0,50);
shower.SetParameter(0,1) ;
shower.SetParameter(1,0.5) ;
shower.SetParameter(2,1);

k = 0;

# loop on the available simulated files and get infos

for length in [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ] :
    particle = "e"
    #length = 23
    filename = "data_homo_" +particle+ "_10GeV_"+ str(length) +"cm_3.2.root"
    f = ROOT.TFile.Open(filename)
    hist2 = f.Get("1")
    hist12 = f.Get("11")    
    hist12.Fit(shower, "","",0,50);
    a = shower.GetParameter(0);
    b = shower.GetParameter(1);
    smax = (a-1.) / b ;
  
    gr_shmax.SetPoint(k,length, smax );    
    gr_reso.SetPoint( k, length, hist2.GetRMS()/hist2.GetMean() )

    k = k+1    
        
# do some plots

c1 = ROOT.TCanvas("c1", "c1");

gr_reso.SetMarkerStyle(20);
gr_reso.SetMarkerSize(1);
gr_reso.SetTitle("Energy resolution")
gr_reso.GetXaxis().SetTitle("DMLength [cm]")
gr_reso.GetYaxis().SetTitle("Resolution ")
gr_reso.SetLineColor(1)
gr_reso.SetMarkerColor(1)
gr_reso.Draw("ALP");

c1.SaveAs("EnergyResolution_3.2.pdf")

#c2 = ROOT.TCanvas("c2", "c2");

#gr_shmax.SetMarkerStyle(20);
#gr_shmax.SetMarkerSize(1);
#gr_shmax.GetXaxis().SetTitle("Length [cm]")
#gr_shmax.GetYaxis().SetTitle("Shower Max (AU)")
#gr_shmax.SetLineColor(1)
#gr_shmax.SetMarkerColor(1)
#gr_shmax.Draw("ALP");

#c2.SaveAs("ShowerMax_length.pdf")

input()
