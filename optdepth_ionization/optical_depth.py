##task 1
##Make functions that calculate the optical depth as a function of altitude and wavelength
# for vertical incidence, and for variable zenith-angle of the incident light

#make plots with altitude variations of thermospheric densities
#make plots with wavelength variations of cross sections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib as mpl

#read in all necessary data and initiate vectors/arrays

#heightprofiles of number densities [cm^-3]
n_data = pd.read_csv("atmosphere/MSIS.dat",sep=r"\s+", skiprows=16)
height = n_data.iloc[:,0].to_numpy()

#Irradiance [W/m^2/nm] for different wavelengths [nm]
I_data = pd.read_csv("optdepth_ionization/fism_daily_hr19990216.dat")
wl = I_data["wavelength (nm)"].to_numpy()*1e-9 #wavelength in data in meters
wl = wl[:1000]
I_inf = I_data["irradiance (W/m^2/nm)"].to_numpy()[:1000]*1e9 #irradiance on top of atmopshere for different wavelengths [m]

#crosssections [m^2] for different species for different wavelengths [m]
cs_data = pd.read_csv("optdepth_ionization/phot_abs.dat",sep=r"\s+", skiprows=8)
wl_short = cs_data.iloc[:,0].to_numpy()
cs_N2 = cs_data.iloc[:,1].to_numpy()
cs_O = cs_data.iloc[:,2].to_numpy()
cs_O2 = cs_data.iloc[:,3].to_numpy()

#interpolate cross sections to irradiance-wavelengths
pol_N2 = np.interp( wl[:1000], wl_short, cs_N2 )
pol_O = np.interp( wl[:1000], wl_short, cs_O )
pol_O2 = np.interp( wl[:1000], wl_short, cs_O2 )

abs_cs = [pol_O, pol_N2, pol_O2] #ordered to fit the order in the n_data file [m^2]
abs_cs_matrix = np.vstack(abs_cs)


#number densities for heightprofiles [m^-3]
n_all = n_data.iloc[:,1:4].to_numpy()*1e6
#column density, integrate from the top down in steps of 1km
dz=1000
column_n = np.cumsum(n_all[::-1, :] * dz, axis=0)[::-1]

#relevant angles for Xi
Xi = [0, 15, 30, 45, 60, 75] #[degrees]



################define functions###############
#optical depth for different angles of Xi, returns a list over all wavelength present in input data
#this should work for angles up to 60 degrees (maybe 75)
def Tau(column_density, absorption_cs, angle):
    tau = column_density[100:] @ absorption_cs
    return tau /np.cos( np.deg2rad(angle) )

#Irradiance
def Irradiance(opt_depth, Irradiance):
    return np.array(Irradiance) * np.exp(-opt_depth)

##convert Watts/m^2 into photons/s/m^2 (Plancks law E= hc/wavelength)
def watts2photons(E_total, wavelengths):
    h = 6.62607015e-34 #plancks constant
    c = 2.99792458e8 #speed of light [m/s]
    E_i = h*c/np.array(wavelengths)
    nr_photons = E_total/E_i
    return nr_photons

###############calculate optical depth and irradiance#################
#calculate optical depth for heights in the thermosphere (100-600km)
#and for different angles of incidence

optical_depth = [] #in [m]
irradiance = []
irradiance_ph = []
for X in Xi:
    T = Tau(column_n, abs_cs_matrix, X)
    I = Irradiance(T , I_inf)
    I_ph = watts2photons(I, wl)
    optical_depth.append(T)
    irradiance.append(I)
    irradiance_ph.append(I_ph)



#make plots
if __name__ == "__main__":
    mpl.rcParams['font.size'] = 14
    xcoord = wl*1e9
    ycoord = n_data.iloc[100:,0]
    for key,ele in enumerate(optical_depth):
        fig, axs = plt.subplots(2,1)
        plt.suptitle(f"solar zenith angle {Xi[key]} degrees")
        plot1 = axs[0].pcolormesh(xcoord, ycoord, ele, norm=mcolors.LogNorm(vmin = 1e-2, vmax= 1e3), cmap="plasma")
        #plot2 = axs[1].pcolormesh(xcoord, ycoord, irradiance[key], norm=mcolors.LogNorm(vmin = 1e0, vmax = 1e8), cmap="plasma")
        plot2 = axs[1].pcolormesh(xcoord, ycoord, irradiance_ph[key]*1e-9, norm=mcolors.LogNorm(vmin = 1e6, vmax = 1e14), cmap="plasma")
        cbar1 = fig.colorbar(plot1)
        cbar2 = fig.colorbar(plot2)
        cbar1.set_label("optical depth")
        cbar2.set_label("Irradiance [photons $m^{-2} s^{-1} nm^{-1}$]")
        axs[0].contour(xcoord, ycoord, ele, levels=[1], linewidths=0.5)
        axs[0].set_xlabel("wavelength [nm]")
        axs[0].set_ylabel("height [km]")
        axs[1].set_xlabel("wavelength [nm]")
        axs[1].set_ylabel("height [km]")
        plt.tight_layout()
        plt.show()

    #number densities
    plt.plot(n_all[90:,0], height[90:], label="O")
    plt.plot(n_all[90:,1], height[90:], label="N2")
    plt.plot(n_all[90:,2], height[90:], label="O2")
    plt.plot(np.sum(n_all[90:,:], axis=1), height[90:], label = "all species")
    plt.xscale("log")
    plt.title("number densities in the thermosphere")
    plt.xlabel("n [cm^-3]")
    plt.ylabel("height [km]")
    plt.legend()
    plt.show()


    #wavelength variations of cross sections
    plt.plot(wl_short*1e9, cs_N2, label = "N2")
    plt.plot(wl_short*1e9, cs_O, label = "O")
    plt.plot(wl_short*1e9, cs_O2, label = "O2")

    plt.plot(wl*1e9,pol_N2, label = "N2 interpolated")
    plt.plot(wl*1e9,pol_O, label = "O interpolated")
    plt.plot(wl*1e9, pol_O2, label = "O2 interpolated")
    plt.title("absorption cross sections for different species")
    plt.xlabel("wavelength [nm]")
    plt.ylabel("absorption cross section [m^2]")
    plt.legend()
    plt.show()