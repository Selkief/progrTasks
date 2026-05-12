##calculate electron proudction rates for different solar zenith angles 
#  as function of altitude and energy (eV)
##calculate photo ionization profiles as function of altitude
##compare total ionization profiles with chapman-profile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from optical_depth import Tau, Irradiance, watts2photons, irradiance_ph, irradiance, Xi, n_all, height, wl, column_n
from atmosphere.scaleheight import H_all_gz, H_all

h = 6.62607015e-34 #plancks constant
h_ev = 4.135667e-15 #plancks constant in eV
c = 2.99792458e8 #speed of light [m/s]
e = 1.60217663e-19 #elementary charge [C]

#load photo-ionisation cross sections
phot_ion = pd.read_csv("optdepth_ionization/phot_ion.dat",sep=r"\s+", skiprows=6)
wl_short = phot_ion.iloc[:,0].to_numpy()

#interpolate photo ionisation cross angles to irradiance data
ion_cs_N2 = np.interp(wl, wl_short, phot_ion.iloc[:,1])
ion_cs_O = np.interp(wl, wl_short, phot_ion.iloc[:,2])
ion_cs_O2 = np.interp(wl, wl_short, phot_ion.iloc[:,3])

#order densities to fit the density file (O-N2-O2) and the irradiance from before
ion_cs = [ion_cs_O, ion_cs_N2, ion_cs_O2]
ion_cs_matrix = np.vstack(ion_cs)

#need to filter out the wavelengths that dont ionize
wl_th = np.array([ 91.1, 79.6, 102.6])*1e-9 #ionization threshold wavelength for the different species (m)
filtered_ion_cs = []
for idx, ele in enumerate(wl_th):
    mask = np.where(wl > ele, 0.0, 1.0)
    filtered = mask * ion_cs[idx]
    filtered_ion_cs.append(filtered)

#calculate photo electron energies --> excess ionization energy
def ph_e_energies(wavelength, wl_thr):
    E = h*c/np.abs(e) * (1/wavelength - 1/wl_thr)
    E = np.where(E<0, 0, E)
    return E

#plot the transformation as test: wl --> E(eV)
energies = []
for j in wl_th:   
    energies.append(ph_e_energies(wl, j))

plt.plot(wl*1e9, energies[0], label="O")
plt.plot(wl*1e9, energies[1], label="N2")
plt.plot(wl*1e9, energies[2], label="O2")
plt.ylim(0,150)
plt.title("photon electron energies for different wavelengths")
plt.ylabel("energy eV")
plt.xlabel("wavelength nm")
plt.legend()
plt.grid()
plt.show()


##calculate total photo ionization rate and photo electrons 
def photo_ion_rate_matrix(densities, ion_cs, EUVflux):
    #total photo ionization rate
    q_total = np.zeros(EUVflux.shape[0])
    #total photo electrons 
    P_total = np.zeros_like(EUVflux)

    for idx, cs_j in enumerate(ion_cs):
        n_z = densities[100:, idx][:,None]
        sigma = cs_j[None, :]

        dq_j = n_z * EUVflux * sigma
        
        q_j = np.trapezoid(dq_j, wl, axis=1)
        q_total += q_j
        
        E_eV = ph_e_energies(wl, wl_th[idx])
        #reorder from low energies to high energies as new xcoord, use next lower integer as value
        idx_sorted = np.argsort(E_eV)
        E_eV = np.floor(E_eV[idx_sorted])
        dq_sorted = dq_j[:, idx_sorted]

        P_total += dq_sorted
        
    return P_total, q_total, E_eV

#calculate photoionization, peak ionization and photo electrons for all SZAs
total_photoion = []
photo_electrons = []
max_ionisation = []
for X in range(len(Xi)):
    P, q, new_x = photo_ion_rate_matrix(n_all, filtered_ion_cs, irradiance_ph[X])
    total_photoion.append(q) 
    photo_electrons.append(P)
    max_index = int(np.argmax(q))
    max_value = float(q[max_index])
    max_ionisation.append([max_value, max_index+100])


#calculate chapman profiles from peak ionization values, assume constant scale height ?
def chapman(peak_ion, scaleheight, z, SZA):
    x = (z - peak_ion[1])/(scaleheight)
    return peak_ion[0] * np.exp( 1 - x - 1/np.cos(np.deg2rad(SZA)) * np.exp(- x ) )

chapman_profiles = []
for idx, X in enumerate(Xi):
    ch_p = chapman(max_ionisation[0], H_all_gz[100:], height[101:], X)
    chapman_profiles.append(ch_p)


#make plots
if __name__ == "__main__":
    fig, axs = plt.subplots(1,2)
    ycoord = height[100:]
    for key,ele in enumerate(total_photoion):
        axs[0].plot(ele, ycoord, label = f"$\chi$ = {Xi[key]}")
        axs[0].scatter(max_ionisation[key][0], max_ionisation[key][1])
    axs[0].set_xlabel("ionization rate [$m^{-3}$ $s^{-1}$]")
    axs[0].set_ylabel("height [km]")
    axs[0].set_xscale("log")
    axs[0].set_title("calculated profiles")
    axs[0].set_xlim(1e5,1e10)
    axs[0].grid()

    ycoord2 = height[101:]
    for key,ele in enumerate(chapman_profiles):
        axs[1].plot(ele, ycoord2, label = f"$\chi$ = {Xi[key]}")
        axs[1].scatter(max_ionisation[key][0], max_ionisation[key][1])
    axs[1].set_title("Chapman profiles")
    axs[1].set_xlabel("ionization rate [$m^{-3}$ $s^{-1}$]")
    axs[1].set_ylabel("height [km]")
    axs[1].set_xscale("log")
    axs[1].set_xlim(1e5,1e10)
    axs[1].grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    ycoord = height[100:]
    for key,ele in enumerate(photo_electrons):
        fig, axs = plt.subplots(1,1)
        plt.suptitle(f"solar zenith angle {Xi[key]} degrees")
        plot1 = axs.pcolormesh(new_x, ycoord, ele, norm=mcolors.LogNorm(vmin = 1e12, vmax = 1e18), cmap="jet")
        cbar1 = fig.colorbar(plot1)
        cbar1.set_label("photon electrons [$m^{-2}s^{-1}$]")
        axs.set_xlabel("Energy [eV]")
        axs.set_ylabel("height [km]")
        axs.set_xlim(0, 150)
        plt.tight_layout()
        plt.show()

    plt.plot(wl*1e9, ion_cs_N2, label = "N2")
    plt.plot(wl*1e9, ion_cs_O, label="O")
    plt.plot(wl*1e9, ion_cs_O2, label = "O2")
    plt.xlabel("wavelength [nm]")
    plt.title("ionisation cross sections")
    plt.legend()
    #plt.show()


