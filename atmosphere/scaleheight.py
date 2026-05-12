import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("atmosphere/MSIS.dat",sep=r"\s+", skiprows=16)

##constants
k_B = 1.38e-23     #[m^2*kg/(K*s^2)] boltzmanns constant
g0 = 9.82 #[m/s] gravitational acceleration
R_E = 6378e3 #[m] earth radius
p0 = 101325 #[Pa] std pressure at sealevel
#O weighs 16amu or g/mol
#N2 weighs 28amu or g/mol
#O2 weighs 32amu or g/mol

##define functions
#calculate scaleheight --> H= k_B*T/(m*g)
#scale height for all species
def calc_Hall(data_file, height, g):
     #nr density all molecules
    n = data_file.iloc[height,1]+ data_file.iloc[height,2]+ data_file.iloc[height,3]
    #average molecule weight
    m = (data_file.iloc[height,1]*16 + data_file.iloc[height,2]*28 + data_file.iloc[height,3]*32)/n *1.66054 * 10e-28 #[kg]
    #constant g or z-dependent g
    if g =="g_z":
        g = g0 * R_E**2 /(R_E + height*1e3)**2
    else:
        g = g0
    #scale height
    H = (k_B * data_file.iloc[height,5]) / (m * g)
    return H

#scale height for single species, cst g or varying g
def calc_Hsingle(data_file, height, g, species):
    if species == "O":
        m = 16 * 1.66054e-27
    elif species == "N2":
        m = 28 * 1.66054e-27
    elif species == "O2":
        m = 32 * 1.66054e-27
    
    if g =="g_z":
        g = g0 * R_E**2 /(R_E + height*1e3)**2
    else:
        g = g0
    H = (k_B * data_file.iloc[height, 5]) / (m * g)
    return H

#experimental scale height H_x = -n_x/(dn_x/dz)
def exp_H(data_file, species):
    if species == "O":
        column = 1
    elif species == "N2":
        column = 2
    elif species == "O2":
        column = 3
    H = []
    for i in range(1,600):
        Hz = (-data_file.iloc[i,column])/((data_file.iloc[i, column]-data_file.iloc[i-1, column])/10e2)
        if Hz>=0:
            H.append(float(Hz))
    return H


##calculate results
#scaleheight for all species at 0km and 10km
H_0 = calc_Hall(data, 0, g0)
H_10 = calc_Hall(data, 10, g0)

#scaleheight for single species at 120km and 600km
#constant g:
H_120_O = calc_Hsingle(data, 120, g0, "O")
H_120_N2 = calc_Hsingle(data, 120, g0, "N2")
H_120_O2 = calc_Hsingle(data, 120, g0, "O2")
H_600_O = calc_Hsingle(data, 600, g0, "O")
H_600_N2 = calc_Hsingle(data, 600, g0, "N2")
H_600_O2 = calc_Hsingle(data, 600, g0, "O2")
H_cstg =[H_120_O, H_120_N2, H_120_O2, H_600_O, H_600_N2, H_600_O2]
#non-constant g:
H_120_gz_O = calc_Hsingle(data, 120, "g_z", "O")
H_120_gz_N2 = calc_Hsingle(data, 120, "g_z", "N2")
H_120_gz_O2 = calc_Hsingle(data, 120, "g_z", "O2")
H_600_gz_O = calc_Hsingle(data, 600, "g_z", "O")
H_600_gz_N2 = calc_Hsingle(data, 600, "g_z", "N2")
H_600_gz_O2 = calc_Hsingle(data, 600, "g_z", "O2")
H_gz = [H_120_gz_O, H_120_gz_N2, H_120_gz_O2, H_600_gz_O, H_600_gz_N2, H_600_gz_O2]

#difference between scaleheight with constant g and varying g (dependend on z)
diff = []
for i in range(6):
   diff.append(float(H_gz[i] - H_cstg[i]))

#exp scaleheight: 
exp_H_O = np.array(exp_H(data, "O"))
#print(len(exp_H_O))
exp_H_N2 = np.array(exp_H(data, "N2"))
#print(len(exp_H_N2))
exp_H_O2 = np.array(exp_H(data, "O2"))
#print(len(exp_H_O2))

if __name__ == "__main__":
    #print solutions for part 1
    print(f"scale height at 0km:{H_0:.2f}m, at 10km: {H_10:.2f}m")
    print(f"scale height 120km cst g for O {H_120_O:.2f}m for N2 {H_120_N2:.2f}m for O2 {H_120_O2:.2f}m")
    print(f"scale height 600km cst g for O {H_600_O:.2f}m for N2 {H_600_N2:.2f}m for O2 {H_600_O2:.2f}m")
    print(f"scale height 120km non-cst g for O {H_120_gz_O:.2f}m for N2 {H_120_gz_N2:.2f}m for O2 {H_120_gz_O2:.2f}m")
    print(f"scale height 600km non-cst g for O {H_600_gz_O:.2f}m for N2 {H_600_gz_N2:.2f}m for O2 {H_600_gz_O2:.2f}m")
    print(f"difference between constant and non-constant g (in m):{diff}")

#print(f"experimental scaleheight at 120km for O {exp_H_O[120-2]:.2f}m for N2 {exp_H_N2[120-2]:.2f}m and O2 {exp_H_O2[120-2]:.2f}m.")
#print(f"experimental scaleheight at 600km for O {exp_H_O[600-2]:.2f}m and for N2 {exp_H_N2[600-2]:.2f}m and for O2 {exp_H_O2[600-2]:.2f}m.")

#part 2: altitude variations of scaleheight from 0-600km
H_all = []
H_all_gz = []
H_O = []
H_N2 = []
H_O2 = []

for k in range(600):
    #all species cst g
    H_all.append(float(calc_Hall(data, k, g0))*10e-4)
    #all species varying g
    H_all_gz.append(float(calc_Hall(data, k, "gz"))*10e-4)
    #single species cst g
    H_O.append(float(calc_Hsingle(data, k, g0, "O"))*10e-4)
    H_N2.append(float(calc_Hsingle(data, k, g0, "N2"))*10e-4)
    H_O2.append((float(calc_Hsingle(data, k, g0, "O2"))*10e-4))

##part 3: pressure variations
#numerical integration
p_num = [0 for _ in range(len(H_all))]
p_num[0] = p0
dz = 1
for i in range(1,len(H_all)):
    p_num[i] = p_num[i-1] - (p_num[i-1]/H_all_gz[i-1])*(dz)
p_num = np.array(p_num)

#(half) analytical solution (p = p0*exp(-integral(1/H)dz)
H_num = [0 for _ in range(len(H_all))]
H_num[0] = 1/H_all_gz[0]
for i in range(1, len(H_all)):
    H_num[i] = H_num[i-1] + 1/(H_all_gz[i-1])*dz
p_analytic = p0 * np.exp(-np.array(H_num))

#ideal gas law p = n*k_B*T where n = N/V
p_gasslaw = []
for i in range(600):
    n = (data.iloc[i,1]+data.iloc[i,2]+data.iloc[i,3])*1e6
    p_gasslaw.append(n*k_B*data.iloc[i,5])
p_gasslaw = np.array(p_gasslaw)

#part 3: temperature gradient and adiabatic lapse rate
grad_T = []
for i in range(1,600):
    grad_T.append(data.iloc[i,5] - data.iloc[i-1, 5])
grad_T = np.array(grad_T)

if __name__ == "__main__":
    #plot results
    plt.rcParams.update({'font.size': 14})
    #scaleheight variations for different species
    z = np.arange(600)
    fig, axs = plt.subplots(1,2)
    axs[0].plot(H_all,z)
    axs[0].grid(True)
    axs[0].set_title("scaleheight for all species")
    axs[0].set_xlabel("scaleheight in km")
    axs[0].set_ylabel("height in km")
    axs[1].plot(H_O, z)
    axs[1].grid(True)
    axs[1].set_title("scaleheight for O")
    axs[1].set_xlabel("scaleheight in km")
    axs[1].set_ylabel("height in km")
    plt.tight_layout()


    fig, axs = plt.subplots(1,2)
    axs[0].plot(H_N2,z)
    axs[0].grid(True)
    axs[0].set_title("scaleheight for N2")
    axs[0].set_xlabel("scaleheight in km")
    axs[0].set_ylabel("height in km")
    axs[1].plot(H_O2, z)
    axs[1].grid(True)
    axs[1].set_title("scaleheight for O2")
    axs[1].set_xlabel("scaleheight in km")
    axs[1].set_ylabel("height in km")
    plt.tight_layout()
    plt.show()

    print(len(exp_H_O))
    plt.scatter(exp_H_O*1e-3, z[99:], label="experimental scaleheight for O", s=10)
    plt.scatter(exp_H_N2*1e-3, z[:-1], label="experimental scaleheight for N2", s=10)
    plt.scatter(exp_H_O2*1e-3, z[:-1], label="experimental scaleheight for O2", s=10)
    plt.xlabel("scaleheight in km")
    plt.ylabel("height in km")
    plt.grid(True)
    plt.legend()
    plt.show()

    #pressure variations numerical, analytic and gass law
    plt.plot(p_num*1e-3, z, label="numerical solution")
    plt.plot(p_analytic*1e-3, z, label = "half analytical solution")
    plt.plot(p_gasslaw*1e-3, z, label="gass law")
    plt.grid(True)
    plt.title("pressure variation")
    plt.xlabel("pressure [kPa]")
    plt.ylabel("height (km)")
    plt.legend()
    plt.show()

    #differences between pressure variations
    plt.plot((p_num-p_analytic)*1e-3, label="p numerical vs analytical")
    plt.plot((p_analytic - p_gasslaw)*1e-3, label="p analytical vs gass law")
    plt.ylabel("difference (kPa)")
    plt.xlabel("height")
    plt.title("difference between solutions p variations")
    plt.grid()
    plt.legend()
    plt.show()

    #temperature gradient
    plt.plot(grad_T, z[:-1], label="dT/dz")
    plt.axvline(x=9.8, color = "green", label="dry adiabat (9.8K/km)")
    plt.ylabel("height (km)")
    plt.xlabel("lapse rate (K/km)")
    plt.grid()
    plt.legend()
    plt.show()