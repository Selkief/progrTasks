#important reactions: with e, O+, NO+, O2+, N2+ and NO
#(10 reactions, which are on sl 234 combined notes )
#must write equation of change in density for each particle form reaction table

#coupled ODEs for each altitude
#dn/dt = P - L (sl 218)
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#load data with all the densities
msis = pd.read_csv("atmosphere/MSIS.dat",sep=r"\s+", skiprows=17)
iri = pd.read_csv("IRI.dat", sep=r"\s+", skiprows=45)

n_e_data = iri.iloc[:,1].to_numpy() #density in m^-3
n_O_data = msis.iloc[100:,1].to_numpy()*1e6 #density in m^-3
frac_Oplus = iri.iloc[:,4].to_numpy()/100
n_Oplus_data = frac_Oplus * n_e_data #density m^-3
n_O2_data = msis.iloc[100:,3].to_numpy()*1e6 #density m^-3
frac_O2plus = iri.iloc[:,7].to_numpy()/100
n_O2plus_data = frac_O2plus * n_e_data #density m^-3
n_N2_data = msis.iloc[100:,2].to_numpy()*1e6 #density m^-3
frac_NOplus = iri.iloc[:,8].to_numpy()/100
n_NOplus_data = frac_NOplus * n_e_data #density m^-3
###dummy values
n_NO_data = np.zeros_like(n_NOplus_data)
n_N2plus_data = np.zeros_like(n_N2_data)


#temperatures
Te = iri.iloc[:,3].to_numpy()
Ti = iri.iloc[:,2].to_numpy()
Tn = msis.iloc[100:,5].to_numpy()
Tr = (Tn + Ti)/2

kB = 1.380649e-23 #[J/K]
#reaction coefficients
alpha1 = 4.2e-13 * np.power((Te/300), -0.85)
alpha2 = 1.9e-13 * np.power((Te/300), -0.5)
alpha3 = 1.8e-13 * np.power((Te/300), -0.39)
alphar = 3.7e-18 * np.power((Te/250), -0.7)
k1 = 1.3e-18 * np.power((Tr/300),-0.5) + 6.8e-16 * np.exp(-1.4e-19/(kB*Tr))
k2 = 2e-17 * np.power((Tr/300), -0.4)
k3 = 4.4e-16
k4 = 5e-22
k5 = 1.4e-16 * np.power((Tr/300), -0.44)
k6 = 5e-17 * np.power((Tr/300), -0.8)



def initial_cond(ht):
    #gets initial densities for a certain height from the arrays with the nr densities and returns them
    n_e = n_e_data[ ht - 100 ]
    n_Oplus = n_Oplus_data[ ht - 100]
    n_O2plus = n_O2plus_data[ ht - 100]
    n_N2plus = n_N2plus_data[ ht - 100]
    n_NO = n_NO_data[ ht - 100]
    n_NOplus = n_NOplus_data[ ht - 100]

    return n_e, n_Oplus, n_O2plus, n_N2plus, n_NOplus, n_NO


#change this to adjust to changes in densities during solving ODEs? sth wrong with q_N2plus!?
def calc_q(ionization_rate, h, Oplus, O2plus, NOplus):
    #calculates ionisation rate for all the ion species and the electrons from total ionization rate
    #returns array with rate for [q_e, q_Oplus, q_O2plus, q_NOplus, q_NO]
    idx = h-100
    q_tot = ionization_rate
    q_e = q_tot
    q_Oplus = Oplus * q_tot
    q_O2plus = O2plus * q_tot
    q_NOplus = NOplus * q_tot
    q_N2plus = q_e - (q_Oplus + q_O2plus + q_NOplus)
    #q_N2plus = q_e * 0.92 * n_N2_data[idx] / (0.92 * n_N2_data[idx] + n_O2_data[idx] + 0.56 * n_O_data[idx])
    q_NO = 0.0 #neutral, doesn't have ionization rate
    
    return [q_e, q_Oplus, q_O2plus, q_N2plus, q_NOplus, q_NO]


def reactions(t, z, q, height):
    #ODEs for coupled cont equations 
    # that calculate production and loss of major species in atmosphere
    #q is an array of all ionizatio rates, in same order as z

    #load initial conditions defined in args of solve_ivp
    n_e, n_Oplus, n_O2plus, n_N2plus, n_NOplus, n_NO = z
    q_e, q_Oplus, q_O2plus, q_N2plus, q_NOplus, q_NO = q

    idx = height - 100

    #add neutral densities that we dont track in ODEs
    n_O = n_O_data[ idx ]
    n_O2 = n_O2_data[ idx ]
    n_N2 = n_N2_data[ idx]

    #reaction rates, many are dependent on temperature which is dependent on height
    alpha_1 = alpha1[idx]
    alpha_2 = alpha2[idx]
    alpha_3 = alpha3[idx]
    alpha_r = alphar[idx]
    k_1 = k1[idx]
    k_2 = k2[idx]
    k_3 = k3
    k_4 = k4
    k_5 = k5[idx]
    k_6 = k6[idx]
    
    #set up ODEs
    dn_e = q_e - n_e * (alpha_1 * n_NOplus + alpha_2 * n_O2plus + alpha_3 * n_N2plus + alpha_r * n_Oplus)
    dn_Oplus = q_Oplus + k_5 * n_O * n_N2plus - n_Oplus * (alpha_r * n_e + k_1 * n_N2 + k_2 * n_O2)
    dn_O2plus = q_O2plus + k_2 * n_Oplus * n_O2 + k_6 * n_N2plus * n_O2 - n_O2plus * (alpha_2 * n_e + k_3 * n_NO + k_4 * n_N2)
    dn_N2plus = q_N2plus - n_N2plus * (alpha_3 * n_e + k_5 * n_O + k_6 * n_O2)
    dn_NOplus = q_NOplus + k_1 * n_Oplus * n_N2 + k_3 * n_O2plus * n_NO + k_4 * n_O2plus * n_N2 - alpha_1 * n_NOplus *n_e
    dn_NO = q_NO + k_4 * n_O2plus * n_N2 - k_3 * n_O2plus * n_NO

    return [dn_e, dn_Oplus, dn_O2plus, dn_N2plus, dn_NOplus, dn_NO]


def ODE_solver(altitude, method):
    #integrate for 3600s with constant ionizatio rate of 1e9 (height 110km)
    q0 = calc_q(1e9, altitude, frac_Oplus[altitude-100], frac_O2plus[altitude-100], frac_NOplus[altitude-100])
    IC = initial_cond(altitude)
    sol0 = solve_ivp(reactions, [0,3600], IC, method = method, args=(q0, altitude))

    #use previous solution as new initial conditions
    #integrate with q_e=2*1e10 for 160s, then no ionisation at all
    Oplus = sol0.y[1,-1]/sol0.y[0,-1]
    O2plus = sol0.y[2,-1]/sol0.y[0,-1]
    NOplus = sol0.y[4,-1]/sol0.y[0,-1]
    q1 = calc_q(2e10, altitude, Oplus, O2plus, NOplus)
    sol1 = solve_ivp(reactions, [3600, 3600+160], sol0.y[:,-1], method=method, args=(q1, altitude) )

    Oplus = sol1.y[1,-1]/sol1.y[0,-1]
    O2plus = sol1.y[2,-1]/sol1.y[0,-1]
    NOplus = sol1.y[4,-1]/sol1.y[0,-1]
    q2 = calc_q(0.0, altitude, Oplus, O2plus, NOplus)
    sol2 = solve_ivp(reactions, [3600+160, 3600+400], sol1.y[:,-1], method=method, args=(q2, altitude) )

    sols = np.hstack((sol0.y, sol1.y, sol2.y))  # Combine solutions along the state variable axis
    times = np.hstack((sol0.t, sol1.t, sol2.t))      #combine time arrays
    return sols, times

solutions_250, time_250 = ODE_solver(250, "BDF")

#charge conservation -->show in percent or in abs values?
electrons = solutions_250[0,:]
ions = np.sum(solutions_250[1:5,:], axis=0)
charge = ions - electrons

labels = ["e", "O+", "O2+", "N2+", "NO+", "NO"]
for idx, ele in enumerate(solutions_250):
    plt.plot(time_250, ele[:], label = labels[idx])
plt.xlabel("t")
plt.legend()
plt.suptitle("densities at height 250km")
plt.tight_layout()
plt.show()

plt.plot(time_250, charge)
plt.title("charge conservation")
plt.show()


