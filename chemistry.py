#important reactions: with e, O+, NO+, O2+, N2+ and NO
#(10 reactions, which are on sl 234 combined notes )
#must write equation of change in density for each particle form reaction table

#coupled ODEs for each altitude
#dn/dt = P - L (sl 218)
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#load data (let everything start at altitude 100km as in iri file)
msis = pd.read_csv("atmosphere/MSIS.dat",sep=r"\s+", skiprows=17)
iri = pd.read_csv("IRI.dat", sep=r"\s+", skiprows=45)

#densities
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
###dummy values as i dont have data
n_NO_data = np.zeros_like(n_NOplus_data)
n_N2plus_data = np.zeros_like(n_N2_data)


#temperatures
T_e = iri.iloc[:,3].to_numpy()
T_i = iri.iloc[:,2].to_numpy()
T_n = msis.iloc[100:,5].to_numpy()

kB = 1.380649e-23 #[J/K]

#reaction coefficients
def reaction_coeffs(Te, Ti, Tn):

    Tr = (Tn + Ti)/2
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

    return alpha1, alpha2, alpha3, alphar, k1, k2, k3, k4, k5, k6



def initial_cond(ht):
    #gets initial densities for a certain height from the arrays with the nr densities and returns them
    n_e = n_e_data[ ht - 100 ]
    n_Oplus = n_Oplus_data[ ht - 100]
    n_O2plus = n_O2plus_data[ ht - 100]
    n_N2plus = n_N2plus_data[ ht - 100]
    n_NO = n_NO_data[ ht - 100]
    n_NOplus = n_NOplus_data[ ht - 100]

    return n_e, n_Oplus, n_O2plus, n_N2plus, n_NOplus, n_NO


def reactions(t, z, q_tot, height, temp_change):
    #ODEs for coupled cont equations 
    # calculate production and loss of major species in atmosphere
    #q is an array of all ionizatio rates, in same order as z

    #load initial conditions defined in args of solve_ivp
    z = np.maximum(0.0, z) #avoid negative densities due to numerical inaccuracies
    n_e, n_Oplus, n_O2plus, n_N2plus, n_NOplus, n_NO = z
    

    idx = height - 100
    #add neutral densities that we dont track in ODEs
    n_O = n_O_data[ idx ]
    n_O2 = n_O2_data[ idx ]
    n_N2 = n_N2_data[ idx]

    #could be outside function as this is cst !!!
    denom = (0.92 * n_N2 + n_O2 + 0.56 * n_O)
    q_e = q_tot
    q_N2plus = q_e * 0.92 * n_N2 / denom
    q_Oplus = q_e * 0.56 * n_O / denom
    q_O2plus = q_e *  n_O2 / denom
    q_NOplus = 0.0 #not produced by photo ionization
    q_NO = 0.0 #neutral, has no ionization production rate


    #reaction rates, many are dependent on temperature which is dependent on height
    alpha1, alpha2, alpha3, alphar, k1, k2, k3, k4, k5, k6 = reaction_coeffs(T_e[idx]+temp_change,T_i[idx]+temp_change, T_n[idx])
    
    #set up ODEs
    dn_e = q_e - n_e * (alpha1 * n_NOplus + alpha2 * n_O2plus + alpha3 * n_N2plus + alphar * n_Oplus)
    dn_Oplus = q_Oplus + k5 * n_O * n_N2plus - n_Oplus * (alphar * n_e + k1 * n_N2 + k2 * n_O2)
    dn_O2plus = q_O2plus + k2 * n_Oplus * n_O2 + k6 * n_N2plus * n_O2 - n_O2plus * (alpha2 * n_e + k3 * n_NO + k4 * n_N2)
    dn_N2plus = q_N2plus - n_N2plus * (alpha3 * n_e + k5 * n_O + k6 * n_O2)
    dn_NOplus = q_NOplus + k1 * n_Oplus * n_N2 + k3 * n_O2plus * n_NO + k4 * n_O2plus * n_N2 - alpha1 * n_NOplus *n_e
    dn_NO = q_NO + k4 * n_O2plus * n_N2 - k3 * n_O2plus * n_NO

    return [dn_e, dn_Oplus, dn_O2plus, dn_N2plus, dn_NOplus, dn_NO]


def ODE_solver(altitude, method, dT0, dT1, rtol):
    #divides the integration into 4 intervals of constant parameters (we have varying q and T)
    #then stacks the solutions to one get one full solution over the whole time interval
    #returns the solution for the 5 coupled equations and the time array
    q0 = 1e9
    IC = initial_cond(altitude)
    sol0 = solve_ivp(reactions, [0,3600], IC, method = method, args=(q0, altitude, dT0), rtol=rtol)

    #use previous solution as new initial conditions
    #integrate with q_e=2*1e10 for 160s, then no ionisation at all
    q1 = 2e10
    sol1 = solve_ivp(reactions, [3601, 3600+80], sol0.y[:,-1], method=method, args=(q1, altitude, dT0), rtol=rtol )

    #different ion and electron temperatures after 80s
    q2 = 2e10
    sol2 = solve_ivp(reactions, [3601+80, 3600+160], sol1.y[:,-1], method=method, args=(q2, altitude, dT1), rtol=rtol)
    
    q3 = 0
    sol3 = solve_ivp(reactions, [3601+160, 3600+400], sol2.y[:,-1], method=method, args=(q3, altitude, dT1), rtol=rtol)

    sols = np.hstack((sol0.y, sol1.y, sol2.y, sol3.y))  # Combine solutions along the state variable axis
    times = np.hstack((sol0.t, sol1.t, sol2.t, sol3.t))      #combine time arrays
    return sols, times

solutions_110, time_110 = ODE_solver(110, "Radau", 0, 0, 1e-8)
solutions_110dT, time_110dT = ODE_solver(110, "Radau", 0, 1000, 1e-8)

solutions_150, time_150 = ODE_solver(150, "Radau", 0, 0, 1e-8)
solutions_150dT, time_150dT = ODE_solver(150, "Radau", 0, 1000, 1e-8)

solutions_180, time_180 = ODE_solver(180, "Radau", 0, 0, 1e-8)
solutions_180dT, time_180dT = ODE_solver(180, "Radau", 0, 2000, 1e-8)

solutions_250, time_250 = ODE_solver(250, "Radau", 0, 0, 1e-8)
solutions_250dT, time_250dT = ODE_solver(250, "Radau", 0, 2000, 1e-8)

#charge conservation -->show in percent or in abs values?
def calc_charge(solution):
    electrons = solution[0,:]
    ions = np.sum(solution[1:5,:], axis=0)
    return ions - electrons

labels = ["e", "O+", "O2+", "N2+", "NO+", "NO"]
colours = ["blue", "red", "purple", "green", "pink", "orange" ]

def plot_odes(time1, solution1, time2, solution2, height):
    #plot the solutions for cst and for varying temperature in one plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for idx in range(len(solution1)):
        ax.plot(time1, solution1[idx][:], label = labels[idx], color = colours[idx])
        ax.plot(time2, solution2[idx][:], label = labels[idx], ls = "--", color = colours[idx])
    #ax.axvline(3600, ls="dotted", color="black")
    #ax.axvline(3680, ls="dotted", color="black")
    #ax.axvline(3760, ls="dotted", color="black")
    ax.set_xlabel("t")
    ax.legend()
    fig.suptitle(f"densities at height {height}km")

plot_odes(time_110, solutions_110, time_110dT, solutions_110dT, 110)
plot_odes(time_150, solutions_150, time_150dT, solutions_150dT, 150)
plot_odes(time_180, solutions_180, time_180dT, solutions_180dT, 180)
plot_odes(time_250, solutions_250, time_250dT, solutions_250dT, 250)
plt.show()

plt.plot(time_110, calc_charge(solutions_110))
plt.plot(time_180dT, calc_charge(solutions_180dT))
plt.title("charge conservation")
plt.show()


