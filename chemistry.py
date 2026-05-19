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


def reactions(t, z, q_func, height, temp_change):
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

    if callable(q_func):
        q_e = q_func(t)
    else:
        q_e = q_func
    denom = (0.92 * n_N2 + n_O2 + 0.56 * n_O)
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
    sol0 = solve_ivp(reactions, [0,3600], IC, method = method, args=(q0, altitude, dT0), rtol=rtol, t_eval=np.arange(0,3600,1))

    #use previous solution as new initial conditions
    #integrate with q_e=2*1e10 for 160s, then no ionisation at all
    q1 = 2e10
    sol1 = solve_ivp(reactions, [3601, 3600+80], sol0.y[:,-1], method=method, args=(q1, altitude, dT0), rtol=rtol, t_eval=np.arange(3601,3680,1))

    #different ion and electron temperatures after 80s
    q2 = 2e10
    sol2 = solve_ivp(reactions, [3601+80, 3600+160], sol1.y[:,-1], method=method, args=(q2, altitude, dT1), rtol=rtol, t_eval=np.arange(3681,3760,1))
    
    q3 = 0
    sol3 = solve_ivp(reactions, [3601+160, 3600+760], sol2.y[:,-1], method=method, args=(q3, altitude, dT1), rtol=rtol, t_eval=np.arange(3761, 3600+760,1))

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

def varying_q(t):
    q_hat = 2e10 #[m^-3s^-1]
    return q_hat * np.sin(2*np.pi*t/20)**2

t_test = np.arange(0,600,1)
q_t = varying_q(t_test)
plt.plot(t_test, q_t)
plt.title("$q_e(t)=\\hat q_e sin^2(2 \\pi t/20)$")
plt.show()

def ODE_solver2(altitude, method, rtol):
    
    q0 = varying_q
    IC = initial_cond(altitude)
    sol0 = solve_ivp(reactions, [0,100], IC, method = method, args=(q0, altitude, 0), rtol=rtol, t_eval=np.arange(0,100,1))

    q1 = 0.0
    sol1 = solve_ivp(reactions, [101, 600], sol0.y[:,-1], method=method, args=(q1, altitude, 0), rtol=rtol, t_eval=np.arange(101,600,1))

    sols = np.hstack((sol0.y, sol1.y))  # Combine solutions along the state variable axis
    times = np.hstack((sol0.t, sol1.t))      #combine time arrays
    return sols, times

varyingq_110, vartimes_110 = ODE_solver2(110, "Radau", 1e-8)
varyingq_150, vartimes_150 = ODE_solver2(150, "Radau", 1e-8)
varyingq_180, vartimes_180 = ODE_solver2(180, "Radau", 1e-8)
varyingq_250, vartimes_250 = ODE_solver2(250, "Radau", 1e-8)

def decay(region, t):
    #calculates expected decay after ionization is "turned off" at t=3760
    #input: reaction rates of the corresponding height/temperature
    #densities before the ionization stops as 
    global Te, Ti, Tn, solutions_110, time_110, solutions_250, time_250
    if region == "E":
        idx = 110 - 100
        n = solutions_110
        t0 = np.argmax(n[0])
        print(n.shape)
        n_e0 = n[0][t0]
        n_NOplus = n[4][t0]
        n_O2plus = n[2][t0]
        n_N2plus = n[3][t0]
        coeffs = reaction_coeffs(T_e[idx], T_i[idx], T_n[idx])
        alpha_1, alpha_2, alpha_3 = coeffs[0], coeffs[1], coeffs[2]
        alpha_e = alpha_1 * n_NOplus/n_e0 + alpha_2 * n_O2plus/n_e0 + alpha_3 * n_N2plus/n_e0
        n_e = n_e0 / (1 + alpha_e * n_e0 * t)
    elif region == "F":
        idx = 250 - 100
        n = solutions_250
        t0 = np.argmax(n[0])
        n_e0 = n[0][t0]
        n_NOplus = n[4][t0]
        n_O2plus = n[2][t0]
        n_O2 = n_O2_data[idx]
        n_N2 = n_N2_data[idx]
        coeffs = reaction_coeffs(T_e[idx], T_i[idx], T_n[idx])
        alpha_1, alpha_2, k_1, k_2 = coeffs[0], coeffs[1], coeffs[4], coeffs[5]
        #include O2plus NOplus diss recomb?
        beta = k_2 * n_O2 + k_1 * n_N2 + alpha_1 * n_NOplus/n_e0 + alpha_2 * n_O2plus/n_e0
        n_e = n_e0 * np.exp(-beta*t)  
    return n_e  

t = np.arange(0, 600,1)
decay_E = decay("E", t)
decay_F = decay("F", t)

#charge conservation -->show in percent or in abs values?
def calc_charge(solution):
    electrons = solution[0,:]
    ions = np.sum(solution[1:5,:], axis=0)
    return ions - electrons


def plot_odes(time1, solution1, time2=None, solution2=None, height=None, temp_var=None):
    labels = ["e", "O+", "O2+", "N2+", "NO+", "NO"]
    colours = ["blue", "red", "purple", "green", "pink", "orange" ]
    #plot the solutions for cst and for varying temperature in one plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for idx in range(len(solution1)):
        ax.plot(time1, solution1[idx][:], label = labels[idx], color = colours[idx])
        if temp_var == "yes":
            ax.plot(time2, solution2[idx][:], ls = "--", color = colours[idx])
            ax.set_xlim(3500,3600+760)
    #ax.axvline(3600, ls="dotted", color="black")
    #ax.axvline(3680, ls="dotted", color="black")
    #ax.axvline(3760, ls="dotted", color="black")
    ax.set_xlabel("t")
    ax.set_ylabel("density [$m^{-3}$]")
    ax.legend()
    fig.suptitle(f"densities at height {height}km")

plot_odes(time_110, solutions_110, time_110dT, solutions_110dT, 110, "yes")
plot_odes(time_150, solutions_150, time_150dT, solutions_150dT, 150, "yes")
plot_odes(time_180, solutions_180, time_180dT, solutions_180dT, 180, "yes")
plot_odes(time_250, solutions_250, time_250dT, solutions_250dT, 250, "yes")
plt.show()

plot_odes(vartimes_110, varyingq_110, height=110)
plot_odes(vartimes_150, varyingq_150, height=150)
plot_odes(vartimes_180, varyingq_180, height=180)
plot_odes(vartimes_250, varyingq_250, height=250)
plt.show()


plt.plot(time_110[3700:], solutions_110[0][3700:], label="modelled electrons")
plt.plot(3760 + t, decay_E, label="$n_e(0) /(1+\\alpha_e n_e(0) t)  $")
plt.xlabel("t")
#plt.yticks(np.arange(0, 5, 1)*1e11)
plt.ylabel("densities")
plt.legend()
plt.grid()
plt.title("electron densities without ionization at 110km")
plt.show()

plt.plot(time_250[3700:], solutions_250[0][3700:], label="modelled electrons")
plt.plot(3760 + t, decay_F, label="$n_e(0) e^{-\\beta t}$")
plt.xlabel("t")
#plt.yticks(np.arange(0,3.0, 0.5)*1e12)
plt.ylabel("densities")
plt.legend()
plt.grid()
plt.title("electron densities without ionization at 250km")
plt.show()


plt.plot(time_110, calc_charge(solutions_110))
plt.plot(time_180dT, calc_charge(solutions_180dT))
plt.title("charge conservation")
#plt.show()


