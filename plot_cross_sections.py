import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import pandas as pd
from nucres.resonance import Resonance, sigma_bw_energy_dep
from nucres.sampling import porter_thomas_factors
from nucres.physics import *

output_dir = 'outputs/22Mg(a,p)25Al/'
file = output_dir +'RUN_0/22Mg(a,p)25Al.in'
#bin_width = [0.1,0.2]
start_marker = 'Resonant Contribution'
end_marker = 'Upper Limits of Resonances'

def extract_data(file):

    extracted_lines = []
    capture = False

    with open(file,'r') as f:

        for line in f:
            if start_marker in line:
                capture=True
                skip_count=0
                continue

            if end_marker in line and capture:
                break


            if capture:

                if skip_count < 2:
                    skip_count += 1
                    continue

                extracted_lines.append(line)

    if extracted_lines:
        extracted_lines.pop()

    block = "".join(extracted_lines)

    df = pd.read_csv(StringIO(block),sep=r'\s+',header=0)

    return df



df = extract_data(file)
df = df.sort_values(by='Ecm',ascending=True) # sort by increasing energy; Er in keV
E_cm = df['Ecm'].values*1e-3 # convert Er to MeV
g1 = df['G1'].values # in eV
g2 = df['G2'].values # in eV
g3 = df['G3'].values # in eV
Jr = df['Jr'].values # spin (dimensionless)
l1 = df['L1'].values
l2 = df['L2'].values
L = l1 + l2
G = g1 + g2 + g3

theta2 = 0.01 # dimensionless; for alpha induced reactions
x = porter_thomas_factors(len(E_cm),df=1)
t2_dist = theta2* x  # apply Porter-Thomas fluctuations to Gamma_i
A1 = 22
A2 = 4
mu = reduced_mass(A1*MASS_PROTON, A2*MASS_PROTON) #kg
R_sq = 1.25e-15**2 * (A1**(1/3) + A2**(1/3))**2  # in m^2
wigner_limit = 3*HBAR**2/(2*mu*R_sq)*6.242e18 # in eV
gamma2_dist = t2_dist * wigner_limit  # in eV
#print(gamma2_dist)

def calc_cross_sections(E,Z1,Z2,A1,A2):
    
    E_arr = np.asarray(E,dtype=float)

    xs_tot = np.zeros_like(E)
    for j in range(len(E_cm)):

        r_j = Resonance(E_cm[j]*10**6,      # resonance energy Er in eV; MeV -> eV
            Jr[j],
            0, 0,
            3.65e-26, #kg
            6.64e-27, #kg
            g1[j],        # Gamma_i at Er (from table)
            g2[j],        # Gamma_o (from table)
        )

        xs_j = sigma_bw_energy_dep(
            E_arr*10**6, #eV
            r_j,
            Z1, Z2, A1, A2 ,
            L[j],
            gamma2=gamma2_dist[j] # in eV
        )
        xs_tot += xs_j # summing contributions from all resonances


    return xs_tot

E_test= np.linspace(0.1,10,10000) #MeV

xs_unint = calc_cross_sections(E_test,12,2,22,4)


xs_lst = []

# Read inputs as strings first
E_min_in = input("Enter minimum energy [MeV] for integration (press Enter for default): ")
E_min = float(E_min_in) if E_min_in.strip() else np.min(E_cm)
E_min = np.round(E_min,2)
print(E_min)

E_max_in = input("Enter maximum energy [MeV] for integration (press Enter for default): ")
E_max = float(E_max_in) if E_max_in.strip() else np.max(E_cm)
E_max = np.round(E_max,2)
print(E_max)

dE_in    = input("Enter bin width(s) [MeV] separated by commas: ")
dE_lst = [float(x.strip()) for x in dE_in.split(",")]

fig = plt.figure(figsize=(8,6))
m = 0
for dE in dE_lst:
    print(f"Running for dE = {dE} MeV")
    xs_bin = []

    N_bins = 1000  # number of internal bins for integration

    E_bins = np.arange(E_min,E_max+dE,dE)

    E_cm = np.round(E_cm,2)

    for E0 in E_bins:
         E1 = E0 + dE
    # # internal grid (MeV or eV consistently)
         E_int = np.linspace(E0, E1, N_bins)
         
         xs_total = calc_cross_sections(E_int,12,2,22,4)
         
         xs_int = np.trapezoid(xs_total, E_int) # MeV*barns
         xs_avg = xs_int / dE  # average cross-section in the bin (barns)
        #print(f"Integrated cross-section over {E_min} to {E_max} MeV with dE={dE} MeV: {xs_int} MeV*barns")
         xs_bin.append(xs_avg)
    print("*************************************************************")

    
    xs_bin = np.array(xs_bin)
    
    filename = f'22Mg_ap_25Al_xs_{m}.txt'

    with open(filename,'w') as f:
        f.write(f'#E_cm (MeV),xs_bin (mb) | bin width = {dE} MeV\n')
        for k in range(len(E_bins)):
            f.write(f'{E_bins[k]:.2f}, {xs_bin[k]*1e3}\n')
    
    m += 1

    #plt.scatter(E_bins,xs_bin*1e3,s=20,label=rf"$\Delta$E = {dE} MeV")
    plt.plot(E_bins,xs_bin*1e3,label=rf"$\Delta$E = {dE} MeV")
    


#plt.scatter(E_test,xs_unint*1e3,color='k',label = 'Before integration')
plt.plot(E_test,xs_unint*1e3,color='k',label = 'Before integration')
plt.yscale('log')
plt.title('22Mg(a,p)25Al')
plt.legend()
plt.xlabel('MeV')
plt.ylabel('mb')

plt.savefig('22Mg_ap_25Al.png')

plt.show()
