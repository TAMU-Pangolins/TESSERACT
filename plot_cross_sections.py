import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import pandas as pd
from nucres.resonance import Resonance, sigma_bw_energy_dep
from scipy.interpolate import make_interp_spline

#resonance_i = Resonance(1,0.5,0.5,0.5,3.65*10**(-26),6.64*10**(-27),1,1)
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
df = df.sort_values(by='Ecm',ascending=True)
E_cm = df['Ecm']
gamma_i = df['G1']
gamma_o = df['G2']
Jr = df['Jr']
l1 = df['L1']
l2 = df['L2']
L = l1 + l2


xs_lst = []

# Read inputs as strings first
E_min_in = input("Enter minimum energy [MeV] for integration (press Enter for default): ")
E_min = float(E_min_in) if E_min_in.strip() else np.min(E_cm*10**(-3))
print(E_min)

E_max_in = input("Enter maximum energy [MeV] for integration (press Enter for default): ")
E_max = float(E_max_in) if E_max_in.strip() else np.max(E_cm*10**(-3))
print(E_max)

dE_in    = input("Enter bin width [MeV] for integration: ")
dE    = float(dE_in)

xs_bin = []

N_bins = 50

E_bins = np.arange(E_min,E_max+dE,dE)

for E0 in E_bins:
    E1 = E0 + dE

    # internal grid (MeV or eV consistently)
    E_int = np.linspace(E0, E1, N_bins)

    xs_total = np.zeros_like(E_int)

    # sum over resonances
    for i in range(len(E_cm)):
        r_i = Resonance(
            E_cm[i]*10**3,      # resonance energy Er in eV
            Jr[i],
            0, 0,
            3.65e-26,
            6.64e-27,
            gamma_i[i],        # Gamma_i at Er (from table)
            gamma_o[i],        # Gamma_o (from table)
        )

        xs_r = sigma_bw_energy_dep(
            E_int*10**6, #eV
            r_i,
            12, 2, 22, 4,
            L[i],
            gamma2_mev=0.0   # not used since Gamma_i is known
        )

        xs_total += xs_r

    # integrate over the bin
    xs_int = np.trapezoid(xs_total, E_int)
    xs_bin.append(xs_int)
    print(E0)

#E_lst = np.arange(E_min,E_max+dE,dE)

#for i in range(len(E_cm)):
    # extracting resonance information
#    r_i = Resonance(E_cm[i],Jr[i],0,0,3.65*10**(-26),6.64*10**(-27),gamma_i[i],gamma_o[i])
    # generating cross sections
#    xs = sigma_bw_energy_dep([E_cm[i]],r_i,12,2,22,4,L[i],0)
#    xs_lst.append(xs)


fig = plt.figure(figsize=(8,6))

plt.plot(E_bins,xs_bin,lw=2,label=rf"$\Delta$E = {dE} MeV")

plt.yscale('log')
plt.title('22Mg(a,p)25Al')
plt.legend()
plt.xlabel('MeV')
plt.ylabel('Barns')

plt.savefig('22Mg_ap_25Al.png')

plt.show()
