import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import pandas as pd
from nucres.resonance import Resonance, sigma_bw_energy_dep
from scipy.interpolate import make_interp_spline

#resonance_i = Resonance(1,0.5,0.5,0.5,3.65*10**(-26),6.64*10**(-27),1,1)
output_dir = 'outputs/22Mg(a,p)25Al/'
file = [output_dir +'RUN_0/22Mg(a,p)25Al.in',output_dir + 'RUN_1/22Mg(a,p)25Al.in']
bin_width = [0.1,0.2]
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

fig = plt.figure(figsize=(8,6))

for j in range(len(file)):

    df = extract_data(file[j])
    df = df.sort_values(by='Ecm',ascending=True)
    E_cm = df['Ecm']
    gamma_i = df['G1']
    gamma_o = df['G2']
    Jr = df['Jr']
    l1 = df['L1']
    l2 = df['L2']
    L = l1 + l2

#resonance_i = Resonance(1,0.5,0.5,0.5,3.65*10**(-26),6.64*10**(-27),1,1)

    xs_lst = []
    for i in range(len(E_cm)):

        r_i = Resonance(E_cm[i]*10**6,Jr[i],0,0,3.65*10**(-26),6.64*10**(-27),gamma_i[i],gamma_o[i])

        xs = sigma_bw_energy_dep([E_cm[i]*10**6],r_i,12,2,22,4,L[i],0)
        xs_lst.append(xs)


        #plt.scatter(E_cm*10**(-3),xs_ls,s=5)
    spline_fit = make_interp_spline(E_cm.values,xs_lst)
    E_spline = np.linspace(E_cm.min(),E_cm.max(),500)
    xs_spline = spline_fit(E_spline)

    plt.plot(E_spline*10**(-3),xs_spline,lw=2,label=rf"$\Delta$E = {bin_width[j]} MeV")

plt.yscale('log')
plt.title('22Mg(a,p)25Al')
plt.legend()
plt.xlabel('MeV')
plt.ylabel('Barns')

plt.savefig('22Mg_ap_25Al.png')

plt.show()
