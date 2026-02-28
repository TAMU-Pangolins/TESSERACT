import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
import pandas as pd

reaction = sys.argv[1]
dE_lst = [0.1,0.2,0.3]

files = sorted(glob.glob(f"{reaction}_xs_part_*.txt"))
#df_unint = pd.read_csv(f"{reaction}_xs_unintegrated_parallel.txt",header=None,comment='#')

if not files:
    raise RuntimeError(f"No chunk files found for {reaction}")

plt.figure(figsize=(8,8))

for f in files:

    data = np.loadtxt(f,delimiter=',')
    E = data[:,0]
    xs = data[:,1]

    job_id = f.split("_")[-1].split(".")[0]

    plt.plot(E,xs*1e3,color='k',alpha=0.5,lw=2)

for dE in dE_lst:
    int_df = pd.read_csv(f"{reaction}_integrated_xs_dE_{dE}.txt",header=None,comment='#')
    plt.plot(int_df[0],int_df[1],lw=2,label=f'Integrated at {dE} MeV')

plt.xlabel('E (MeV)')
plt.ylabel('Cross section (mb)')
plt.ylim(1e-10)
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()
