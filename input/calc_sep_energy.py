import pandas as pd
import numpy as np

df = pd.read_csv('../data/ame20.csv',header=0)

mass_excess_proj = df['mass_excess'][(df['Z'] == 2) & (df['A'] == 4)].values.astype(float)
mass_excess_exit = df['mass_excess'][(df['Z'] == 1) & (df['A'] == 1)].values.astype(float)

def calc_sep_energy(Z_tar,A_tar):

    Z_comp = Z_tar + 2
    A_comp = A_tar + 4

    Z_res = Z_comp - 1
    A_res = A_comp - 1

    mass_excess_tar = df['mass_excess'][(df['Z'] == Z_tar) & (df['A'] == A_tar)].values.astype(float)

    mass_excess_comp = df['mass_excess'][(df['Z'] == Z_comp) & (df['A'] == A_comp)].values.astype(float)

    mass_excess_res = df['mass_excess'][(df['Z'] == Z_res) & (df['A'] == A_res)].values.astype(float)

    proj_sep_energy = mass_excess_proj + mass_excess_tar - mass_excess_comp
    exit_sep_energy = mass_excess_res + mass_excess_exit - mass_excess_comp

    return proj_sep_energy, exit_sep_energy

Z = np.arange(8,22,2)

A = np.arange(14,40,4)


for i in range(len(Z)):
    proj_sep_energy, exit_sep_energy = calc_sep_energy(Z[i],A[i])
    print("Z = ", Z[i], "A = ", A[i], "proj_sep_energy = ", proj_sep_energy, "exit_sep_energy = ", exit_sep_energy)
    print()