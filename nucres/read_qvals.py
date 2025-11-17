import csv
from functools import lru_cache

'''
This python executable is made to calculate the Q-values for any particular reaction.

INPUT: takes the Z and A of projectile and target
OUTPUT: computes the Q values using the AME data table (in data/ directory)
'''

@lru_cache(maxsize=1)
def load_ame(path="../data/ame20.csv"):
    lut = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lut[(int(row["Z"]), int(row["A"]))] = float(row["mass_excess"])
    return lut

def calc_qval(Z_p, A_p, Z_t, A_t, Z_eject, A_eject, ame=None):
    ame = ame or load_ame()
    Z_res = Z_p + Z_t - Z_eject
    A_res = A_p + A_t - A_eject
    if Z_res < 0 or A_res < 0:
        return "The reaction does not make sense. The Z and A of the residual nuclei is negative!"
    try:
        return (ame[(Z_p, A_p)] + ame[(Z_t, A_t)]
                - ame[(Z_eject, A_eject)] - ame[(Z_res, A_res)]) * 1e-3
    except KeyError:
        return "Mass excess missing for one of the nuclides."


# e.g. Example Q-value calculation for 7Li + 48Ca -> 54V + n
#print(calc_qval(3,7,20,48,0,1))



