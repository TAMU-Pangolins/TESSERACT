import pandas as pd


def lab_to_cm(E_lab, A_tar, sym_tar, A_pro, sym_pro):
    """Convert a projectile lab-frame energy to center-of-mass energy in MeV."""

    df_ame = pd.read_csv("../common/ame20.csv", header=0)  # load the ame file

    # find the isotope's atomic mass (in amu) in the AME table.
    amu_tar = df_ame["atomic_mass"][
        (df_ame["el"].str.lower().isin([sym_tar.lower()])) & (df_ame["A"] == A_tar)
    ].values

    amu_pro = df_ame["atomic_mass"][
        (df_ame["el"].str.lower().isin([sym_pro.lower()])) & (df_ame["A"] == A_pro)
    ].values

    # convert the strings to floats and calculate the energy in center of mass frame.
    try:
        amu_tar = amu_tar.astype(float)
        amu_pro = amu_pro.astype(float)
        E_cm = E_lab * amu_tar / (amu_tar + amu_pro)
        return E_cm.item()

    except:
        # if conversion to string not possible, print error message along with the masses of the target and projectile from the AME table.

        err_message = f"Could not convert mass numbers to floats. Mass of target = {amu_tar}; mass of projectile = {amu_pro}"

        return err_message


# example run
# print(lab_to_cm(5,22,'ne',4,'he'))
