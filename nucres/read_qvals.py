import pandas as pd 


'''
This python executable is made to calculate the Q-values for any particular reaction.
INPUT: takes the Z and A of projectile and target
OUTPUT: computes the Q values using the AME data table (in common/ directory)
'''

def calc_qval(Z_p,A_p,Z_t,A_t,Z_eject,A_eject):

	df_ame = pd.read_csv('../common/ame20.csv',header=0)

	failure = 'The reaction does not make sense. The Z and A of the residual nuclei is negative!'

	mass_excess_proj = df_ame['mass_excess'][(df_ame['Z'] == Z_p) & (df_ame['A'] == A_p)].astype('float').values

	mass_excess_tar = df_ame['mass_excess'][(df_ame['Z'] == Z_t) & (df_ame['A'] == A_t)].astype('float').values

	mass_excess_eject = df_ame['mass_excess'][(df_ame['Z'] == Z_eject) & (df_ame['A'] == A_eject)].astype('float').values

	Z_res = Z_p + Z_t - Z_eject

	A_res = A_p + A_t - A_eject

	if Z_res >= 0 and A_res >= 0:

		mass_excess_res = df_ame['mass_excess'][(df_ame['Z'] == Z_res) & (df_ame['A'] == A_res)].astype('float').values
		Q_val = mass_excess_proj + mass_excess_tar - mass_excess_eject - mass_excess_res

		return Q_val[0]*1e-3 # MeV

	else:
		return failure

# e.g. Example Q-value calculation for 7Li + 48Ca -> 54V + n
#print(calc_qval(3,7,20,48,0,1))



