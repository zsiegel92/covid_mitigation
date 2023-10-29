import numpy as np
from math import prod
import pandas as pd
import numpy as np

def no_policy_model(m,n,T, N, KI, KR, KD, I0, S0, D0, R0,P_param, C_policy, C_setup, C_infected, C_death,individual_multiplier, days_per_period, cost_multiplier,allow_multiple_policies_per_period=False,cost_per_susceptible_only=False,use_logarithm=False, force_no_policy=True):
	df = pd.DataFrame(0.000,index=range(T),columns=("S","I","R","d","D"))
	S = df["S"]
	I = df["I"]
	R = df["R"]
	d = df["d"]
	D = df["D"]
	S[0] = S0
	I[0] = I0
	R[0] = R0
	d[0] = 0
	D[0] = D0
	# df.loc[0,"S"]= S0
	# df.loc[0,"I"] = I0
	for t in range(1,T):
		S[t] = S[t-1] - KI[t]*S[t-1]*I[t-1]
		I[t] = I[t-1] + KI[t]*S[t-1]*I[t-1]- KR[t]*I[t-1] - KD[t]*I[t-1]
		R[t] = R[t-1] + KR[t]*I[t-1]
		d[t] = KD[t]*I[t-1]
		D[t] = D[t-1] + d[t]
	for c in ("S","I","R","d","D"):
		df[c] = individual_multiplier*df[c]
	df['tot'] = sum(df[k] for k in ("S","I","R","D"))
	return df
