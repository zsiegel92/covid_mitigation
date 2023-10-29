import numpy as np
import casadi
from math import prod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from casadi_utils import Problem_Maker
from time import time

## Global, fixed parameters

# m = 2 # i = {1,...,m} policies
# n = 1 # j = {1,...,n} levels
# T = 90 # t = {1,...,T} periods

# From EM & PM spreadsheet
N = 300000
k1_0 = 0.0000006
k2_0 = 0.028
k3_0 = 0.04263157895
k4_0 = 0.005
k5_0 = 0.01428571429
k6_0 = 0.03571428571
k7_0 = 0.015
k8_0 = 0.02040816327

days_per_period = 7
k1,k2,k3,k4,k5,k6,k7,k8 = k1_0*days_per_period, k2_0*days_per_period, k3_0*days_per_period, k4_0*days_per_period, k5_0*days_per_period, k6_0*days_per_period, k7_0*days_per_period, k8_0*days_per_period

S0 = N
I0 = 100 #100 initially infected. Must be >0. Numerically seems more stable with larger I0.
R0 = 0
D0 = 0
d0 = 0
opti = None

# parameters that will be changed in sensitivity analyses
def get_params(m,n,T,c_setup, c_infected,c_death):
	P_param = np.full((m,n),.0001)
	C_policy = np.full((m,n),.0001)
	# Just basic example of costs increasing and probabilities decreasing in i and j
	for i in range(m):
		P_param[i,:] = np.linspace(start=1,stop=1/(i+1), num=n)
		C_policy[i,:] = np.linspace(start=1,stop=i, num=n)
	C_policy = C_policy * days_per_period

	C_setup = np.full((m,n,T),c_setup) # m by n by T
	C_infected = np.full((T),c_infected) # T by 1
	C_death = c_death # 1 by 1

	KI = np.full((T),k1) # T by 1 # S->I = k1
	KR = np.full((T),k3+k6+k8) # T by 1 # I->R = k3 + k6 + k8
	KD = np.full((T),k7) # T by 1 # I->D = k7
	return P_param, C_policy, C_setup, C_infected, C_death, KI, KR, KD

def solve_bonmin_casadi(m,n,T,c_setup=1000,c_infected=100,c_death=1000,allow_multiple_policies_per_period=False,cost_per_susceptible_only=False,use_logarithm=False, force_no_policy=False):
	global opti
	P_param, C_policy, C_setup, C_infected, C_death, KI, KR, KD = get_params(m,n,T,c_setup,c_infected,c_death)
	opti = Problem_Maker()
	S = opti.get_variables_mat((T),False)
	I = opti.get_variables_mat((T),False)
	R = opti.get_variables_mat((T),False)
	D = opti.get_variables_mat((T),False)
	d = opti.get_variables_mat((T),False)
	y = opti.get_variables_mat((m,n,T),True)
	P = opti.get_variables_mat(T,False)

	if cost_per_susceptible_only:
		variable_cost = sum(C_policy[i,j]*S[t]*y[i,j,t] for i in range(m) for j in range(n) for t in range(T))
	else:
		variable_cost = sum(C_policy[i,j]*N*y[i,j,t] for i in range(m) for j in range(n) for t in range(T))
	disease_cost = sum(C_infected[t]*I[t] + C_death*d[t] for t in range(T))
	setup_cost =sum(C_setup[i,j,t]*y[i,j,t] for i in range(m) for j in range(n) for t in range(T))
	opti.minimize(disease_cost + setup_cost + variable_cost)

	opti.subject_to(S[0] == S0)
	opti.subject_to(I[0] == I0)
	opti.subject_to(R[0] == R0)
	opti.subject_to(D[0] == D0)
	opti.subject_to(d[0] == D0)

	opti.subject_to([S[t] == S[t-1] - KI[t]*P[t]*S[t-1]*I[t-1] for t in range(1,T)]) # (1)
	opti.subject_to([I[t] == KI[t]*P[t]*S[t-1]*I[t-1] + I[t-1] - KR[t]*I[t-1] - KD[t]*I[t-1] for t in range(1,T)]) # (2)
	opti.subject_to([R[t] == R[t-1] + KR[t]*I[t-1] for t in range(1,T)]) # (3)
	opti.subject_to([d[t] == KD[t]*I[t-1] for t in range(1,T)]) # (4)
	opti.subject_to([D[t] == D[t-1] + d[t] for t in range(1,T)]) # (5)
	if use_logarithm:
		opti.subject_to([casadi.log(P[t]) == sum(casadi.log(P_param[i,j] * y[i,j,t] + (1-y[i,j,t])) for i in range(m) for j in range(n)) for t in range(T)]) # (6) # Bonmin fails with logarithm formulation
	else:
		opti.subject_to([P[t] == prod(P_param[i,j] * y[i,j,t] + (1-y[i,j,t]) for i in range(m) for j in range(n)) for t in range(T)]) # (6)

	if allow_multiple_policies_per_period:
		opti.subject_to([sum(y[i,j,t] for j in range(n)) <= 1 for i in range(m) for t in range(T)]) # (7)
	else:
		opti.subject_to([sum(y[i,j,t] for j in range(n) for i in range(m)) <= 1 for t in range(T)]) # (7) at most one policy can be chosen each period

	opti.subject_to([opti.bounded(0,y[i,j,t],1) for i in range(m) for j in range(n) for t in range(T)]) # (8)
	if force_no_policy:
		opti.subject_to([y[i,j,t] == 0 for i in range(m) for j in range(n) for t in range(T)])


	opti.set_solver()
	sol = opti.solve()

	Svals = opti.extract_values(sol,S)
	Ivals = opti.extract_values(sol,I)
	Rvals = opti.extract_values(sol,R)
	Dvals = opti.extract_values(sol,D)
	dvals = opti.extract_values(sol,d)
	yvals = opti.extract_values(sol,y)
	Pvals = opti.extract_values(sol,P)
	objVal = sol.value(opti.f)
	print(f"Solved using CasADI/BONMIN")
	df = pd.DataFrame({"S": Svals, "I": Ivals, "R" : Rvals, "D": Dvals, "d": dvals,"P":Pvals,'t': list(range(T))})
	for i in range(m):
		for j in range(n):
			df[f'y_{i}_{j}'] = yvals[i,j]
	return df, objVal
	# return Svals, Ivals, Rvals, Dvals, dvals, yvals, Pvals, objVal


if __name__=="__main__":
	experiments = {}
	Ts = list(range(10,90,10))
	ms = list(range(1,8))
	ns = list(range(1,8))
	for T in Ts:
		for m in ms:
			for n in ns:
				if (m,n,T) in experiments:
					continue
				print(f"{m=}, {n=}, {T=}")
				print(experiments)
				params = dict(m=m,n=n,T=T,c_setup=100,c_infected=1000,c_death=10000)
				try:
					start = time()
					df,objVal = solve_bonmin_casadi(**params,allow_multiple_policies_per_period=True)
					end = time()
					experiments[m,n,T] = end-start
				except:
					experiments[m,n,T] = np.inf

	df = pd.DataFrame([{'m' : m, 'n': n, 'T' : T, 'runTime' : runTime} for (m,n,T),runTime in experiments.items()])


	if True:
		fig, axs = plt.subplots(nrows=1,ncols=len(Ts),figsize=(10*len(Ts),10),sharey=True)
		cmap = plt.cm.get_cmap('viridis')
		cbar_ax = fig.add_axes([0.9,0.2, 0.01, .6]) # [1.015,0.13, 0.015, 0.8]
		for plotIndex, (ax,T) in enumerate(zip(axs,Ts)):
			records = {m : {n : df.loc[ (df['m'] == m) & (df['n'] == n) & (df['T']==T)]['runTime'].values[0] for n in ns} for m in ms}
			mat = pd.DataFrame.from_records(records)
			im = sns.heatmap(data=mat,
			                 vmin=0,
			                 vmax=df.loc[df['runTime']!=np.inf]['runTime'].max(),
			                 ax=ax,
			                 cmap=cmap,
			                 cbar=True if plotIndex ==0 else False,
			                 cbar_ax=cbar_ax if plotIndex == 0 else None)
			ax.set_title(f"T={T} Time Periods")
			ax.set_xlabel(f"m Number Policies")
			ax.set_ylabel(f"n Number Policy Levels")
		fig.suptitle("Time to Solve Problems of various Sizes\n(no value implies failure to solve)",fontsize=18)
		plt.savefig(f"figures/choke_points_bonmin.pdf")


	# # https://stackabuse.com/ultimate-guide-to-heatmaps-in-seaborn-with-python







