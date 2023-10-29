from math import prod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import time
import numpy as np

import pyomo
import pyomo.environ as pyo
from pyomo.environ import RangeSet, Set

from pyomo.solvers.plugins.solvers.GAMS import GAMSDirect
import subprocess

from pyomo_utils import solve_gams_direct, add_constraints


# License in /Users/zach/Library/Application Support/GAMS/gamslice.txt

## Global, fixed parameters

# m = 2 # i = {1,...,m} policies
# n = 1 # j = {1,...,n} levels
# T = 90 # t = {1,...,T} periods

# From EM & PM spreadsheet
# N = 300000
N = 100
k1_0 = 0.0000006
k2_0 = 0.028
k3_0 = 0.04263157895
k4_0 = 0.005
k5_0 = 0.01428571429
k6_0 = 0.03571428571
k7_0 = 0.015
k8_0 = 0.02040816327

days_per_period = 1
k1,k2,k3,k4,k5,k6,k7,k8 = k1_0*days_per_period, k2_0*days_per_period, k3_0*days_per_period, k4_0*days_per_period, k5_0*days_per_period, k6_0*days_per_period, k7_0*days_per_period, k8_0*days_per_period

S0 = N
I0 = 10 #100 initially infected. Must be >0. Numerically seems more stable with larger I0.
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
		P_param[i,:] = .01*np.linspace(start=1,stop=1/(i+1), num=n)
		C_policy[i,:] = .1*np.linspace(start=1,stop=i, num=n)
	C_policy = C_policy * days_per_period

	C_setup = .1*np.full((m,n,T),c_setup) # m by n by T
	C_infected = 1000*np.full((T),c_infected) # T by 1
	C_death = 10000*c_death # 1 by 1

	KI = np.full((T),k1) # T by 1 # S->I = k1
	KR = np.full((T),k3+k6+k8) # T by 1 # I->R = k3 + k6 + k8
	KD = np.full((T),k7) # T by 1 # I->D = k7
	return P_param, C_policy, C_setup, C_infected, C_death, KI, KR, KD


def solve_pyomo(m,n,T,c_setup=10,c_infected=100,c_death=1000,allow_multiple_policies_per_period=False,cost_per_susceptible_only=False,use_logarithm=False, force_no_policy=False,tee=False,solver="DICOPT",keepfiles=False):


	P_param, C_policy, C_setup, C_infected, C_death, KI, KR, KD = get_params(m,n,T,c_setup,c_infected,c_death)

	model = pyo.ConcreteModel()

	P_param, C_policy, C_setup, C_infected, C_death, KI, KR, KD = get_params(m,n,T,c_setup,c_infected,c_death)
	model.i = pyo.RangeSet(0,m-1)
	model.j = pyo.RangeSet(0,n-1)
	model.t = pyo.RangeSet(0,T-1)

	model.S = pyo.Var(model.t,domain=pyo.NonNegativeReals)
	model.I = pyo.Var(model.t,domain=pyo.NonNegativeReals)
	model.R = pyo.Var(model.t,domain=pyo.NonNegativeReals)
	model.D = pyo.Var(model.t,domain=pyo.NonNegativeReals)
	model.d = pyo.Var(model.t,domain=pyo.NonNegativeReals)
	model.y = pyo.Var(model.i*model.j*model.t,domain=pyo.Binary)
	model.P = pyo.Var(model.t,domain=pyo.NonNegativeReals)
	model.constr = pyo.ConstraintList()
	S = model.S
	I = model.I
	R = model.R
	D = model.D
	d = model.d
	y = model.y
	P = model.P
	constr = model.constr

	if cost_per_susceptible_only:
		variable_cost = sum(C_policy[i,j]*S[t]*y[i,j,t] for i in model.i for j in model.j for t in model.t)
	else:
		variable_cost = sum(C_policy[i,j]*N*y[i,j,t] for i in model.i for j in model.j for t in model.t)
	disease_cost = sum(C_infected[t]*I[t] + C_death*d[t] for t in model.t)
	setup_cost =sum(C_setup[i,j,t]*y[i,j,t] for i in model.i for j in model.j for t in model.t)

	model.obj = pyo.Objective(expr=disease_cost + setup_cost + variable_cost) #minimizes?


	constr.add(S[0] == S0)
	constr.add(I[0] == I0)
	constr.add(R[0] == R0)
	constr.add(D[0] == D0)
	constr.add(d[0] == D0)

	add_constraints(constr,[S[t] == S[t-1] - KI[t]*P[t]*S[t-1]*I[t-1] for t in (tt for tt in model.t if tt != model.t.first()) ]) # (1)
	add_constraints(constr,[I[t] == KI[t]*P[t]*S[t-1]*I[t-1] + I[t-1] - KR[t]*I[t-1] - KD[t]*I[t-1] for t in (tt for tt in model.t if tt != model.t.first())]) # (2)
	add_constraints(constr,[R[t] == R[t-1] + KR[t]*I[t-1] for t in (tt for tt in model.t if tt != model.t.first())]) # (3)
	add_constraints(constr,[d[t] == KD[t]*I[t-1] for t in (tt for tt in model.t if tt != model.t.first())]) # (4)
	add_constraints(constr,[D[t] == D[t-1] + d[t] for t in (tt for tt in model.t if tt != model.t.first())]) # (5)
	if use_logarithm:
		add_constraints(constr,[casadi.log(P[t]) == sum(casadi.log(P_param[i,j] * y[i,j,t] + (1-y[i,j,t])) for i in model.i for j in model.j) for t in model.t]) # (6) # Bonmin fails with logarithm formulation
	else:
		add_constraints(constr,[P[t] == prod(P_param[i,j] * y[i,j,t] + (1-y[i,j,t]) for i in model.i for j in model.j) for t in model.t]) # (6)

	if allow_multiple_policies_per_period:
		add_constraints(constr,[sum(y[i,j,t] for j in model.j) <= 1 for i in model.i for t in model.t]) # (7)
	else:
		add_constraints(constr,[sum(y[i,j,t] for j in model.j for i in model.i) <= 1 for t in model.t]) # (7) at most one policy can be chosen each period

	add_constraints(constr,[y[i,j,t]<= 1 for i in model.i for j in model.j for t in model.t]) # (8)
	if force_no_policy:
		add_constraints(constr,[y[i,j,t] == 0 for i in model.i for j in model.j for t in model.t])
	start = time()
	sol = solve_gams_direct(model,solver=solver,tee=tee,keepfiles=keepfiles)
	end = time()
	timeToSolve = end-start
	objVal = model.obj.expr()

	df = pd.DataFrame({"S": [S[t].value for t in model.t], "I": [I[t].value for t in model.t], "R" : [R[t].value for t in model.t], "D": [D[t].value for t in model.t], "d": [d[t].value for t in model.t],"P":[P[t].value for t in model.t],'t': list(model.t)})

	for i in model.i:
		for j in model.j:
			df[f'y_{i}_{j}'] = [model.y[i,j,t].value for t in model.t]
	return df, objVal, model, timeToSolve


if __name__=="__main__":
	experiments = {}
	results = {}
	solvers = ("DICOPT","BARON",)
	experiment_name = "_large_only"
	Ts = list(range(280,300,10))
	ms = list(range(40,48,2))
	ns = list(range(40,48,2))
	# Ts = list(range(10,30,10))
	# ms = list(range(1,12,4))
	# ns = list(range(1,12,4))
	# Ts=(150,)
	# ms = (15,)
	# ns = (15,)
	for solver in solvers:
		for T in reversed(Ts):
			for m in reversed(ms):
				for n in reversed(ns):
					if (m,n,T) in experiments:
						continue
					nVar = 6*T + m*n*T
					print(f"{nVar} variables ",end="")
					# print(f"{m=}, {n=}, {T=}")
					params = dict(m=m,n=n,T=T,c_setup=100,c_infected=1000,c_death=10000)
					try:
						start = time()
						solDf,objVal,model,timeToSolve = solve_pyomo(**params,
																	  allow_multiple_policies_per_period=True,
																	  solver=solver,
																	  tee=True,
																	  keepfiles=False)
						end = time()
						print(f"Policy used? {any(model.y.extract_values().values())}")
						experiments[m,n,T] = (timeToSolve,objVal)
					except Exception as e:
						print(e)
						experiments[m,n,T] = (np.inf,-1)
					nConstr=len(list(model.constr_index))
					print(f"experiments[{m},{n},{T}]={experiments[m,n,T][0]:.2}s solving ({(end-start) - timeToSolve:.3} s setup, {nVar} variables, {nConstr} constraints")
		df = pd.DataFrame([{'m' : m, 'n': n, 'T' : T, 'runTime' : runTime,'objVal': objVal} for (m,n,T),(runTime,objVal) in experiments.items()])
		results[solver]=df

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
			plt.savefig(f"figures/choke_points{experiment_name}_{solver}.pdf")

			results[solver].to_csv(f"time_to_solve{experiment_name}_{solver}.csv")



















