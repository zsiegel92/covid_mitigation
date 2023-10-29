import numpy as np
import casadi
from math import prod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from casadi_utils import Problem_Maker

## Global, fixed parameters

m = 2 # i = {1,...,m} policies
n = 1 # j = {1,...,n} levels
T = 30 # t = {1,...,T} periods

# From EM & PM spreadsheet
N = 300000
k1 = 0.0000006
k2 = 0.028
k3 = 0.04263157895
k4 = 0.005
k5 = 0.01428571429
k6 = 0.03571428571
k7 = 0.015
k8 = 0.02040816327

KI = np.full((T),k1) # T by 1 # S->I = k1
KR = np.full((T),k3+k6+k8) # T by 1 # I->R = k3 + k6 + k8
KD = np.full((T),k7) # T by 1 # I->D = k7
S0 = N
I0 = 100 #100 initially infected. Must be >0. Numerically seems more stable with larger I0.
R0 = 0
D0 = 0
d0 = 0

# parameters that will be changed in sensitivity analyses
def get_params(C1,C2,P1,P2,c_setup,c_infected,c_death):
	P_param = np.array([[P1,P2] for j in range(n)]).T # m by n
	C_policy = np.array([[C1,C2] for j in range(n)]).T # m by n
	C_setup = np.full((m,n,T),c_setup) # m by n by T
	C_infected = np.full((T),c_infected) # T by 1
	C_death = c_death # 1 by 1
	return P_param, C_policy, C_setup, C_infected, C_death

def solve_enumerative(C1=10,C2=1,P1=0.5,P2=0.85,c_setup=1000,c_infected=100,c_death=1000,allow_multiple_policies_per_period=False,cost_per_susceptible_only=False,use_logarithm=False):
	P_param, C_policy, C_setup, C_infected, C_death = get_params(C1,C2,P1,P2,c_setup,c_infected,c_death)
	best_obj = np.inf


	S = np.zeros((T))
	I = np.zeros((T))
	R = np.zeros((T))
	D = np.zeros((T))
	d = np.zeros((T))
	y = np.zeros((m,n,T))
	P = np.zeros((T))

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

	p_options = {"discrete":opti.discrete,"expand":True}
	s_options = {"max_iter": 1000000,'tol': 1}
	opti.solver('bonmin',p_options,s_options)
	sol = opti.solve()

	Svals = opti.extract_values(sol,S)
	Ivals = opti.extract_values(sol,I)
	Rvals = opti.extract_values(sol,R)
	Dvals = opti.extract_values(sol,D)
	dvals = opti.extract_values(sol,d)
	yvals = opti.extract_values(sol,y)
	Pvals = opti.extract_values(sol,P)

	objVal = sol.value(opti.f)
	print(f"Solved using CasADI")
	df = pd.DataFrame({"S": Svals, "I": Ivals, "R" : Rvals, "D": Dvals, "d": dvals,"P":Pvals,'t': list(range(T))})
	for i in range(m):
		for j in range(n):
			df[f'y_{i}_{j}'] = yvals[i,j]
	return df, objVal
	# return Svals, Ivals, Rvals, Dvals, dvals, yvals, Pvals, objVal

