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
T = 20 # t = {1,...,T} periods

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

KI = np.full((T),k1) # T by 1 # S->I = k1
KR = np.full((T),k3+k6+k8) # T by 1 # I->R = k3 + k6 + k8
KD = np.full((T),k7) # T by 1 # I->D = k7
S0 = N
I0 = 100 #100 initially infected. Must be >0. Numerically seems more stable with larger I0.
R0 = 0
D0 = 0
d0 = 0
opti = None



# parameters that will be changed in sensitivity analyses
def get_params(C1,C2,P1,P2,c_setup,c_infected,c_death):
	P_param = np.array([[P1,P2] for j in range(n)]).T # m by n
	C_policy = np.array([[C1,C2] for j in range(n)]).T # m by n
	C_setup = np.full((m,n,T),c_setup) # m by n by T
	C_infected = np.full((T),c_infected) # T by 1
	C_death = c_death # 1 by 1
	return P_param, C_policy, C_setup, C_infected, C_death

def solve_bonmin_casadi(C1=10,C2=1,P1=0.5,P2=0.85,c_setup=1000,c_infected=100,c_death=1000,allow_multiple_policies_per_period=False,cost_per_susceptible_only=False,use_logarithm=False, force_no_policy=False):
	global opti
	P_param, C_policy, C_setup, C_infected, C_death = get_params(C1,C2,P1,P2,c_setup,c_infected,c_death)
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
	# Casadi plugin options
	# https://web.casadi.org/python-api/
	p_options = {"discrete":opti.discrete }
	# ,"expand":True
	# Solver options (IPOPT/BONMIN)
	# https://www.coin-or.org/Bonmin/option_pages/options_list_bonmin.html
	# https://coin-or.github.io/Ipopt/OPTIONS.html
	# s_options = {'file_print_level' : 9,'output_file' : 'output.txt', 'expect_infeasible_problem' : 'no' , 'print_user_options' : 'yes'}
	s_options = {'expect_infeasible_problem' : 'no'}
	# ,'print_options_documentation' : 'yes'
	# , 'print_advanced_options' : 'yes',
	# 'print_options_documentation' : True
	# s_options = {"max_iter": 1000000000,'tol': .00000001}
	# s_options = {'iteration_limit': 100000000, 'time_limit': 100000000000}
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
	print(f"Solved using CasADI/BONMIN")
	df = pd.DataFrame({"S": Svals, "I": Ivals, "R" : Rvals, "D": Dvals, "d": dvals,"P":Pvals,'t': list(range(T))})
	for i in range(m):
		for j in range(n):
			df[f'y_{i}_{j}'] = yvals[i,j]
	return df, objVal
	# return Svals, Ivals, Rvals, Dvals, dvals, yvals, Pvals, objVal


if __name__=="__main__":
	experiments = [
		dict(C1=1.1,C2=1,P1=0.0001,P2=0.2,c_setup=50,c_infected=10,c_death=50), #policy 2 most periods
		# dict(C1=.5,C2=.1,P1=0.001,P2=0.2,c_setup=1000,c_infected=1000,c_death=10000), #policy 2 most periods
		# dict(C1=1.5,C2=1,P1=0.005,P2=0.2,c_setup=1000,c_infected=1000,c_death=10000), #policy 2 a few periods
	]

	fig, ax = plt.subplots(figsize=(10,10))
	markers_map = {}
	# all_markers = ["+","x",",","^","d","s","2","3","4"]
	all_markers = [',',  'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X','.',]
	for params in experiments:
		df,objVal = solve_bonmin_casadi(**params,allow_multiple_policies_per_period=True)
		label = f"Objective: \$${objVal:.1}$\n$C=({params['C1']},{params['C2']})$,$P=({params['P1']},{params['P2']})$,$(C^D,C^{{setup}},C^I)=({params['c_death']},{params['c_setup']},{params['c_death']})$"
		lineplot = sns.lineplot(data=df,x='t',y='D',label=label,ax=ax)
		line_color = lineplot.get_lines()[0].get_color().lstrip("#")
		rgb = tuple(int(line_color[i:i+2], 16) for i in (0, 2, 4))
		df['policy'] = [' '.join(f"({i},{j})" for i in range(m) for j in range(n) if df.loc[t, f'y_{i}_{j}']==1)for t in range(df.shape[0])]
		df['policy'] = ['Policy: ' + policy if policy else 'Do nothing.' for policy in df['policy']]
		for policy in df['policy']:
			if policy not in markers_map:
				markers_map[policy] = all_markers.pop(0)
		sns.scatterplot(data=df,x='t',y='D',style='policy',palette=[rgb],markers=markers_map)
	ax.set_title("Optimal Cumulative Deaths under Different Parameters")
	ax.set_ylabel("Cumulative Deaths")
	ax.set_xlabel("Period")
	ax.legend(*[*zip(*{l:h for h,l in zip(*ax.get_legend_handles_labels())}.items())][::-1],fontsize=8) #remove legend duplicates https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
	# plt.savefig(f"figures/cumulative_deaths_vs_time_bonmin_alt.pdf")

