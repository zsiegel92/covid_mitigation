from math import prod
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
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


## Global, fixed parameters

m = 2 # i = {1,...,m} policies
n = 1 # j = {1,...,n} levels
T = 250 # t = {1,...,T} periods

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

days_per_period = 1
pct_inf0 = .1
I0 = 2 #N*pct_inf0 # pct_inf% initially infected. Must be >0.
M0 = 0
V0 = 0
S0 = N - I0
R0 = 0
D0 = 0
d0 = 0

individual_multiplier = 100
I0 = I0/individual_multiplier
S0 = S0/individual_multiplier



# parameters that will be changed in sensitivity analyses
def get_params(C1,C2,P1,P2,c_setup,c_infected,c_death):
	P_param = np.array([[P1,P2] for j in range(n)]).T # m by n
	C_policy = individual_multiplier*np.array([[C1,C2] for j in range(n)]).T # m by n
	C_setup = np.full((m,n,T),c_setup) # m by n by T
	C_infected = individual_multiplier*np.full((T),c_infected) # T by 1
	C_death = individual_multiplier*c_death # 1 by 1
	return P_param, C_policy, C_setup, C_infected, C_death

def solve_pyomo(C1=10,C2=1,P1=0.5,P2=0.85,c_setup=1000,c_infected=100,c_death=1000,allow_multiple_policies_per_period=False,cost_per_susceptible_only=False,use_logarithm=False, force_no_policy=False,tee=False,solver="DICOPT",keepfiles=False,warmstart=True):



	model = pyo.ConcreteModel()

	P_param, C_policy, C_setup, C_infected, C_death = get_params(C1,C2,P1,P2,c_setup,c_infected,c_death)
	model.i = pyo.RangeSet(0,m-1)
	model.j = pyo.RangeSet(0,n-1)
	model.t = pyo.RangeSet(0,T-1)

	model.S = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=N)
	model.I = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.M = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.V = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.R = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.D = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.d = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.y = pyo.Var(model.i*model.j*model.t,domain=pyo.Binary,initialize=0)
	model.P = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=1)
	model.constr = pyo.ConstraintList()
	S = model.S
	I = model.I
	M = model.M
	V = model.V
	R = model.R
	D = model.D
	d = model.d
	y = model.y
	P = model.P
	constr = model.constr

	if cost_per_susceptible_only:
		variable_cost = sum(C_policy[i,j]*S[t]*y[i,j,t] for i in model.i for j in model.j for t in model.t)
	else:
		variable_cost = sum(C_policy[i,j]*(N/individual_multiplier)*y[i,j,t] for i in model.i for j in model.j for t in model.t)
	disease_cost = sum(C_infected[t]*I[t] + C_death*d[t] for t in model.t)
	setup_cost =sum(C_setup[i,j,t]*y[i,j,t] for i in model.i for j in model.j for t in model.t)
	model.obj = pyo.Objective(expr=disease_cost + setup_cost + variable_cost,sense=pyo.minimize)

	t0 = model.t.first() # do not have to start indices at 0.
	t_except_first = list(t for t in model.t if t != t0) #periods except first period

	constr.add(S[t0] == S0)
	constr.add(I[t0] == I0)
	constr.add(M[t0] == M0)
	constr.add(V[t0] == V0)
	constr.add(R[t0] == R0)
	constr.add(D[t0] == D0)
	constr.add(d[t0] == D0)


	add_constraints(constr,[S[t] == S[t-1] - k1*P[t]*S[t-1]*I[t-1] for t in t_except_first ]) # (1)

	add_constraints(constr,[I[t] == (1-(k2+k3+k4))*I[t-1] +k1*P[t]*S[t-1]*I[t-1] for t in t_except_first]) # I
	add_constraints(constr,[M[t] == (1- (k5+k6)) * M[t-1] + k2*I[t-1] for t in t_except_first ]) # M
	add_constraints(constr,[V[t] == (1-(k7+k8))* V[t-1] + k4*I[t-1] + k5*M[t-1] for t in t_except_first ]) # V
	add_constraints(constr,[R[t] == R[t-1] +k3*I[t-1] + k6*M[t-1] + k8*V[t-1] for t in t_except_first]) # R

	add_constraints(constr,[d[t] == k7*V[t-1] for t in t_except_first]) # (4)
	add_constraints(constr,[D[t] == D[t-1] + d[t] for t in t_except_first]) # (5)

	if use_logarithm:
		add_constraints(constr,[pyo.log10(P[t]) == sum(pyo.log10(P_param[i,j] * y[i,j,t] + (1-y[i,j,t])) for i in model.i for j in model.j) for t in model.t]) # (6)
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
	sol = solve_gams_direct(model,solver=solver,tee=tee,keepfiles=keepfiles,warmstart=warmstart)
	end = time()
	timeToSolve = end-start
	objVal = model.obj.expr()
	df = pd.DataFrame({k :[(individual_multiplier if k!="P" else 1)*getattr(model,k)[t].value for t in model.t] for k in ("S","I","M","V","R","D")})
	df.insert(0,"t",list(model.t))
	# df = pd.DataFrame({'t': list(model.t),"S": [individual_multiplier*S[t].value for t in model.t], "I": [individual_multiplier*I[t].value for t in model.t],"M": [individual_multiplier*M[t].value for t in model.t], "V": [individual_multiplier*V[t].value for t in model.t], "R" : [individual_multiplier*R[t].value for t in model.t], "D": [individual_multiplier*D[t].value for t in model.t], "d": [individual_multiplier*d[t].value for t in model.t],"P":[P[t].value for t in model.t],})
	df.set_index('t')
	df['tot'] = df['S'] + df['I']  + df['M'] + df['V'] + df['R'] + df['D']
	for i in model.i:
		df[f'y_{i}'] = - 1
		for j in model.j:
			df[f'y_{i}_{j}'] = [model.y[i,j,t].value for t in model.t]
			for t in model.t:
				if model.y[i,j,t].value==1:
					df[f'y_{i}'][t]=j
	return df, objVal, model, timeToSolve



if __name__=="__main__":
	solver = "DICOPT"
	params = dict(C1=2,C2=1,P1=0.5,P2=0.9,c_setup=100,c_infected=100000,c_death=200000) #policy 2 most periods
		# dict(C1=.5,C2=.1,P1=0.001,P2=0.2,c_setup=1000,c_infected=1000,c_death=10000), #policy 2 most periods
		# dict(C1=1.5,C2=1,P1=0.005,P2=0.2,c_setup=1000,c_infected=1000,c_death=10000), #policy 2 a few periods

	# params = experiments[0]
	# df,objVal = solve_pyomo(**params,allow_multiple_policies_per_period=True)
	# model, sol = solve_pyomo(**params,allow_multiple_policies_per_period=True)


	gs = gridspec.GridSpec(nrows=2,ncols=1,height_ratios=[3,1])
	fig = plt.figure(figsize=(15,10))
	ax1 = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])
	fig.subplots_adjust(hspace=0.5)




	markers_map = {}
	# all_markers = ["+","x",",","^","d","s","2","3","4"]
	all_markers = [',',  'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X','.',]

	solver_params = dict(allow_multiple_policies_per_period=True,
							cost_per_susceptible_only=False,
							use_logarithm=False,
							solver=solver,
							tee=True,
							keepfiles=False,
							warmstart=True,
							**params)
	df_no_policy,objVal_no_policy, model_no_policy, timeToSolve_no_policy = solve_pyomo(**solver_params,force_no_policy=True)
	df,objVal, model, timeToSolve = solve_pyomo(**solver_params)


	print(f"Used any policy? {any(model.y.extract_values().values())}")
	# model.obj.pprint()
	populations = {'Infected' : "I",'Cumulative Deaths': "D", "Recovered" : "R", "Susceptible" : "S" }
	for population in populations:
		sns.lineplot(data=df_no_policy,x='t',y=populations[population],label=f"{population} - No intervention", ax=ax1) #,palette=[rgb],
		sns.lineplot(data=df,x='t',y=populations[population],label=population, ax=ax1) #,palette=[rgb],
		lines = {line.get_label() : line for line in ax1.lines} # ax1.lines gets shuffled each time, have to find correct line
		line_color = lines[population].get_color()#.lstrip("#")
		control_line = lines[f'{population} - No intervention']
		control_line.set_color(line_color)
		control_line.set_linestyle("--")


	# lineplot = sns.lineplot(data=df,x='t',y='D',label="Cumulative Deaths",ax=ax1)
	# lineplot = sns.lineplot(data=df,x='t',y='S',label="Susceptible",ax=ax1)
	# lineplot = sns.lineplot(data=df,x='t',y='R',label="Recovered",ax=ax1)
	# lineplot = sns.lineplot(data=df,x='t',y='I',label="Infected",ax=ax1)

	# lines = {line.get_label() : line for line in ax1.lines}
	# for population in populations:
	# 	infected_line_color = lines[population].get_color()#.lstrip("#")
	# 	# rgb = tuple(int(infected_line_color[i:i+2], 16) for i in (0, 2, 4))
	# 	control_line = lines[f'{population} - No intervention']
	# 	control_line.set_color(infected_line_color)
	# 	control_line.set_linestyle("--")

	df['policy'] = [' '.join(f"({i+1},{j+1})" for i in range(m) for j in range(n) if df.loc[t, f'y_{i}_{j}']==1)for t in range(df.shape[0])]
	df['policy'] = ['Policy: ' + policy if policy else 'Do nothing.' for policy in df['policy']]
	# line_color = lineplot.get_lines()[0].get_color().lstrip("#")
	# rgb = tuple(int(line_color[i:i+2], 16) for i in (0, 2, 4))
	# for policy in df['policy']:
	# 	if policy not in markers_map:
	# 		markers_map[policy] = all_markers.pop(0)
	# sns.scatterplot(data=df,x='t',y='D',style='policy',palette=[rgb],markers=markers_map)


	ax1.set_title(f"Objective: \$${objVal}$\n$C=({params['C1']},{params['C2']})$,$P=({params['P1']},{params['P2']})$,$(C^D,C^{{setup}},C^I)=({params['c_death']},{params['c_setup']},{params['c_death']})$")
	ax1.set_ylabel("Number Individuals")
	ax1.set_xlabel("Period")
	# ax1.legend(*[*zip(*{l:h for h,l in zip(*ax1.get_legend_handles_labels())}.items())][::-1],fontsize=8) #remove legend duplicates https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
	ax1.legend(fontsize=8)
	policy_changes = [df['policy'][0]]
	policy_change_periods = [0]
	for t in range(T):
		if df['policy'][t] != policy_changes[-1]:
			policy_changes.append(df['policy'][t])
			policy_change_periods.append(t)
	if len(policy_changes) > 1:
		policy_changes.pop(0)
		policy_change_periods.pop(0)
	nChanges = len(policy_changes)

	ylim = ax1.get_ylim()[1]
	for policy_change in policy_change_periods:
		ax1.axvline(policy_change, 0,ylim,linestyle=":",alpha=0.4,zorder=-1)



	ax2.axis('off')
	nRows = m
	nCols = nChanges
	cols = [f"{policy_change_periods[t]}-{policy_change_periods[t+1]-1}" if t!=nChanges-1 else f"{policy_change_periods[t]}-{T-1}" for t in range(nChanges)]

	# colors = [line_content['color'] for (plotLabel, line_content) in lumped_vals_separated]
	# rows = [plotLabel for (plotLabel, line_content) in lumped_vals_separated]
	rows = [f"Level of Policy {i+1}" for i in range(m)]
	content = [[str(df[f'y_{i}'][t] + 1) if df[f'y_{i}'][t] != -1 else '' for t in policy_change_periods] for i in range(m)]

	# # content.reverse()
	# # colors = get_n_colors(nRows)
	# # Adjust layout to make room for the table:
	plt.subplots_adjust(bottom=0.3)


	table = ax2.table(cellText = content,
					  # rowColours=colors,
					  rowLabels=rows,
					  colLabels=cols,
					  loc='center',
					  cellLoc='left')
	table_d = table.get_celld()


	min_width = table_d[1,0].get_width()/5
	base_w = (table.get_celld()[1,0].get_width())*nChanges/T

	for j in range(nCols):
		if j < len(policy_change_periods)-1:
			col_mult = policy_change_periods[j+1] - policy_change_periods[j]
		else:
			col_mult = T-1 - policy_change_periods[j]
		for i in range(nRows+1):
			table_d[i,j].set_width(max(min_width,base_w * col_mult))

	table.auto_set_font_size(False)
	table.set_fontsize(8)
	# plt.savefig(f"figures/system_state_vs_time_SIMVRD_{solver}.pdf")

	# testing total population conservation
	df_no_policy.to_csv("SIMVRD_optimized.csv",index=False)
