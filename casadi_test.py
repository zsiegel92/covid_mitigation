import numpy as np
import casadi
from math import prod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from casadi_utils import Problem_Maker




x = opti.get_variables_mat((1),True)

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
plt.savefig(f"figures/cumulative_deaths_vs_time_bonmin_alt.pdf")

