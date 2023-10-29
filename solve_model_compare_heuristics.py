# cd "/Users/zach/UCLA Anderson Dropbox/Zach Siegel/UCLA_classes/research/covid_mitigation" && python -m IPython -i "pyomo_implementation.py" && exit
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo

from policy_helpers import Policy, PolicyList
from process_results import get_results_df, plot_results, Solution, plot_time_trials
from solvers_and_heuristics import get_params, get_model, fix_and_unfix_range, get_heuristic_solver, solve_and_process, solve_and_process_index, solve_and_process_lookahead, solve_and_process_local_search


# nPolicies_subset can be any integer in 1,...,9
# policy_subset can be any subset of list(range(9))
def get_all_policies(T, pruning=False, policy_subset=None, nPolicies_subset=None):
	C_L = 800
	C_M = 1000
	C_MH = 1225
	C_H = 1400
	P_L = 0.9
	P_M = 0.925
	P_ML = 0.95
	P_MH = 0.975
	P_H = 0.995
	C_setup_L = 25000
	C_setup_M = 50000
	C_setup_H = 100000
	C_switch_L = C_setup_L * 2
	C_switch_M = C_setup_M * 2
	C_switch_H = C_setup_H * 2
	type_MH = dict(
		C=[C_M, C_H],
		P=[P_M, P_L],
		C_setup=[C_setup_M, C_setup_H],
		C_switch=[C_switch_M, C_switch_H],
		)
	type_MH_setup0 = dict(
		C=[C_M, C_H],
		P=[P_H, P_M],
		C_setup=[0, 0],
		C_switch=[0, 0],
		)
	type_LMMhH_setup0 = dict(
		C=[C_L, C_M, C_MH, C_H],
		P=[P_H, P_MH, P_ML, P_L],
		C_setup=[0, 0, 0, 0],
		C_switch=[0, 0, 0, 0],
		)
	type_LMH = dict(
		C=[C_L, C_M, C_H],
		P=[P_H, P_M, P_L],
		C_setup=[C_setup_L, C_setup_M, C_setup_H],
		C_switch=[C_switch_L, C_switch_M, C_switch_H],
		)
	type_LMH_setup0 = dict(
		C=[C_L, C_M, C_H],
		P=[P_H, P_M, P_L],
		C_setup=[0, 0, 0],
		C_switch=[0, 0, 0],
		)
	type_M_setup0 = dict(
		C=[C_M],
		P=[P_L],
		C_setup=[0],
		C_switch=[0],
		)

	base_policies = [
		Policy(**type_MH, T=T, name="Movement"),
		Policy(**type_MH_setup0, T=T, name="Education (University level)"),
		Policy(**type_LMMhH_setup0, T=T, name="Social Gatherings (in a house)"),
		Policy(**type_LMH, T=T, name="Non-Food Service (bank,retail, etc)"),
		Policy(**type_MH, T=T, name="Restaurants"),
		Policy(**type_LMH_setup0, T=T, name="Masking"),
		Policy(**type_LMH, T=T, name="Mega Events"),
		Policy(**type_MH, T=T, name="Border Control"),
		Policy(**type_M_setup0, T=T, name="Physical Distancing"),
		]

	if nPolicies_subset is not None:
		# assert nPolicies_subset < len(base_policies)
		policy_subset = list(range(nPolicies_subset))
	if policy_subset is not None:
		policies = PolicyList(*[base_policies[i] for i in policy_subset])
	else:
		policies = PolicyList(*base_policies)
	print(policies)
	if pruning:
		policies = policies.pruned()
		print(policies)
		allow_multiple_policies_per_period = False
	return policies


def get_all_kwargs(
	allow_multiple_policies_per_period, T, individual_multiplier, days_per_period, N,
	cost_multiplier, policies, KI0, KR0, KD0, solver, bilinear, solve_from_binary, optGap, max_time,
	multistart
	):
	multistart_kwargs = dict(
		iterations=-1, HCS_max_iterations=20
		) #-1-> high confidence stopping. Default 10
	model_kwargs = dict(
		allow_multiple_policies_per_period=allow_multiple_policies_per_period,
		cost_per_susceptible_only=False,
		use_logarithm=False,
		time_varying_costs_and_probabilities=False
		)
	param_kwargs = dict(
		T=T,
		individual_multiplier=individual_multiplier,
		days_per_period=days_per_period,
		N=N,
		cost_multiplier=cost_multiplier,
		policies=policies,
		KI0=KI0,
		KR0=KR0,
		KD0=KD0,
		c_infected=10000,
		c_death=10000000,
		num_infected0=100,
		num_recovered0=0,
		num_dead0=0
		)
	solver_params = dict(
		solver=solver,
		bilinear=bilinear,
		tee=True,
		keepfiles=True,
		warmstart=True,
		solve_from_binary=solve_from_binary,
		optGap=optGap,
		max_time=max_time,
		multistart=multistart,
		multistart_kwargs=multistart_kwargs
		)

	return model_kwargs, param_kwargs, solver_params


def get_all_heuristics(use_gurobipy=False):
	local_search = dict(
		do_local_search=True,
		local_search_kwargs=dict(
			w=2, iterations=2, iterate_until_no_improvement=False, optGap=0.3, tee=True
			)
		)
	no_solution_heuristic = get_heuristic_solver(
		solve_and_process,
		get_no_policy_solution=True,
		only_no_policy=True,
		use_gurobipy=use_gurobipy,
		optGap=0.01
		)
	exact_heuristic = get_heuristic_solver(
		solve_and_process,
		get_no_policy_solution=False,
		use_gurobipy=False,
		only_policy=True,
		optGap=0.01
		)
	heuristics = [no_solution_heuristic, exact_heuristic]
	for ls in (local_search, {}):
		lookahead_heuristic = get_heuristic_solver(
			solve_and_process_lookahead,
			w=12,
			truncate_costs=5,
			tee=True,
			**ls,
			)
		index_heuristic = get_heuristic_solver(
			solve_and_process_index,
			max_number_of_policies=1,
			tee=True,
			**ls,
			)
		index_heuristic = get_heuristic_solver(
			solve_and_process_index,
			max_number_of_policies=5,
			tee=True,
			**ls,
			)
		heuristics.extend([lookahead_heuristic, index_heuristic])
		# heuristics.extend([solver_heuristic, lookahead_heuristic])
	for optGap in (0.5, 0.85, 1.0):
		solver_heuristic = get_heuristic_solver(
			solve_and_process,
			get_no_policy_solution=False,
			use_gurobipy=False,
			only_policy=True,
			optGap=optGap
			)
		heuristics.append(solver_heuristic)
	return no_solution_heuristic, heuristics


def get_csv_with_indexing(df, index_cols, csv_basename):
	x = df.copy()
	x.reset_index()
	# put index columns first
	for i, colname in enumerate(index_cols):
		x['tmp'] = x[colname]
		del x[colname]
		x.insert(i, colname, x['tmp'])
		del x['tmp']
	x.sort_values(by=index_cols, inplace=True)
	x.to_csv(f"{csv_basename}.csv", index=False)
	# clear columns
	n_ind = len(index_cols)
	for clearcol in range(1, n_ind):
		# print(clearcol)
		repcols = index_cols[:-clearcol]
		# print(repcols)
		# print(index_cols[-clearcol - 1])
		clearcolname = repcols[-1]
		x.loc[x[repcols].duplicated(), clearcolname] = ''
	x.to_csv(f"{csv_basename}_masked.csv", index=False)
	return x


def compare_heuristics():
	# m = 2 # i = {1,...,m} policies
	# n = 1 # j = {1,...,n} levels
	# T = 40 # t = {1,...,T} periods

	individual_multiplier = 1000
	days_per_period = 7
	cost_multiplier = 1000000
	N = 300000
	KI0 = 0.0000006
	KD0 = 0.015
	KR0 = 0.03

	solver = "DICOPT"
	optGap = {
		"dicopt": 0.01,
		"baron": .05, # 0.10,
		"bonmin": 1,
		"gurobi": 0.8
		}[solver.lower()]
	solve_from_binary = solver.lower() in ('bonmin', 'couenne')
	use_gurobipy = False
	bilinear = False # True if solver.lower() == "gurobi" else False
	pruning = False
	allow_multiple_policies_per_period = True
	if pruning:
		# sets to False automatically if pruning, considering assortments
		allow_multiple_policies_per_period = False
	multistart = False

	max_time = 100

	no_solution_heuristic, heuristics = get_all_heuristics()

	Tvals = [20, 50, 80, 110]
	ms = list(range(1, 9, 2))
	sols_df = pd.DataFrame()
	for T in Tvals:
		for nPolicies_subset in ms:
			policies = get_all_policies(T=T, pruning=pruning, nPolicies_subset=nPolicies_subset)
			model_kwargs, param_kwargs, solver_params = get_all_kwargs(
				allow_multiple_policies_per_period, T, individual_multiplier, days_per_period, N,
				cost_multiplier, policies, KI0, KR0, KD0, solver, bilinear, solve_from_binary,
				optGap, max_time, multistart
				)
			no_policy_sol = no_solution_heuristic(solver_params, param_kwargs, model_kwargs)
			print(f"GOT 'NO POLICY' SOLUTION")

			for heuristic in heuristics:
				sol = heuristic(solver_params, param_kwargs, model_kwargs)
				row = dict(Heuristic=heuristic.__name__, **sol.to_dict(include_model_stuff=True))
				sols_df = sols_df.append(row, ignore_index=True)

	# sols_df.sort_values(by=["Heuristic", "T", "m", "n", "nPolicies"], inplace=True)
	get_csv_with_indexing(sols_df, ["Heuristic", "T", "m"], "heuristic_comparison_df")
	get_csv_with_indexing(sols_df, ['T', 'm', 'Heuristic'], "heuristic_comparison_df_by_heuristic")


def fix_df_heuristic_names():
	unwanted_kwargs = (
		"use_gurobipy", "only_policy", "get_no_policy_solution", 'tee', 'only_no_policy'
		)
	for dfname in ("heuristic_comparison_df", "heuristic_comparison_df_by_heuristic"):
		for extra in ("_masked", ""):
			fname = f"{dfname}{extra}.csv"
			df = pd.read_csv(fname)
			for i, row in df.iterrows():
				heuristic_name = df.loc[i, "Heuristic"]
				if hasattr(heuristic_name, "replace"):
					for unwanted_kwarg in unwanted_kwargs:
						for boolval in (True, False):
							heuristic_name = heuristic_name.replace(
								f"_{unwanted_kwarg}_{boolval}", ""
								)
				df.loc[i, "Heuristic"] = heuristic_name
			df.to_csv(fname, index=False)


def fix_title(ax):
	ax.set_title("=\n".join(ax.get_title().rsplit("=", 1)))
	if "local_search" in ax.get_title():
		ax.set_title("\nlocal_search".join(ax.get_title().rsplit("_local_search", 1)))
	ax.set_title(ax.get_title(), fontsize=4)


# This is equivalent to df.melt(...)!! To check equivalence:
# for k in df_new2.columns:
#		print(f"{k} : {all(df_new2[k].values == df_new[k].values)}")
def combine_columns_add_key_old(df, old_col1, old_col2, new_col, key_col, key1, key2):
	df1, df2 = df.copy(), df.copy()
	df1[new_col] = df1[old_col1]
	df2[new_col] = df2[old_col2]
	df1[key_col] = key1
	df2[key_col] = key2
	df_new = pd.concat([df1, df2])
	df_new.drop([old_col1, old_col2], axis=1, inplace=True)
	return df_new


def combine_columns_add_key(df, old_col1, old_col2, new_col, key_col, key1, key2):
	value_vars = [old_col1, old_col2]
	id_vars = [col for col in df.columns if col not in value_vars]
	df_new = pd.melt(
		df, id_vars=id_vars, value_vars=value_vars, var_name=key_col, value_name=new_col
		)
	return df_new


def report_equality(df1, df2, df3=None):
	for k in df_combined.columns:
		equality = f"{k} in (df1, df2): {all(df1[k].values == df2[k].values)}"
		if df3 is not None:
			equality += f", and (df1, df3): {all(df1[k].values == df3[k].values)}"
		print(equality)


def facetgrid_two_axes(data, color, label):
	df = data
	ax = plt.gca()
	ax2 = ax.twinx()
	# ax2.set_ylabel('Second Axis!')
	sns.lineplot(data=df, x="T", y="objVal", palette=['red'], label="Objective Value", ax=ax)
	sns.lineplot(data=df, x="T", y="timeToSolve", palette=['blue'], label="Time To Solve", ax=ax2)


if __name__ == "__main__":
	# compare_heuristics()
	mpl.use('PDF')
	df = pd.read_csv('heuristic_comparison_df.csv')

	df_combined = combine_columns_add_key(
		df, "timeToSolve", "objVal", "performance", "performance_type", "timeToSolve", "objVal"
		)

	if True:
		g = sns.FacetGrid(df, col="m", hue="Heuristic")
		g.map_dataframe(sns.lineplot, x="T", y="timeToSolve")
		for ax in g.axes.flatten():
			# fix_title(ax)
			ax.set(yscale="log")
		g.add_legend()
		g.fig.suptitle('Time to Solve for Different Heuristics', fontsize=16)
		# plt.show(block=False)
		# plt.tight_layout()
		g.fig.subplots_adjust(top=1.5)
		g.savefig(f"figures/timeToSolve_vs_m_T_heuristics_hues.pdf")

	if True:
		g = sns.FacetGrid(df, col="m", hue="Heuristic")
		g.map_dataframe(sns.lineplot, x="T", y="objVal")
		# for ax in g.axes.flatten():
		# fix_title(ax)
		# ax.set(yscale="log")
		g.add_legend()
		g.fig.suptitle('Objective Value for Different Heuristics', fontsize=16)
		# plt.show(block=False)
		# plt.tight_layout()
		g.fig.subplots_adjust(top=1.5)
		g.savefig(f"figures/objVal_vs_m_T_heuristics_hues.pdf")

	if True:
		g = sns.FacetGrid(df_combined, col="m", row="performance_type", hue="Heuristic")
		g.map_dataframe(sns.lineplot, x="T", y="performance")
		for ax in g.axes.flatten():
			# fix_title(ax)
			if 'timeToSolve' in ax.get_title():
				ax.set(yscale="log")
		g.add_legend()
		g.fig.suptitle('Performancefor Different Heuristics', fontsize=16)
		# plt.show(block=False)
		# plt.tight_layout()
		# g.fig.subplots_adjust(top=2)
		g.savefig(f"figures/performance_vs_m_T_heuristics_hues.pdf")

	# if True:
	# 	g = sns.FacetGrid(df, col="Heuristic", row="m")
	# 	g.map_dataframe(sns.lineplot, x="T", y="objVal", palette=['red'], label="Objective Value")
	# 	g.map_dataframe(
	# 		sns.lineplot, x="T", y="timeToSolve", palette=['blue'], label="Time To Solve"
	# 		)
	# 	g.add_legend()
	# 	g.fig.suptitle('Time to Solve for Different Heuristics', fontsize=16)
	# 	for ax in g.axes.flatten():
	# 		fix_title(ax)
	# 		ax.set(yscale="log")
	# 	# plt.show(block=False)
	# 	g.add_legend()
	# 	# plt.tight_layout()
	# 	g.fig.subplots_adjust(top=0.90)
	# 	g.savefig(f"figures/timeToSolve_vs_m_T_heuristics.pdf")

	if True:
		g = sns.FacetGrid(df_combined, col="Heuristic", row="m", hue="performance_type")
		g.map_dataframe(sns.lineplot, x="T", y="performance")
		g.fig.suptitle('Time to Solve for Different Heuristics', fontsize=16)
		for ax in g.axes.flatten():
			fix_title(ax)
			ax.set(yscale="log")
		# plt.show(block=False)
		g.add_legend()
		# plt.tight_layout()
		g.fig.subplots_adjust(top=0.90)
		g.savefig(f"figures/timeToSolve_vs_m_T_heuristics_combined.pdf")

print("\a")

# if True:
# 	import importlib
# 	importlib.reload(process_results)
# 	from process_results import plot_results
