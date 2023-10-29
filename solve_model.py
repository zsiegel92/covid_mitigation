# cd "/Users/zach/UCLA Anderson Dropbox/Zach Siegel/UCLA_classes/research/covid_mitigation" && python -m IPython -i "pyomo_implementation.py" && exit
import pandas as pd
import numpy as np
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from datetime import datetime

# for debugging
import pyomo.environ as pyo
import pyomo
# import gurobipy as gp

from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo
from model import get_SIRD_model
from model_bilinear import get_SIRD_model_bilinear
from model_gurobi_capable import get_SIRD_model_gurobi_capable
# from model_gurobipy import get_model_and_solve_gurobi
from model_no_policy import no_policy_model
from policy_helpers import Policy, PolicyList

from process_results import get_results_df, plot_results, Solution, plot_time_trials
from solvers_and_heuristics import get_params, get_model, fix_and_unfix_range, get_heuristic_solver, solve_and_process, solve_and_process_index, solve_and_process_lookahead, solve_and_process_local_search, solve_and_process_lagrangian, solve_and_process_early_stopping, solve_and_process_no_policy, solve_and_process_quadratic, solve_and_process_simple_index, solve_and_process_vaccination

from problem_utils import get_all_kwargs, get_all_policies, save_parameters, local_search

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")

if __name__ == "__main__":
	# test_solvers()
	# test_gurobi_bilinear()
	# m = 2 # i = {1,...,m} policies
	# n = 1 # j = {1,...,n} levels

	# T = 40 # t = {1,...,T} periods
	# T = 80
	individual_multiplier = 1000
	days_per_period = 7
	cost_multiplier = 1000000
	N = 300000
	# KI0 = 0.0000006
	# KD0 = 0.015
	# KR0 = 0.03

	# Threads= 1: 55s
	# Threads=24:
	saving = True

	solver = "baron"
	optGap = {
		"dicopt": 0.05,
		"baron": .8, # 0.10,
		"gurobi": 0.8,
		"bonmin": 0.5,
		"couenne": 1
		}[solver.lower()]
	# solve_from_binary = solver.lower() in ('bonmin', 'couenne')
	solve_from_binary = False

	no_policy_heuristic = get_heuristic_solver(solve_and_process_no_policy)
	# heuristic = None
	# heuristic = get_heuristic_solver(solve_and_process_lookahead,w=12,truncate_costs=5,tee=True,**local_search)
	# heuristic = get_heuristic_solver(solve_and_process_index, tee=True)
	# heuristic = get_heuristic_solver(solve_and_process_early_stopping, optGap=0.9)

	use_gurobipy = False
	# bilinear = True if solver.lower() == "gurobi" else False
	bilinear = False
	pruning = False
	allow_multiple_policies_per_period = True # False automatically if pruning or considering assortments
	multistart = False
	multistart_kwargs = dict(
		iterations=-1, HCS_max_iterations=20
		) #-1-> high confidence stopping. Default 10
	max_time = None

	no_policy_heuristic = get_heuristic_solver(solve_and_process_no_policy)
	early_stopping_heuristic = get_heuristic_solver(
		solve_and_process_early_stopping,
		optGap=.8,
		)
	lagrangian_heuristic = get_heuristic_solver(
		solve_and_process_lagrangian,
		threshold=0.075,
		use_smart_stepsize=True,
		L1_optGap=0.025,
		L2_optGap=0.075,
		L2_max_time=500,
		)
	quadratic_heuristic = get_heuristic_solver(solve_and_process_quadratic)
	simple_index_heuristic_blocksize_1 = get_heuristic_solver(
		solve_and_process_simple_index, block_size=1
		)
	simple_index_heuristic_blocksize_7 = get_heuristic_solver(
		solve_and_process_simple_index, block_size=7
		)

	T_vaxs = (1, 20, 40, 60, 80, 99)
	S0_antivax_factors = (0.2, 0.35, 0.4)
	KVs = (.05,)

	# T_vaxs = (10,)
	# S0_antivax_factors = (0.25,)
	# KVs = (.05,)
	vaccinated_heuristics = [
		get_heuristic_solver(
			solve_and_process_vaccination, T_vax=T_vax, S0_antivax_factor=S0_antivax_factor, KV=KV
			) for T_vax in T_vaxs for S0_antivax_factor in S0_antivax_factors for KV in KVs
		]

	T = 100

	# heuristics = (
	# 	simple_index_heuristic_blocksize_1,
	# 	simple_index_heuristic_blocksize_7,
	# 	early_stopping_heuristic,
	# 	lagrangian_heuristic,
	# 	quadratic_heuristic,
	# 	)

	heuristics = vaccinated_heuristics

	sols = {}
	for heuristic in heuristics:
		for zero_switching_costs in (False,): #(True, False)
			policies = get_all_policies(
				T=T,
				pruning=False,
				nPolicies_subset=None,
				trial_cost_multiplier=1,
				trial_effect_multiplier=1,
				zero_switching_costs=zero_switching_costs
				)
			print(policies)
			if pruning:
				policies = policies.pruned()
				print(policies)
				allow_multiple_policies_per_period = False

			model_kwargs, param_kwargs, solver_params = get_all_kwargs(
				allow_multiple_policies_per_period,
				T,
				individual_multiplier,
				days_per_period,
				N,
				cost_multiplier,
				policies,
				solver,
				bilinear,
				solve_from_binary,
				optGap,
				max_time,
				multistart,
				)
			cumulative_cols = []
			if "vaccin" in heuristic.__name__:
				vaccinations = True
			if vaccinations:
				sols_heuristic = heuristic(solver_params, param_kwargs, model_kwargs)
				sol_no_policy = sols_heuristic[True]
				sol_heuristic = sols_heuristic[False]
			else:
				sol_no_policy = no_policy_heuristic(solver_params, param_kwargs, model_kwargs)
				sol_heuristic = heuristic(solver_params, param_kwargs, model_kwargs)

			if "quadratic" in heuristic.__name__:
				extra_objVal = (
					f"Quadratic approximation objective", sol_heuristic.quadratic_solution
					)
			else:
				extra_objVal = ()

			plot_results(
				sol_heuristic,
				sol_no_policy,
				param_kwargs,
				solver_params=solver_params,
				model_kwargs=model_kwargs,
				policies=policies,
				one_table_row=True,
				plot_cost_scale=100,
				figsize=(40, 4 * (1 + len(policies))), #width, height
				height_ratios=(90, 1 + 2 * len(policies)),
				tight_layout_padding=8,
				hspace=0.5,
				auto_set_fontsize=False,
				table_fontsize=17,
				first_row_fontsize=12,
				last_row_fontsize=7,
				first_col_fontsize=12,
				extra=('_pruned' if policies.is_pruned else '') +
				(f"_{heuristic.__name__}"
					if heuristic else "") + ("_zeroSwitchingCosts" if zero_switching_costs else ""),
				extra_extra='multistart' if multistart else '',
				title_extra=f"Solved using {heuristic.__name__}" if heuristic else '',
				xticks_at_policy_changes=True,
				legend_fontsize=18,
				title_fontsize=18,
				figsize_scale_factor=0.7,
				linewidth=5.5,
				markersize=2,
				extra_objVal=extra_objVal,
				zero_switching_costs=zero_switching_costs,
				vaccination=vaccinations
				)

print("\a")

# https://stackoverflow.com/questions/7271082/how-to-reload-a-modules-function-in-python
# if False:
# 	import importlib
# 	importlib.reload(process_results)
# 	from process_results import plot_results
