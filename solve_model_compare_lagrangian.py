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
import gurobipy as gp

from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo
from model import get_SIRD_model
from model_bilinear import get_SIRD_model_bilinear
from model_gurobi_capable import get_SIRD_model_gurobi_capable
from model_gurobipy import get_model_and_solve_gurobi
from model_no_policy import no_policy_model
from policy_helpers import Policy, PolicyList

from process_results import get_results_df, plot_results, Solution, plot_time_trials
from solvers_and_heuristics import get_params, get_model, fix_and_unfix_range, get_heuristic_solver, solve_and_process, solve_and_process_index, solve_and_process_lookahead, solve_and_process_local_search, solve_and_process_lagrangian, solve_and_process_early_stopping, solve_and_process_no_policy, solve_and_process_quadratic, solve_and_process_simple_index

from problem_utils import get_all_kwargs, get_all_policies, save_parameters, get_row_from_solutions, timestamp, manually_defined_cols, local_search

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

	Tvals = (150, 100, 50)
	# Tvals = (7,)
	# Tvals = (2, 5, 10, 20)
	# Tvals = (150,)
	# trial_cost_multipliers = (1, 0.5, 0.75, 1.25, 1.5)
	# trial_effect_multipliers = (1, 0.5, 0.75, 1.25, 1.5)
	trial_cost_multipliers = (1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5)
	trial_effect_multipliers = (1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5)

	# Tvals = (7, 14)
	# trial_cost_multipliers = (1,)
	# trial_effect_multipliers = (1,)

	sols_df = pd.DataFrame()

	for T in Tvals:
		for trial_cost_multiplier in trial_cost_multipliers:
			for trial_effect_multiplier in trial_effect_multipliers:
				if not ((trial_cost_multiplier == 1) or (trial_effect_multiplier == 1)):
					continue

				# T = 150
				# trial_cost_multiplier = 0.7
				# trial_effect_multiplier = 1
				print(f"DOING TRIAL {trial_cost_multiplier=}, {trial_effect_multiplier=}, {T=}")
				policies = get_all_policies(
					T=T,
					pruning=False,
					nPolicies_subset=None,
					trial_cost_multiplier=trial_cost_multiplier,
					trial_effect_multiplier=trial_effect_multiplier,
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
				sol_no_policy = no_policy_heuristic(solver_params, param_kwargs, model_kwargs)
				sol_early_stopping = early_stopping_heuristic(
					solver_params, param_kwargs, model_kwargs
					)
				sol_lagrangian = lagrangian_heuristic(solver_params, param_kwargs, model_kwargs)

				sol_quadratic_heuristic = quadratic_heuristic(
					solver_params, param_kwargs, model_kwargs
					)
				sol_simple_index_heuristic_blocksize_1 = simple_index_heuristic_blocksize_1(
					solver_params, param_kwargs, model_kwargs
					)
				sol_simple_index_heuristic_blocksize_7 = simple_index_heuristic_blocksize_7(
					solver_params, param_kwargs, model_kwargs
					)

				row, cumulative_cols = get_row_from_solutions(
					sol_no_policy,
					sol_early_stopping,
					sol_lagrangian,
					solver_params,
					trial_cost_multiplier,
					trial_effect_multiplier,
					lagrangian_heuristic,
					sol_quadratic_heuristic,
					sol_simple_index_heuristic_blocksize_1,
					sol_simple_index_heuristic_blocksize_7,
					)

				sols_df = sols_df.append(row, ignore_index=True)
				# raise
			# sols_df = sols_df[manually_defined_cols + cumulative_cols]
	if saving:
		sols_df.to_csv(
			f'output/lagrangian_heuristic_comparison_time_{timestamp}_(T={"_".join(str(T) for T in Tvals)}).csv',
			index=False
			)
		sols_df.to_csv(
			f'output/lagrangian_heuristic_comparison_(T={"_".join(str(T) for T in Tvals)}).csv',
			index=False
			)

print("\a")

# https://stackoverflow.com/questions/7271082/how-to-reload-a-modules-function-in-python
# if False:
# 	import importlib
# 	importlib.reload(process_results)
# 	from process_results import plot_results
