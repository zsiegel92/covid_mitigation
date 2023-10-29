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
from solvers_and_heuristics import get_params, get_model, fix_and_unfix_range, get_heuristic_solver, solve_and_process, solve_and_process_index, solve_and_process_lookahead, solve_and_process_local_search, solve_and_process_lagrangian, solve_and_process_early_stopping, solve_and_process_no_policy

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

	T = 100
	policies = get_all_policies(
		T=T,
		pruning=False,
		nPolicies_subset=None,
		trial_cost_multiplier=1,
		trial_effect_multiplier=1,
		)
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
	vax_param_kwargs = None
	save_parameters(param_kwargs, vax_param_kwargs=vax_param_kwargs)
	overleaf = "/Users/zach/UCLA Anderson Dropbox/Zach Siegel/UCLA_classes/research/covid_mitigation/writing/covid_mitigation"
	dest2 = f'{overleaf}/content_to_share/parameters.txt'
	save_parameters(
		param_kwargs, dest2, pickling=False, latex=True, vax_param_kwargs=vax_param_kwargs
		)

print("\a")

# https://stackoverflow.com/questions/7271082/how-to-reload-a-modules-function-in-python
# if False:
# 	import importlib
# 	importlib.reload(process_results)
# 	from process_results import plot_results
