# cd "/Users/zach/UCLA Anderson Dropbox/Zach Siegel/UCLA_classes/research/covid_mitigation" && python -m IPython -i "pyomo_implementation.py" && exit
import pandas as pd
import numpy as np
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

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

from problem_utils import get_all_kwargs, get_all_policies
# flake8: warning
# E123 - closing bracket does not match indentation of opening bracket's line


def time_trials(
	policies, model_kwargs, param_kwargs, solver_params, Tmin=10, Tmax=40, Tstep=10, plotting=True
	):
	trial_sols = {}
	Tvals = list(reversed(range(Tmin, Tmax, Tstep)))
	for T in Tvals:
		# if True:
		print(f"\nTRIAL: T={T}\n")

		policies = policies.new_T(T)
		# self.C_ = C
		# self.P_ = P
		# self.C_setup_ = C_setup
		param_kwargs['T'] = T
		param_kwargs['policies'] = policies

		sols = solve_and_process(
			solver_params, param_kwargs, model_kwargs, get_no_policy_solution=True
			)
		sol = sols[False]
		for force_no_policy, sol in sols.items():
			print(
				f"Force no policy: {force_no_policy}\n\tStatus: {sol.status}; UB: {sol.ub}; lb: {sol.lb}; %Gap: {(sol.ub-sol.lb)/sol.ub}"
				) #note the denominator is LB in BARON and UB in Gurobi output
		del sol.model
		trial_sols[T] = sol

	if plotting:
		policy_descr = policies.summary()
		extra = ('_pruned' if policies.is_pruned else '') + ('_multistart' if multistart else '') + (
			f"_T_{Tmin}_{max(trial_sols.keys())}_by{Tstep}"
			) + (f'_{policies.num_assortments}assortments')
		plot_time_trials(
			trial_sols,
			param_kwargs,
			solver_params=solver_params,
			model_kwargs=model_kwargs,
			policies=policies,
			extra=extra,
			policy_descr=policy_descr
			)

	return trial_sols


if __name__ == "__main__":
	trial_sols = time_trials(
		policies, model_kwargs, param_kwargs, solver_params, Tmin=10, Tmax=40, Tstep=10,
		plotting=True
		)
