# cd "/Users/zach/UCLA Anderson Dropbox/Zach Siegel/UCLA_classes/research/covid_mitigation" && python -m IPython -i "pyomo_implementation.py" && exit
import pandas as pd
import numpy as np
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from datetime import datetime

import pyomo.environ as pyo
import pyomo
import gurobipy as gp

from model import get_SIRD_model
from model_lagrangians import fix_full_model, get_L1_model, get_L2_model
from model_no_policy import no_policy_model

from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo, fix_all_components
from policy_helpers import Policy, PolicyList
from process_results import get_results_df, plot_results, Solution, plot_time_trials
from solvers_and_heuristics import get_params, get_model, fix_and_unfix_range, get_heuristic_solver, solve_and_process, solve_and_process_index, solve_and_process_lookahead, solve_and_process_local_search, solve_and_process_lagrangian, solve_and_process_early_stopping, solve_and_process_no_policy

from problem_utils import get_all_kwargs, get_all_policies

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")


def prepare_L2():
	individual_multiplier = 1000
	days_per_period = 7
	cost_multiplier = 1000000
	N = 300000

	# Threads= 1: 55s
	# Threads=24:
	saving = False

	solver = "baron"
	optGap = {
		"dicopt": 0.05,
		"baron": .8, # 0.10,
		"gurobi": 0.8,
		"bonmin": 0.5,
		"couenne": 1
		}[solver.lower()]
	# solve_from_binary = solver.lower() in ('bonmin', 'couenne')
	solve_from_binary = True
	local_search = dict(
		do_local_search=True,
		local_search_kwargs=dict(
			w=2,
			iterations=3,
			iterate_until_no_improvement=False,
			optGap=0.3,
			tee=False,
			)
		)
	no_policy_heuristic = get_heuristic_solver(solve_and_process_no_policy)

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

	T = 120
	# Tvals = (2, 5, 10, 20)
	# Tvals = (150,)

	policies = get_all_policies(
		T=T,
		pruning=False,
		nPolicies_subset=None,
		trial_cost_multiplier=1,
		trial_effect_multiplier=1,
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

	sol_no_policy = no_policy_heuristic(solver_params, param_kwargs, model_kwargs)

	L1_optGap = 0.025
	L2_optGap = 0.05
	L2_max_time = 20

	model_full = get_model(*get_params(**param_kwargs), **model_kwargs)
	llambda = {t: 0 for t in model_full.t}

	model_L1 = get_L1_model(model_full, llambda)
	model_L2 = get_L2_model(model_full, llambda)
	# breakpoint()
	print("SOLVING L1")

	return model_L2, solver_params, L2_max_time, L2_optGap


def joint_quasiconvexity_plots():
	model_L2, solver_params, L2_max_time, L2_optGap = prepare_L2()
	P_lower_bound = model_L2.P_lower_bound
	nTrials = 8
	assert nTrials % 2 == 0
	nPerTrial = 100
	thetavals = np.linspace(0, 1, nPerTrial).tolist()
	conv_counterexamples = {}
	conc_counterexamples = {}
	non_counterexamples = {}
	trial = 0
	while True:
		trial += 1
		P1 = {k: np.random.uniform(P_lower_bound, 1) for k in model_L2.P}
		P2 = {k: np.random.uniform(P_lower_bound, 1) for k in model_L2.P}
		yvals = []
		for theta in thetavals:
			P = {k: (1-theta) * P1[k] + theta * P2[k] for k in P1}
			for k, component in model_L2.P.items():
				component.fix(P[k])

			sol_L2 = Solution(
				*solve_pyomo(
					model_L2,
					**{
						**solver_params,
						**dict(
							tee=False,
							max_time=L2_max_time,
							)
						},
					optGap_override=L2_optGap,
					),
				get_DF=False
				)
			yvals.append(sol_L2.objVal)

		max_theta = thetavals[yvals.index(max(yvals))]
		min_theta = thetavals[yvals.index(min(yvals))]
		if 0.1 < min_theta < 0.9:
			conc_counterexamples[trial] = yvals
			print("Found quasiconcavity counterexample!")
		elif 0.1 < max_theta < 0.9:
			conv_counterexamples[trial] = yvals
			print("Found quasiconvexity counterexample!")
		else:
			non_counterexamples[trial] = yvals
		# breakpoint()
		# if max(yvals) > max(yvals[0], yvals[1]) + 500:
		# 	quasiconvex = False
		# 	nqc += 1
		# 	print("Found quasiconvexity counterexample!")
		# if min(yvals) < min(yvals[0], yvals[1]) - 2000:
		# 	quasiconcave = False
		# 	nqc += 1
		# 	print("Found quasiconcavity counterexample!")
		# fix_all_components(model_full.y, model_L1.y)
		if (len(conc_counterexamples) >= 1) and (
			len(conv_counterexamples) >= 1
			) and (len(conv_counterexamples) + len(conc_counterexamples) + len(non_counterexamples) >=
				nTrials):
			break
	# fix_all_components(model_full.y, model_L1.y)
	solutions = {}
	while len(solutions) < nTrials:
		if len(solutions) < nTrials:
			if len(conc_counterexamples) > 0:
				k, v = conc_counterexamples.popitem()
				solutions[k] = v
		if len(solutions) < nTrials:
			if len(conv_counterexamples) > 0:
				k, v = conv_counterexamples.popitem()
				solutions[k] = v
		if len(solutions) < nTrials:
			if len(non_counterexamples) > 0:
				k, v = non_counterexamples.popitem()
				solutions[k] = v
	if True:
		plt.clf()
		plt.close("all")
		fig, axs = plt.subplots(2, nTrials // 2, figsize=((nTrials/2) * 5, 5 * 2))
		for ax, (trial, solution) in zip(axs.flatten(), solutions.items()):
			sns.lineplot(thetavals, solution, ax=ax)
			minsol, maxsol = min(solution), max(solution)
			ax.plot(
				[min(thetavals), max(thetavals)],
				[minsol, maxsol],
				color='green',
				# marker='o',
				linestyle='dashed',
				linewidth=2,
				# markersize=12,
				)
			ax.plot(
				[min(thetavals), max(thetavals)],
				[maxsol, minsol],
				color='red',
				# marker='o',
				linestyle='dashed',
				linewidth=2,
				# markersize=12,
				)
			sns.lineplot()
			ax.set_title(f"Trial {trial}")
			ax.set_xlabel("$\\theta$")
			ax.set_ylabel("Optimal Value of Subproblem")
		fig.suptitle(
			'Optimal Value of $L2(\\lambda)$ Subproblem ($V$)\nwith fixed $P=(1-\\theta) P_1 + \\theta P_2$\nfor random $P_1$ and $P_2$ in each trial',
			fontsize=20
			)
		fig.tight_layout()
		plt.savefig(f"figures/L2_quasiconvexity_test_{timestamp}.pdf")
		plt.savefig(
			"/Users/zach/UCLA Anderson Dropbox/Zach Siegel/UCLA_classes/research/covid_mitigation/writing/covid_mitigation/figures/L2_quasiconvexity_test.pdf",
			)
		plt.show(block=False)


def componentwise_quasiconvexity_plots():
	model_L2, solver_params, L2_max_time, L2_optGap = prepare_L2()
	P_lower_bound = model_L2.P_lower_bound
	nTrials = 8
	assert nTrials % 2 == 0
	nPerTrial = 100
	thetavals = np.linspace(0, 1, nPerTrial).tolist()
	while True:
		solutions = {trial: [] for trial in range(nTrials)}
		quasiconvex = True
		quasiconcave = True
		for trial in range(nTrials):

			P1 = {k: np.random.uniform(P_lower_bound, 1) for k in model_L2.P}
			P2_t = np.random.uniform(P_lower_bound, 1)
			index_to_vary = np.random.choice(model_L2.P.keys())

			for theta in thetavals:
				P = {k: P1[k] for k in P1}
				P[index_to_vary] = (1-theta) * P1[index_to_vary] + theta*P2_t
				for k, component in model_L2.P.items():
					component.fix(P[k])

				sol_L2 = Solution(
					*solve_pyomo(
						model_L2,
						**{
							**solver_params,
							**dict(
								tee=False,
								max_time=L2_max_time,
								)
							},
						optGap_override=L2_optGap,
						),
					get_DF=False
					)
				solutions[trial].append(sol_L2.objVal)
			yvals = solutions[trial]
			if max(yvals) > max(yvals[0], yvals[1]):
				quasiconvex = False
				print("Found quasiconvexity counterexample!")
			if min(yvals) < min(yvals[0], yvals[1]):
				quasiconcave = False
				print("Found quasiconcavity counterexample!")
		# fix_all_components(model_full.y, model_L1.y)
		if (not quasiconvex) and (not quasiconcave):
			break
	if True:
		plt.clf()
		plt.close("all")
		fig, axs = plt.subplots(2, nTrials // 2, figsize=((nTrials/2) * 5, 5 * 2))
		for ax, (trial, solution) in zip(axs.flatten(), solutions.items()):
			sns.lineplot(thetavals, solution, ax=ax)
			minsol, maxsol = min(solution), max(solution)
			ax.plot(
				[min(thetavals), max(thetavals)],
				[minsol, maxsol],
				color='green',
				# marker='o',
				linestyle='dashed',
				linewidth=2,
				# markersize=12,
				)
			ax.plot(
				[min(thetavals), max(thetavals)],
				[maxsol, minsol],
				color='red',
				# marker='o',
				linestyle='dashed',
				linewidth=2,
				# markersize=12,
				)
			ax.set_title(f"Trial {trial}")
			ax.set_xlabel("$\\theta$")
			ax.set_ylabel("Optimal Value of Subproblem")
		fig.suptitle(
			'Optimal Value of $L2(\\lambda)$ Subproblem ($V$)\nwith fixed $P=(1-\\theta) P_1 + \\theta P_2$\nfor random $P_1$ and $P_2$ that differ in a single component in each trial',
			fontsize=20
			)
		fig.tight_layout()
		# plt.savefig("figures/L2_quasiconvexity_test_componentwise.pdf")
		plt.savefig(f"figures/L2_quasiconvexity_test_componentwise_{timestamp}.pdf")
		plt.savefig(
			"/Users/zach/UCLA Anderson Dropbox/Zach Siegel/UCLA_classes/research/covid_mitigation/writing/covid_mitigation/figures/L2_quasiconvexity_test_componentwise.pdf",
			)
		plt.show(block=False)


if __name__ == "__main__":
	joint_quasiconvexity_plots()
	# componentwise_quasiconvexity_plots()

print("\a")

# https://stackoverflow.com/questions/7271082/how-to-reload-a-modules-function-in-python
# if False:
# 	import importlib
# 	importlib.reload(process_results)
# 	from process_results import plot_results
