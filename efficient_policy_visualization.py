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

from problem_utils import get_all_kwargs, get_all_policies, save_parameters

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")

if __name__ == "__main__":

	individual_multiplier = 1000
	days_per_period = 7
	cost_multiplier = 1000000
	N = 300000

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
	pruned = policies.pruned()
	df = policies.assortment_df()
	# df.loc[policies.keeper_assortments, 'efficient'] = True
	df_efficient = df.loc[df['efficient']]
	df = df.sort_values(by='efficient')

	if True:
		predicted = ['C_policy', 'C_setup', 'C_switch']
		deg = 2
		coefs = np.polyfit(
			df_efficient['P'].values,
			df_efficient[predicted].values,
			deg=deg,
			)
		coefs = pd.DataFrame(coefs, columns=predicted)
		get_coef = lambda col, power: coefs.loc[deg - power, col]
		minP = min(df['P'].values)
		Pvals = np.linspace(minP * .85, 1, 1000)
		predictions = pd.DataFrame({
			**{
				col: [sum(get_coef(col, i) * P**i for i in range(deg + 1)) for P in Pvals] for col in predicted
				}, "P": Pvals
			})
	df['efficient'] = df['efficient'].map({
		True: 'Efficient',
		False: 'Inefficient'
		}) #only run this once

	if True:

		for feature in ("Policy", "Setup", "Switch"):
			featureCol = f'C_{feature.lower()}'
			featureCol_latex = f"C_{{{feature.lower()}}}"
			get_coef = lambda power: coefs.loc[deg - power, featureCol]
			cdot = '\\cdot'
			approx = '\\approx'
			line_of_best_fit_string = f"${featureCol_latex}{approx}" + " + ".join(
				f"{get_coef(i):.1f}{f'{cdot} P^{i}' if i > 0 else ''}" for i in range(deg + 1)
				) + "$".replace("+ -", "-")

			plt.clf()
			plt.close('all')
			ax = sns.scatterplot(data=df, y=featureCol, x='P', hue='efficient')
			ax.plot(
				predictions['P'],
				predictions[featureCol],
				color='red',
				label=f'{line_of_best_fit_string}', #${feature.lower()}$
				)
			ax.legend(fontsize='x-small')
			ax.set_ylabel(f"${featureCol_latex}$")
			ax.set_xlabel(f"$P$")
			ax.set_title(
				f"Cost of \"{feature}\" vs. Policy Effectiveness\n", fontdict={'fontsize': 12}
				)
			# plt.show(block=False)

			extkwargs = {'pdf': {}, 'png': {'dpi': 600}}
			for ext in ('png', 'pdf'):
				folder = "figures/policy_visualization"
				fname = f'efficient_policies_{feature.lower()}.{ext}'
				fullpath = f"{folder}/{fname}"
				plt.savefig(fullpath, **extkwargs.get(ext))
				if ext == 'png':
					overleaf = "/Users/zach/UCLA Anderson Dropbox/Zach Siegel/UCLA_classes/research/covid_mitigation/writing/covid_mitigation"
					with open(fullpath, 'rb') as src, open(f"{overleaf}/figures/{fname}",
															'wb') as dst:
						dst.write(src.read())
				# break
			# break

print("\a")

# https://stackoverflow.com/questions/7271082/how-to-reload-a-modules-function-in-python
# if False:
# 	import importlib
# 	importlib.reload(process_results)
# 	from process_results import plot_results
