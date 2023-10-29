import pickle
import jinja2
from datetime import datetime

from policy_helpers import Policy, PolicyList
from process_results import get_results_df, plot_results, Solution, plot_time_trials

timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")

manually_defined_cols = [
	'T',
	'm',
	'n',
	'trial_cost_multiplier',
	'trial_effect_multiplier',
	'no_policy_objVal',
	"solver_objVal",
	"lagrangian_objVal",
	"lagrangian_LB",
	"lagrangian_LB_guaranteed",
	"solver_LB",
	"solver_optGap",
	"lagrangian_optGap",
	"lagrangian_desired_L1_optGap",
	"lagrangian_desired_L2_optGap",
	"lagrangian_desired_optGap",
	"solver_desired_optGap",
	"lagrangian_L1_objVal",
	"lagrangian_L2_objVal",
	# "solver_vs_lagrangian_lb_gap",
	"solver_timeToSolve",
	"lagrangian_timeToSolve",
	"lagrangian_timeToSolve_L1_total",
	"lagrangian_timeToSolve_L2_total",
	'nConstraints',
	'nPolicies',
	'nVariables',
	"lagrangian_solution_path",
	'lagrangian_cumulative_cost',
	'lagrangian_cumulative_deaths',
	'lagrangian_cumulative_disease_cost',
	'lagrangian_cumulative_policy_cost',
	'lagrangian_cumulative_recovered',
	'lagrangian_solution_path',

	# 'no_policy_cumulative_cost',
	# 'no_policy_cumulative_deaths',
	# 'no_policy_cumulative_disease_cost',
	# 'no_policy_cumulative_policy_cost',
	# 'no_policy_cumulative_recovered',
	'quadratic_approx_objVal',
	'quadratic_heuristic_objVal',
	'simple_index_blocksize_1_objVal',
	'simple_index_blocksize_7_objVal',
	'solver_cumulative_cost',
	'solver_cumulative_deaths',
	'solver_cumulative_disease_cost',
	'solver_cumulative_policy_cost',
	'solver_cumulative_recovered',
	]

local_search = dict(
	do_local_search=True,
	local_search_kwargs=dict(
		w=2,
		iterations=3,
		iterate_until_no_improvement=False,
		optGap=0.3,
		tee=True,
		)
	)


def get_all_kwargs(
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
	):
	multistart_kwargs = dict(
		iterations=-1, HCS_max_iterations=20
		) #-1-> high confidence stopping. Default 10  # noqa: E123
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
		KI0=0.0000006,
		KR0=0.03,
		KD0=0.015,
		c_infected=10000,
		c_death=10000000,
		num_infected0=1000,
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


# nPolicies_subset can be any integer in 1,...,9
# policy_subset can be any subset of list(range(9))
def get_all_policies(
	T,
	pruning=False,
	policy_subset=None,
	nPolicies_subset=None,
	trial_cost_multiplier=1,
	trial_effect_multiplier=1,
	zero_switching_costs=False,
	):
	C_L = 800 * trial_cost_multiplier
	C_M = 1000 * trial_cost_multiplier
	C_MH = 1225 * trial_cost_multiplier
	C_H = 1400 * trial_cost_multiplier
	C_setup_L = 250000 * trial_cost_multiplier
	C_setup_M = 500000 * trial_cost_multiplier
	C_setup_H = 1000000 * trial_cost_multiplier
	C_switch_L = C_setup_L * 2 * trial_cost_multiplier
	C_switch_M = C_setup_M * 2 * trial_cost_multiplier
	C_switch_H = C_setup_H * 2 * trial_cost_multiplier
	if zero_switching_costs:
		C_switch_L = 0
		C_switch_M = 0
		C_switch_H = 0

	P_L = 1 - (1-0.925) * trial_effect_multiplier # 0.9
	P_M = 1 - (1-0.95) * trial_effect_multiplier # 0.925
	P_ML = 1 - (1-0.975) * trial_effect_multiplier # 0.95
	P_MH = 1 - (1-0.99) * trial_effect_multiplier # 0.975
	P_H = 1 - (1-0.995) * trial_effect_multiplier # 0.995

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


def get_saved_parameters(source='output/parameters.pickle'):
	with open(source, 'rb') as f:
		param_kwargs = pickle.load(f)
	return param_kwargs


def save_parameters(
	param_kwargs, dest='output/parameters.txt', vax_param_kwargs=None, pickling=True, latex=False
	):
	if True:
		with open(dest, 'w') as f:
			f.writelines(f"{k} : {v}\n" for k, v in param_kwargs.items() if k != 'policies')
			# for k, v in param_kwargs.items():
			# if k != 'policies':
			# f.write(f"{k} : {v}")
			if vax_param_kwargs is not None:
				f.writelines(f"{k} : {v}\n" for k, v in vax_param_kwargs.items() if k != 'policies')
			f.write(str(param_kwargs['policies']))
	if pickling:
		pickle_dest = "/".join((dest.rsplit('/', 1)[0], 'parameters.pickle'))
		with open(pickle_dest, 'wb') as f:
			pickle.dump(param_kwargs, f)

	if latex:
		latex_dest = "/".join((dest.rsplit('/', 2)[0], 'parameters_table.tex'))
		with (open("latex_param_template.tex", "r")) as f:
			template_str = f.read()
		if vax_param_kwargs is None:
			rendered = jinja2.Template(template_str).render(
				params={k: v for k, v in param_kwargs.items() if k not in ('policies', 'T')},
				policies=param_kwargs['policies'],
				len=len
				)
		else:
			rendered = jinja2.Template(template_str).render(
				params={k: v for k, v in param_kwargs.items() if k not in ('policies', 'T')},
				policies=param_kwargs['policies'],
				len=len,
				vax_params=vax_param_kwargs
				)
		with open(latex_dest, "w") as f:
			f.write(rendered)


# used in "solve_model_compare_lagrangian.py"
def get_row_from_solutions(
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
	):
	cumulative_stats = (
		"cumulative_cost",
		"cumulative_policy_cost",
		"cumulative_disease_cost",
		"cumulative_deaths",
		"cumulative_recovered",
		)
	cumulative_cols = []
	soldict_no_policy = sol_no_policy.to_dict(include_model_stuff=True, cumulative_stats=True)
	general_keys = ("T", "m", "n", "nConstraints", "nPolicies", "nVariables")
	row = {k: v for k, v in soldict_no_policy.items() if k in general_keys}
	row['no_policy_objVal'] = soldict_no_policy['objVal']
	row['trial_cost_multiplier'] = trial_cost_multiplier
	row['trial_effect_multiplier'] = trial_effect_multiplier
	for stat in cumulative_stats:
		row[f'no_policy_{stat}'] = soldict_no_policy[stat]
		cumulative_cols.append(f'no_policy_{stat}')

	# raise
	soldict_early_stopping = sol_early_stopping.to_dict(cumulative_stats=True)
	row['solver_desired_optGap'] = solver_params['optGap']
	row['solver_LB'] = soldict_early_stopping['LB']
	row['solver_objVal'] = soldict_early_stopping['objVal']
	row['solver_optGap'] = soldict_early_stopping['optGap']
	row['solver_timeToSolve'] = soldict_early_stopping['timeToSolve']

	for stat in cumulative_stats:
		row[f'solver_{stat}'] = soldict_early_stopping[stat]
		cumulative_cols.append(f'solver_{stat}')

	soldict_lagrangian = sol_lagrangian.to_dict(cumulative_stats=True)
	row['lagrangian_desired_optGap'] = lagrangian_heuristic.threshold
	row['lagrangian_desired_L1_optGap'] = lagrangian_heuristic.L1_optGap
	row['lagrangian_desired_L2_optGap'] = lagrangian_heuristic.L2_optGap
	row['lagrangian_LB'] = soldict_lagrangian['LB']
	row['lagrangian_LB_guaranteed'] = sol_lagrangian.lb_guaranteed
	row['lagrangian_objVal'] = soldict_lagrangian['objVal']
	row['lagrangian_optGap'] = soldict_lagrangian['optGap']
	row['lagrangian_timeToSolve'] = soldict_lagrangian['timeToSolve']
	row['lagrangian_solution_path'] = sol_lagrangian.description
	# row['solver_vs_lagrangian_lb_gap'] = (
	# 	row['solver_objVal'] - row['lagrangian_objVal']
	# 	) / row['lagrangian_objVal']
	for stat in cumulative_stats:
		row[f'lagrangian_{stat}'] = soldict_lagrangian[stat]
		cumulative_cols.append(f'lagrangian_{stat}')

	row['lagrangian_L1_objVal'] = sol_lagrangian.solution_path[-1]['L1']['objVal']
	row['lagrangian_L2_objVal'] = sol_lagrangian.solution_path[-1]['L2']['objVal']
	row['lagrangian_timeToSolve_L1_total'] = sum(
		sol_lagrangian.solution_path[k]['L1']['timeToSolve']
		for k in range(len(sol_lagrangian.solution_path))
		)
	row['lagrangian_timeToSolve_L2_total'] = sum(
		sol_lagrangian.solution_path[k]['L2']['timeToSolve']
		for k in range(len(sol_lagrangian.solution_path))
		)

	row['quadratic_heuristic_objVal'] = sol_quadratic_heuristic.objVal
	row['quadratic_approx_objVal'] = sol_quadratic_heuristic.quadratic_solution

	row['simple_index_blocksize_1_objVal'] = sol_simple_index_heuristic_blocksize_1.objVal
	row['simple_index_blocksize_7_objVal'] = sol_simple_index_heuristic_blocksize_7.objVal

	return row, cumulative_cols
	# pass
	# sol_no_policy = no_policy_heuristic(solver_params, param_kwargs, model_kwargs)
	# sol_early_stopping = early_stopping_heuristic(solver_params, param_kwargs, model_kwargs)
	# sol_lagrangian = lagrangian_heuristic(solver_params, param_kwargs, model_kwargs)
