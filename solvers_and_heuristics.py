import pandas as pd
import numpy as np
import pyomo.environ as pyo
from time import time
# import pyomo
# import gurobipy as gp

from model import get_SIRD_model
from model_bilinear import get_SIRD_model_bilinear
from model_gurobi_capable import get_SIRD_model_gurobi_capable
# from model_gurobipy import get_model_and_solve_gurobi
from model_no_policy import no_policy_model
from pyomo_utils import add_constraints, solve_pyomo, fix_all_components
from process_results import get_results_df, plot_results, Solution, plot_time_trials
from model_lagrangians import fix_full_model, get_L1_model, get_L2_model
from model_quadratic import get_quadratic_model

from model_vaccinated import get_model_vaccinated


# parameters that will be changed in sensitivity analyses
def get_params(
	T,
	individual_multiplier,
	N,
	KI0,
	KR0,
	KD0,
	policies,
	c_infected,
	c_death,
	days_per_period=1,
	pct_infected=.05,
	num_infected0=None,
	cost_multiplier=1,
	num_recovered0=0,
	num_dead0=0
	):
	m, n = policies.m, policies.n
	P_param = policies.P_param.copy()
	C_policy = individual_multiplier * policies.C_policy.copy()
	C_setup = policies.C_setup.copy()
	C_switch = policies.C_switch.copy()

	C_infected = individual_multiplier * np.full((T), c_infected) # T by 1
	C_death = individual_multiplier * c_death # 1 by 1
	KI = np.full(
		(T), days_per_period * KI0 * individual_multiplier
		) # T by 1 # S->I = k1 # multiply by individual_multiplier because term is quadratic and later quantities are multiplied by individual_multiplier again
	KR = np.full((T), days_per_period * KR0) # T by 1 # I->R = k3 + k6 + k8
	KD = np.full((T), days_per_period * KD0) # T by 1 # I->D = k7
	if num_infected0 is None:
		I0 = N * pct_infected # pct_inf% initially infected. Must be >0.
	else:
		I0 = num_infected0
	D0 = num_dead0
	R0 = num_recovered0
	S0 = N - I0 - D0 - R0
	I0 = I0 / individual_multiplier
	S0 = S0 / individual_multiplier
	D0 = D0 / individual_multiplier
	R0 = R0 / individual_multiplier
	return m, n, T, N, KI, KR, KD, I0, S0, D0, R0, P_param, C_policy, C_setup, C_switch, C_infected, C_death, individual_multiplier, days_per_period, cost_multiplier, policies


def get_vaccination_params(
	T_vax=None,
	S0_antivax_factor=0.2,
	KV=.05,
	**model_params,
	):
	m, n, T, N, KI, KR, KD, I0, S0, D0, R0, P_param, C_policy, C_setup, C_switch, C_infected, C_death, individual_multiplier, days_per_period, cost_multiplier, policies = get_params(
		**model_params
		)
	if T_vax is None:
		T_vax = T // 2
	# if S0_antivax_factor is None:
	# S0_antivax_factor = 0.2
	# T_vax =
	KI_vax = KI * 0.2
	KR_vax = KR * 1.5
	KD_vax = KD / 15
	KI_novax = KI * 1.1 # because population product smaller naturally due to partition of population, the same KI will result in fewer infections. Make it slightly larger to compensate
	KR_novax = KR
	KD_novax = KD
	S0_antivax = S0_antivax_factor * S0
	return T_vax, KI_vax, KR_vax, KD_vax, KI_novax, KR_novax, KD_novax, KV, S0_antivax


def get_model(*args, **kwargs):
	use_gurobipy = kwargs.pop('use_gurobipy', False)
	bilinear = kwargs.pop("bilinear", False)
	if use_gurobipy:
		return get_SIRD_model_gurobi_capable(*args, **kwargs)
	if bilinear:
		return get_SIRD_model_bilinear(*args, **kwargs)
	else:
		return get_SIRD_model(*args, **kwargs)


# fixvals is a dict of variable names and values that all components of that variable outside unfixrange should be set to
# a value of None indicates the variable should be fixed to its current value.
def fix_and_unfix_range(model, fixvals, unfixrange):
	T0 = min(unfixrange)
	T_horizon = max(unfixrange)
	for varname, fixvalue in fixvals.items():
		var = getattr(model, varname)
		dim = var.dim()
		for k, component in var.items():
			try:
				t = k[-1] # variable index k is tuple, e.g. y[(i,j,t)]
			except:
				t = k # variable index k is scalar, e.g. P[t]
			if T0 <= t < T_horizon:
				component.unfix()
			elif t < T0: #fix past values
				component.fix(component.value)
			else: #fix future values
				if fixvalue is not None:
					component.fix(fixvalue)
				else:
					component.fix(component.value)


# Note: heuristics do not work with gurobipy formulation
def get_heuristic_solver(
	heuristic_f,
	tee=True,
	do_local_search=False,
	local_search_kwargs={},
	**heuristic_kwargs,
	):
	unwanted_kwargs = (
		"use_gurobipy", "only_policy", "get_no_policy_solution", 'tee', 'only_no_policy'
		)
	fname = "_".join([heuristic_f.__name__] + [
		f"{k}_{v}" for k, v in heuristic_kwargs.items() if k not in unwanted_kwargs
		])
	if do_local_search:
		fname += "_".join([f"_local_search"] + [
			f"{k}_{v}" for k, v in local_search_kwargs.items() if k not in unwanted_kwargs
			])
	# for unwanted_kwarg in ("use_gurobipy", "only_policy", "get_no_policy_solution",'tee'):
	# 	for boolval in (True, False):
	# 		fname = fname.replace(f"_{unwanted_kwarg}_{boolval}", "")

	def _f(*args, **kwargs):
		print(
			f"Executing heuristic: {heuristic_f.__name__} with arguments:" +
			', '.join([f'{k} : {v}' for k, v in heuristic_kwargs.items()]) + f"\nLabel: '{fname}'",
			)
		start = time()
		output = heuristic_f(*args, **kwargs, **heuristic_kwargs, tee=tee)
		if do_local_search:
			model = output.model
			output = solve_and_process_local_search(
				*args, **kwargs, **local_search_kwargs, model=model
				)
		timeToSolve = time() - start
		if hasattr(output, 'timeToSolve'):
			output.timeToSolve = timeToSolve
		else:
			print(f"Cannot return timeToSolve: {timeToSolve:,.0} seconds")
		return output

	_f.__name__ = fname
	for k, v in heuristic_kwargs.items():
		setattr(_f, k, v)
	return _f


def solve_and_process_vaccination(
	solver_params,
	param_kwargs,
	model_kwargs,
	get_no_policy_solution=True,
	use_gurobipy=False,
	only_no_policy=False,
	only_policy=False,
	# vaccination_params=None,
	S0_antivax_factor=None,
	T_vax=None,
	KV=None,
	optGap=None,
	tee=True
	):
	# if vaccination_params is None:
	# 	raise

	old_optGap = solver_params['optGap']
	if optGap is not None:
		solver_params['optGap'] = optGap

	sols = {}
	solver = solver_params['solver']
	solve_from_binary = solver_params['solve_from_binary']
	bilinear = solver_params['bilinear']
	# bilinear = True if solver.lower() == "gurobi" else False
	if get_no_policy_solution:
		if only_no_policy:
			forcing_no_policy = (True,)
		else:
			forcing_no_policy = (True, False)
	else:
		forcing_no_policy = (False,)
	if only_policy:
		forcing_no_policy = (False,)
	for force_no_policy in forcing_no_policy:
		print(f"\n\nGETTING MODEL with {'no policy' if force_no_policy else 'policies active'}")
		print(solver, solve_from_binary)

		model = get_model_vaccinated(
			*get_params(**param_kwargs),
			*get_vaccination_params(
				T_vax=T_vax,
				S0_antivax_factor=S0_antivax_factor,
				**param_kwargs,
				),
			**model_kwargs,
			bilinear=bilinear,
			force_no_policy=force_no_policy,
			)
		print("GOT MODEL\n\n")
		sols[force_no_policy] = Solution(
			*solve_pyomo(model, **{
				**solver_params,
				**dict(tee=(not force_no_policy) and tee)
				}),
			vaccination=True,
			)
	if only_policy:
		return sols[False]
	if only_no_policy:
		return sols[True]
	solver_params['optGap'] = old_optGap
	return sols


def solve_and_process(
	solver_params,
	param_kwargs,
	model_kwargs,
	get_no_policy_solution=True,
	use_gurobipy=False,
	only_no_policy=False,
	only_policy=False,
	optGap=None,
	tee=True
	):
	old_optGap = solver_params['optGap']
	if optGap is not None:
		solver_params['optGap'] = optGap

	sols = {}
	solver = solver_params['solver']
	solve_from_binary = solver_params['solve_from_binary']
	bilinear = solver_params['bilinear']
	# bilinear = True if solver.lower() == "gurobi" else False
	if get_no_policy_solution:
		if only_no_policy:
			forcing_no_policy = (True,)
		else:
			forcing_no_policy = (True, False)
	else:
		forcing_no_policy = (False,)
	if only_policy:
		forcing_no_policy = (False,)
	for force_no_policy in forcing_no_policy:
		print(f"\n\nGETTING MODEL with {'no policy' if force_no_policy else 'policies active'}")
		print(solver, solve_from_binary)
		if use_gurobipy and not force_no_policy:
			gurobi_solution = get_model_and_solve_gurobi(
				*get_params(**param_kwargs),
				**model_kwargs,
				force_no_policy=force_no_policy,
				**solver_params
				)
			sols[force_no_policy] = Solution(*gurobi_solution) #Solution(*gurobi_solution)
		else:
			model = get_model(
				*get_params(**param_kwargs),
				**model_kwargs,
				bilinear=bilinear,
				force_no_policy=force_no_policy,
				use_gurobipy=use_gurobipy
				)
			print("GOT MODEL\n\n")
			sols[force_no_policy] = Solution(
				*solve_pyomo(model, **{
					**solver_params,
					**dict(tee=(not force_no_policy) and tee)
					})
				)
	if only_policy:
		return sols[False]
	if only_no_policy:
		return sols[True]
	solver_params['optGap'] = old_optGap
	return sols


def solve_and_process_early_stopping(
	solver_params,
	param_kwargs,
	model_kwargs,
	optGap=0.85,
	tee=True,
	):
	return solve_and_process(
		solver_params,
		param_kwargs,
		model_kwargs,
		get_no_policy_solution=False,
		use_gurobipy=False,
		only_no_policy=False,
		only_policy=True,
		optGap=optGap,
		tee=tee
		)


def solve_and_process_no_policy(
	solver_params,
	param_kwargs,
	model_kwargs,
	tee=True,
	):
	return solve_and_process(
		solver_params,
		param_kwargs,
		model_kwargs,
		get_no_policy_solution=True,
		use_gurobipy=False,
		only_no_policy=True,
		optGap=0.01,
		tee=tee
		)


def solve_and_process_lagrangian(
	solver_params,
	param_kwargs,
	model_kwargs,
	initial_stepsize=10,
	use_smart_stepsize=True,
	smart_stepsize=2,
	min_iterations=0,
	max_iterations=15,
	threshold=0.075,
	optGap=0.01,
	tee=True, #UNUSED - respects 'tee' in solver_params
	L1_optGap=0.025,
	L2_optGap=0.075,
	L2_max_time=500,
	):
	stepsize = initial_stepsize
	model_full = get_model(*get_params(**param_kwargs), **model_kwargs)
	llambda = {t: 0 for t in model_full.t}
	prev_ub = np.inf
	prev_lb = -np.inf
	best_ub = np.inf
	best_lb = -np.inf
	n_iterations = 0
	solution_path = []
	L1_sol_times = []
	L2_sol_times = []
	while True:
		n_iterations += 1
		print(f"ITERATION {n_iterations} OF LAGRANGIAN")
		model_L1 = get_L1_model(model_full, llambda)
		model_L2 = get_L2_model(model_full, llambda)
		# breakpoint()
		print("SOLVING L1")
		sol_L1 = Solution(
			*solve_pyomo(
				model_L1,
				**{
					**solver_params,
					**dict(tee=False)
					},
				optGap_override=L1_optGap,
				),
			get_DF=False
			)
		print("SOLVING L2")
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
		fix_all_components(model_full.y, model_L1.y)
		print("SOLVING BOOKKEEPING MODEL")
		sol_full = Solution(
			*solve_pyomo(
				model_full,
				**{
					**solver_params,
					**dict(tee=False)
					},
				optGap_override=.001,
				)
			)
		lb = sol_L1.objVal + sol_L2.objVal
		ub = sol_full.objVal
		solution_path.append(
			dict(
				lb=int(lb),
				ub=int(ub),
				timeToSolve_L1=int(sol_L1.timeToSolve),
				timeToSolve_L2=int(sol_L2.timeToSolve),
				timeToSolve_bookkeeping=int(sol_full.timeToSolve),
				L1=sol_L1.to_dict(),
				L2=sol_L2.to_dict(),
				)
			)
		if lb > best_lb:
			best_lb = lb
		if ub < best_ub:
			best_ub = ub
		# consider stopping
		if n_iterations > min_iterations:
			if abs(ub - best_lb) / abs(ub) < threshold:
				break
			if n_iterations > max_iterations:
				break
		# update multipliers
		if use_smart_stepsize:
			smart_denominator = sum((
				pyo.value(model_L1.multiplier_coefficients[t]) +
				pyo.value(model_L2.multiplier_coefficients[t])
				)**2 for t in model_full.t)
			if not (lb > prev_lb): #use a one-iteration rule
				smart_stepsize /= 2
			stepsize = smart_stepsize * (best_ub-lb) / smart_denominator
		else:
			stepsize = initial_stepsize / ((n_iterations + 1)**(3 / 2))
			# stepsize = stepsize * (1/2)
		for t in model_full.t:
			llambda[t] = max(
				llambda[t] + stepsize * (
					pyo.value(model_L1.multiplier_coefficients[t]) +
					pyo.value(model_L2.multiplier_coefficients[t])
					), 0
				)

		prev_lb = lb
		prev_ub = ub
		print(f"LB: {lb}, UB: {ub}, %gap: {100*(ub-lb)/ub}%")
		# print(llambda)

	sol_full.lb = best_lb
	sol_full.ub = ub
	# sol_full.optGap = optGap
	sol_full.optGap = (ub-lb) / ub

	sol_full.lb_guaranteed = sol_L1.lb + sol_L2.lb

	sol_full.multipliers = llambda
	sol_full.solution_path = solution_path
	# sol_full.threshold = threshold
	# L1_sol_times_str = ", ".join(f"{tts:d}" for tts in L1_sol_times)
	# L2_sol_times_str = ", ".join(f"{tts:d}" for tts in L2_sol_times)
	sol_full.description = " -> ".join(
		f"(lb: {entry['lb']:,}, ub: {entry['ub']:,}, timeToSolve: [{entry['timeToSolve_L1']:d}, {entry['timeToSolve_L2']:d}, {entry['timeToSolve_bookkeeping']:d}])"
		for entry in solution_path
		)
	# f"L1 Solution Times: {L1_sol_times_str}; L2 Solution Times: {L2_sol_times_str}"
	return sol_full


# sol_L1 = Solution(*solve_pyomo(model_L1, **{**solver_params, **dict(tee=False)}), get_DF=False)
# sol_L2 = Solution(*solve_pyomo(model_L2, **{**solver_params, **dict(tee=False)}), get_DF=False)


def solve_and_process_index(
	solver_params,
	param_kwargs,
	model_kwargs,
	max_number_of_policies=1,
	optGap=None,
	tee=False,
	separate_assortments=False
	):
	assert not ((max_number_of_policies > 1) and separate_assortments)
	old_optGap = solver_params['optGap']
	if optGap is not None:
		solver_params['optGap'] = optGap
	policies = param_kwargs['policies']
	solver = solver_params['solver']
	bilinear = solver_params['bilinear']
	if separate_assortments and not policies.is_pruned:
		pruned_policies = policies.pruned()
		model_kwargs['allow_multiple_policies_per_period'] = False
	else:
		pruned_policies = policies

	model = get_model(
		*get_params(**param_kwargs), **model_kwargs, bilinear=bilinear, force_no_policy=False
		)
	bestObj, bestSol = np.inf, None
	selected_policies = {} # {i : j for every selected (i,j)}
	for iteration_ind in range(max_number_of_policies):
		best_new_policy = None
		for i, policy in enumerate(pruned_policies):
			if i in selected_policies: continue
			for j in range(policy.n):
				considered_policies = {i: j, **selected_policies}
				for (ii, jj, tt), component in model.y.items():
					if ii in considered_policies:
						if ii in selected_policies:
							component.fix(component.value)
						else:
							if considered_policies[ii] == jj:
								component.unfix()
							else:
								component.fix(0)
					else:
						component.fix(0)
				print(
					f"\nConsidering Policy {i+1}/{len(pruned_policies)} at level {j+1} (along with {selected_policies}): {policy}\n"
					)

				sol = Solution(*solve_pyomo(model, **{**solver_params, **dict(tee=tee)}))
				if sol.objVal < bestObj:
					bestObj = sol.objVal
					bestSol = sol
					best_new_policy = (i, j)
		if best_new_policy is not None:
			selected_policies[best_new_policy[0]] = best_new_policy[1]
		else:
			print(f"NO IMPROVEMENT AT ITERATION {iteration_ind}")
			break
	solver_params['optGap'] = old_optGap
	return bestSol


# Note: does not work with gurobipy formulation
# if truncate_costs is True: objective is truncated to the lookahead horizon, i.e. "only consider next w periods"
# if truncate_costs is False or None (or 0): objective is not truncated, i.e. "hands are tied after next w periods"
# if truncate_costs is numeric, then the objective is truncated to the next w+truncate_costs periods
# More truncation, like a shorter lookahead horizon, makes the problem easier. Unlike a longer lookahead horizon,
# increasing the truncation horizon does not increase the number of integer variables
def solve_and_process_lookahead(
	solver_params, param_kwargs, model_kwargs, w=1, truncate_costs=True, optGap=None, tee=False
	):
	print(f"Using lookahead heuristic with {w=}, and {truncate_costs=}")
	old_optGap = solver_params['optGap']
	if optGap is not None:
		solver_params['optGap'] = optGap
	solver = solver_params['solver']
	# solve_from_binary = solver_params['solve_from_binary']
	bilinear = solver_params['bilinear']
	T = param_kwargs['T']
	params = get_params(**param_kwargs)
	model = get_model(*params, **model_kwargs, bilinear=bilinear, force_no_policy=False)
	print(f"GOT MODEL\n\n")
	for T_horizon in range(w, T + 1):
		T0 = T_horizon - w
		print(f"\n\nSolving lookahead for periods ({T0},{T_horizon}) out of {T}\n\n")
		fix_and_unfix_range(model, dict(P=1, y=0), (T0, T_horizon))
		if truncate_costs:
			if type(truncate_costs) in (int, float):
				objective_horizon = min(T_horizon + int(truncate_costs), T)
			else:
				objective_horizon = T_horizon
			model.obj.deactivate()
			del model.obj
			model.obj = pyo.Objective(
				expr=model.get_objective_cumulative_TT_periods(model, objective_horizon),
				sense=pyo.minimize
				)
		solve_pyomo(model, **{**solver_params, **dict(tee=tee)})
	model.obj = pyo.Objective(
		expr=model.get_objective_cumulative_TT_periods(model, T), sense=pyo.minimize
		) #original, full objective
	sol = Solution(*solve_pyomo(model, **solver_params))
	solver_params['optGap'] = old_optGap
	return sol


def solve_and_process_local_search(
	solver_params,
	param_kwargs,
	model_kwargs,
	w=1,
	iterations=1,
	iterate_until_no_improvement=False,
	tee=False,
	optGap=None,
	model=None
	):
	if iterate_until_no_improvement:
		iterations = 999999999
	T = param_kwargs['T']
	old_optGap = solver_params['optGap']
	if optGap is not None:
		solver_params['optGap'] = optGap
	if model is None:
		solver = solver_params['solver']
		# solve_from_binary = solver_params['solve_from_binary']
		bilinear = solver_params['bilinear']
		params = get_params(**param_kwargs)
		model = get_model(*params, **model_kwargs, bilinear=bilinear, force_no_policy=False)
	objVal = model.obj.expr()
	old_objVal = objVal
	nChanges = 0
	for iteration in range(iterations):
		nChanges_per_iteration = 0
		for T_horizon in range(w, T + 1):
			T0 = T_horizon - w
			print(f"\n\nDoing local search for periods ({T0},{T_horizon}) out of {T}\n\n")
			fix_and_unfix_range(model, dict(P=None, y=None), (T0, T_horizon))
			solve_pyomo(model, **{**solver_params, **dict(tee=tee)})
			if (new_objVal := model.obj.expr()) < objVal:
				objVal = new_objVal
				nChanges_per_iteration += 1
				print(f"IMPROVEMENT for range ({T0},{T_horizon})")
		print(
			f"Local Search made {nChanges_per_iteration} improvements in iteration {iteration+1}/{iterations}, yielding {old_objVal - objVal:.1} in total so far!"
			)
		nChanges += nChanges_per_iteration
		if nChanges_per_iteration == 0:
			print(f"Terminating local search on iteration {iteration+1} due to no improvement")
			break
	sol = Solution(*solve_pyomo(model, **solver_params))
	print(
		f"Local Search made {nChanges} improvements, amounting to {old_objVal - objVal:.1} in total!"
		)
	solver_params['optGap'] = old_optGap
	return sol


def solve_and_process_quadratic(
	solver_params,
	param_kwargs,
	model_kwargs,
	epsilon=0.01,
	optGap=0.075,
	tee=True, #UNUSED - respects 'tee' in solver_params
	):
	model_full = get_model(*get_params(**param_kwargs), **model_kwargs)
	model_quadratic = get_quadratic_model(model_full, epsilon=epsilon)

	sol_quadratic = Solution(
		*solve_pyomo(
			model_quadratic,
			**{
				**solver_params,
				**dict(tee=False)
				},
			optGap_override=optGap,
			),
		get_DF=False
		)
	best_policy_assortments = model_quadratic.find_closest_policy_assortments()

	for k, component in model_full.y.items():
		component.fix(0)

	for t in model_full.t:
		policy_assortment = best_policy_assortments[t]
		for ii, jj in enumerate(policy_assortment):
			if jj > -1:
				model_full.y[ii, jj, t].fix(1)

	sol_full = Solution(
		*solve_pyomo(
			model_full,
			**{
				**solver_params,
				**dict(tee=False)
				},
			optGap_override=.001,
			)
		)
	# sol_full.quadratic_lb = sol_quadratic.lb
	sol_full.quadratic_solution = sol_quadratic.objVal
	return sol_full


def solve_and_process_simple_index(
	solver_params,
	param_kwargs,
	model_kwargs,
	tee=True, #UNUSED - respects 'tee' in solver_params
	block_size=7,
	myopic=True,
	):
	model = get_model(*get_params(**param_kwargs), **model_kwargs)
	T = model.T
	blocks = [[tt for tt in range(t, t + block_size) if tt < T] for t in range(0, T, block_size)]

	def get_index(i, j):
		return model.policies[i].P_[j] * model.policies[i].C_[j]

	def get_obj():
		sol_tuple = solve_pyomo(
			model,
			**{
				**solver_params,
				**dict(tee=False)
				},
			optGap_override=.001,
			)
		objVal = sol_tuple[0]
		return objVal

	def fix_block(block, used_policies):
		for (i, j, t), component in model.y.items():
			if t in block:
				if (i, j) in used_policies:
					component.fix(1)
				else:
					component.fix(0)

	index_order_with_duplicate_i = sorted([(i, j)
											for i, policy in enumerate(model.policies)
											for j in range(policy.n)],
											key=lambda tup: get_index(*tup))

	index_order = []
	seen_i = []
	for (i, j) in index_order_with_duplicate_i:
		if i not in seen_i:
			index_order.append((i, j))
			seen_i.append(i)

	for k, component in model.y.items():
		component.fix(0)

	for block in blocks:
		if myopic:
			model.obj.deactivate()
			del model.obj
			model.obj = pyo.Objective(
				expr=model.get_objective_cumulative_TT_periods(model, max(block)), sense=pyo.minimize
				)

		used_policies = []
		unused_policies = index_order.copy()
		while True:
			print(f"Processing block {block} with policies {used_policies}")
			fix_block(block, used_policies)
			old_obj = get_obj()
			if len(unused_policies) > 0:
				new_policy = unused_policies.pop(0)
				fix_block(block, used_policies + [new_policy])
				new_obj = get_obj()
				if old_obj < new_obj:
					break
				else:
					used_policies.append(new_policy)
			else:
				break

	sol = Solution(
		*solve_pyomo(
			model,
			**{
				**solver_params,
				**dict(tee=False)
				},
			optGap_override=.001,
			),
		get_DF=True
		)
	return sol


if __name__ == "__main__":
	x = 5
	print(x)
