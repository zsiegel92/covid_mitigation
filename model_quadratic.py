import numpy as np
from math import prod
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo, get_constraint_adder, fix_all_components


def get_cost_fit_coefs(model_full, deg=2):
	pruned = model_full.policies.pruned()
	df = model_full.policies.assortment_df()
	# df.loc[policies.keeper_assortments, 'efficient'] = True
	df_efficient = df.loc[df['efficient']]

	predicted = ['C_policy', 'C_setup', 'C_switch']

	fitted_coefs = np.polyfit(
		df_efficient['P'].values,
		df_efficient[predicted].values,
		deg=deg,
		)
	fitted_coefs = pd.DataFrame(fitted_coefs, columns=predicted)
	return pruned, df_efficient, fitted_coefs


def get_closest_policy_assortment_finder(pruned, policies, model_quadratic, epsilon=0.01):
	def get_closest_policy_assortment_composition(P_t, A_t, C_t, epsilon=0.01):
		errors = [(assortment.P_[0] - P_t) ^ 2 for assortment in pruned]
		min_error = min(errors)
		match = pruned[next(i for i in range(len(errors)) if errors[i] == min_error)]
		return match.assortment_composition

	def finder():
		P = model_quadratic.P
		A = model_quadratic.A
		C = model_quadratic.C
		return [
			get_closest_policy_assortment_composition(
				P[t].value, A[t].value, C[t].value, epsilon=epsilon
				) for t in model_quadratic.t
			]

	return finder


def get_fitted_cost_function(fitted_coefs, col):
	deg = fitted_coefs.shape[0] - 1
	# print(f"Degree is calculated as {deg}")
	get_coef = lambda col, power: fitted_coefs.loc[deg - power, col]
	return lambda P: sum(get_coef(col, i) * P**i for i in range(deg + 1))


def get_quadratic_model(model_full, epsilon=0.01):

	model = pyo.ConcreteModel()
	constrain = get_constraint_adder(model)
	model.i = pyo.RangeSet(0, model_full.m - 1)
	model.j = pyo.RangeSet(0, model_full.n - 1)
	model.t = pyo.RangeSet(0, model_full.T - 1)

	P_lower_bound = np.prod(np.min(np.min(model_full.P_param, axis=1), axis=1))
	S0, I0 = model_full.S0, model_full.I0

	model.P_lower_bound = P_lower_bound
	# np.prod([min(model_full.P_param[t, :, :].flatten()) for t in range(model_full.P_param.shape[0])])

	model.S = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=S0, bounds=(0, S0 + I0))
	model.I = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0))
	model.R = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0))
	model.D = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0))
	model.d = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0))
	model.P = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=1, bounds=(P_lower_bound, 1))
	model.A = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0)
	model.C = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0)

	S = model.S
	I = model.I
	R = model.R
	D = model.D
	d = model.d
	P = model.P
	A = model.A
	C = model.C

	pruned, df_efficient, fitted_coefs = get_cost_fit_coefs(model_full, deg=2)
	fitted_setup_cost = get_fitted_cost_function(fitted_coefs, 'C_setup')
	fitted_variable_cost = get_fitted_cost_function(fitted_coefs, 'C_policy')

	constrain([A[t] == fitted_setup_cost(P[t]) for t in model_full.t])
	constrain([C[t] == fitted_variable_cost(P[t]) for t in model_full.t])

	model.pruned = pruned
	model.df_efficient = df_efficient
	model.fitted_coefs = fitted_coefs
	model.fitted_setup_cost = fitted_setup_cost
	model.fitted_variable_cost = fitted_variable_cost

	model.find_closest_policy_assortments = get_closest_policy_assortment_finder(
		model.pruned,
		model_full.policies,
		model,
		epsilon=epsilon,
		)

	t0 = model.t.first() # do not have to start indices at 0.
	t_except_first = list(t for t in model.t if t != t0) #periods except first period

	S[t0].fix(S0)
	I[t0].fix(I0)
	R[t0].fix(model_full.R0)
	D[t0].fix(model_full.D0)
	d[t0].fix(0)

	constrain([
		S[t] == S[t - 1] - model_full.KI[t] * P[t] * S[t - 1] * I[t - 1] for t in t_except_first
		]) # (1)
	constrain([
		I[t] == I[t - 1] + model_full.KI[t] * P[t] * S[t - 1] * I[t - 1] -
		model_full.KR[t] * I[t - 1] - model_full.KD[t] * I[t - 1] for t in t_except_first
		]) # (2)
	constrain([R[t] == R[t - 1] + model_full.KR[t] * I[t - 1] for t in t_except_first]) # (3)
	constrain([d[t] == model_full.KD[t] * I[t - 1] for t in t_except_first]) # (4)
	constrain([D[t] == D[t - 1] + d[t] for t in t_except_first]) # (5)

	setup_cost = sum(A[t] for t in model_full.t)
	variable_cost = sum(C[t] * model.S[t] / model_full.individual_multiplier for t in model_full.t)

	policy_cost = (1 / model_full.cost_multiplier) * (setup_cost+variable_cost)

	disease_cost = (1 / model_full.cost_multiplier) * sum(
		model_full.C_infected[t] * I[t] + model_full.C_death * d[t] for t in model_full.t
		)

	model.obj = pyo.Objective(expr=policy_cost + disease_cost, sense=pyo.minimize)

	return model
