import numpy as np
from math import prod
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo, get_constraint_adder, fix_all_components


def fix_full_model(model, L1, L2=None):
	fix_all_components(model.y, L1.y)


def get_L1_model(model_full, llambda):
	model = pyo.ConcreteModel()
	constrain = get_constraint_adder(model)
	model.i = pyo.RangeSet(0, model_full.m - 1)
	model.j = pyo.RangeSet(0, model_full.n - 1)
	model.t = pyo.RangeSet(0, model_full.T - 1)
	model.y = pyo.Var(model.i * model.j * model.t, domain=pyo.Binary, initialize=0)
	model.z = pyo.Var(
		model.i * model.j * model.t, domain=pyo.NonNegativeReals, bounds=(0, 1), initialize=0
		)
	y = model.y
	z = model.z
	t0 = model.t.first() # do not have to start indices at 0.
	t_except_first = list(t for t in model.t if t != t0) #periods except first period

	constrain([
		sum(y[i, j, t] for j in model_full.j) <= 1 for i in model_full.i for t in model_full.t
		]) # (7)
	constrain([
		z[i, j, t] >= y[i, j, t] - y[i, j, t - 1] for i in model.i for j in model.j
		for t in t_except_first
		]) # (9)
	constrain([z[i, j, 0] >= y[i, j, 0] for i in model.i for j in model.j]) # (9)

	# easiest way to make some policies "not exist" is to set them to zero if costs are infinite or utility is zero (P==1)
	for i in model_full.i:
		for j in model_full.j:
			for t in model_full.t:
				if (model_full.C_setup[i, j, t] == np.inf) or (model_full.C_policy[
					i, j, t] == np.inf) or (model_full.P_param[i, j, t] == 1):
					model.y[i, j, t].fix(0)
	setup_cost = sum(
		model_full.C_setup[i, j, t] * model.y[i, j, t] for i in model_full.i
		for j in model_full.j for t in model_full.t if (model_full.C_setup[i, j, t] != np.inf)
		)
	switching_cost = sum(
		model_full.C_switch[i, j, t] * model.z[i, j, t] for i in model_full.i
		for j in model_full.j for t in model_full.t if (model_full.C_switch[i, j, t] != np.inf)
		)
	variable_cost = sum(
		model_full.C_policy[i, j, t] * (model_full.N / model_full.individual_multiplier) *
		model.y[i, j, t] for i in model_full.i
		for j in model_full.j for t in model_full.t if (model_full.C_policy[i, j, t] != np.inf)
		)
	policy_cost = (1 / model_full.cost_multiplier) * (setup_cost+switching_cost+variable_cost)

	disease_cost = 0
	model.multiplier_coefficients = {
		t: sum(
			pyo.log(1 + (model_full.P_param[i, j, t] - 1) * model.y[i, j, t])
			for i in model_full.i
			for j in model_full.j
			if ((model_full.C_setup[i, j, t] != np.inf) and (model_full.P_param[i, j, t] != 1))
			) for t in model_full.t
		}
	multiplier_cost = sum(llambda[t] * model.multiplier_coefficients[t] for t in model_full.t)

	model.obj = pyo.Objective(expr=policy_cost + disease_cost + multiplier_cost, sense=pyo.minimize)

	# if force_no_policy:
	# 	add_constraints(constr,[y[i,j,t] == 0 for i in model_full.i for j in model_full.j for t in model_full.t])
	return model


def get_L2_model(model_full, llambda):
	model = pyo.ConcreteModel()
	constrain = get_constraint_adder(model)
	S0, I0 = model_full.S0, model_full.I0
	# lower bound on possible value of P_t if all policies are used at maximum level
	# using 0 as a (inclusive) lower bound is numerically unstable because logarithm undefined at 0
	P_lower_bound = np.prod(np.min(np.min(model_full.P_param, axis=1), axis=1))
	model.P_lower_bound = P_lower_bound
	# np.prod([min(model_full.P_param[t, :, :].flatten()) for t in range(model_full.P_param.shape[0])])
	model.i = pyo.RangeSet(0, model_full.m - 1)
	model.j = pyo.RangeSet(0, model_full.n - 1)
	model.t = pyo.RangeSet(0, model_full.T - 1)
	model.S = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=S0, bounds=(0, S0 + I0))
	model.I = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0))
	model.R = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0))
	model.D = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0))
	model.d = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0))
	model.P = pyo.Var(model.t, domain=pyo.NonNegativeReals, initialize=1, bounds=(P_lower_bound, 1))
	S = model.S
	I = model.I
	R = model.R
	D = model.D
	d = model.d
	P = model.P

	t0 = model_full.t.first() # do not have to start indices at 0.
	t_except_first = list(t for t in model_full.t if t != t0) #periods except first period

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
	model.multiplier_coefficients = {t: (-1) * pyo.log(model.P[t]) for t in model_full.t}

	policy_cost = 0
	disease_cost = (1 / model_full.cost_multiplier) * sum(
		model_full.C_infected[t] * I[t] + model_full.C_death * d[t] for t in model_full.t
		)
	multiplier_cost = sum(llambda[t] * model.multiplier_coefficients[t] for t in model_full.t)
	model.obj = pyo.Objective(expr=policy_cost + disease_cost + multiplier_cost, sense=pyo.minimize)
	# if force_no_policy:
	# 	add_constraints(constr,[y[i,j,t] == 0 for i in model.i for j in model.j for t in model.t])
	return model
