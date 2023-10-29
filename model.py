import numpy as np
from math import prod
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo, get_constraint_adder


def get_cumulative_disease_cost(model, TT):
	return (1 / model.cost_multiplier) * sum(
		model.C_infected[t] * model.I[t] + model.C_death * model.d[t]
		for t in range(TT)
		)


def get_cumulative_policy_cost(model, TT):
	setup_cost = 0
	variable_cost = 0
	setup_cost = sum(
		model.C_setup[i, j, t] * model.y[i, j, t] for i in model.i
		for j in model.j
		for t in range(TT)
		if (model.C_setup[i, j, t] != np.inf)
		)
	switching_cost = sum(
		model.C_switch[i, j, t] * model.z[i, j, t] for i in model.i
		for j in model.j
		for t in range(TT)
		if (model.C_switch[i, j, t] != np.inf)
		)
	if model.cost_per_susceptible_only:
		variable_cost = sum(
			model.C_policy[i, j, t] * model.S[t] * model.y[i, j, t]
			for i in model.i for j in model.j
			for t in range(TT)
			if (model.C_policy[i, j, t] != np.inf)
			)
	else:
		variable_cost = sum(
			model.C_policy[i, j, t] * (model.N / model.individual_multiplier) *
			model.y[i, j, t] for i in model.i for j in model.j
			for t in range(TT)
			if (model.C_policy[i, j, t] != np.inf)
			)
	return (1 /
			model.cost_multiplier) * (setup_cost+switching_cost+variable_cost)


def get_objective_cumulative_TT_periods(model, TT):
	disease_cost = get_cumulative_disease_cost(model, TT)
	if model.force_no_policy:
		return disease_cost
	else:
		policy_cost = get_cumulative_policy_cost(model, TT)
		return disease_cost + policy_cost


def get_SIRD_model(
	m,
	n,
	T,
	N,
	KI,
	KR,
	KD,
	I0,
	S0,
	D0,
	R0,
	P_param,
	C_policy,
	C_setup,
	C_switch,
	C_infected,
	C_death,
	individual_multiplier,
	days_per_period,
	cost_multiplier,
	policies,
	allow_multiple_policies_per_period=False,
	cost_per_susceptible_only=False,
	use_logarithm=False,
	force_no_policy=False,
	time_varying_costs_and_probabilities=False,
	efficient=False
	):
	varnames = locals().copy() #stores *args and **kwargs
	model = pyo.ConcreteModel()
	constrain = get_constraint_adder(model)
	# stores all arguments of this function as attributes of model for access later on
	for k, v in varnames.items():
		setattr(model, k, v)

	model.bilinear = False

	model.i = pyo.RangeSet(0, m - 1)
	model.j = pyo.RangeSet(0, n - 1)
	model.t = pyo.RangeSet(0, T - 1)
	t0 = model.t.first() # do not have to start indices at 0.
	t_except_first = list(
		t for t in model.t if t != t0
		) #periods except first period

	model.S = pyo.Var(
		model.t, domain=pyo.NonNegativeReals, initialize=S0,
		bounds=(0, S0 + I0)
		)
	model.I = pyo.Var(
		model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0)
		)
	model.R = pyo.Var(
		model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0)
		)
	model.D = pyo.Var(
		model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0)
		)
	model.d = pyo.Var(
		model.t, domain=pyo.NonNegativeReals, initialize=0, bounds=(0, S0 + I0)
		)

	S = model.S
	I = model.I
	R = model.R
	D = model.D
	d = model.d

	S[t0].fix(S0)
	I[t0].fix(I0)
	R[t0].fix(R0)
	D[t0].fix(D0)
	d[t0].fix(0)

	constrain([R[t] == R[t - 1] + KR[t] * I[t - 1]
				for t in t_except_first]) # (3)
	constrain([d[t] == KD[t] * I[t - 1] for t in t_except_first]) # (4)
	constrain([D[t] == D[t - 1] + d[t] for t in t_except_first]) # (5)

	if (not force_no_policy) or (not efficient):
		print(f"Creating {m}*{n}*{T} = {m*n*T} binary variables")
		model.y = pyo.Var(
			model.i * model.j * model.t, domain=pyo.Binary, initialize=0
			)
		model.z = pyo.Var(
			model.i * model.j * model.t,
			domain=pyo.NonNegativeReals,
			bounds=(0, 1),
			initialize=0
			)
		model.P = pyo.Var(
			model.t, domain=pyo.NonNegativeReals, initialize=1, bounds=(0, 1)
			)
		P = model.P
		y = model.y
		z = model.z
		if use_logarithm:
			constrain([
				pyo.log10(P[t]) == sum(
					pyo.log10(P_param[i, j, t] * y[i, j, t] + (1 - y[i, j, t]))
					for i in model.i
					for j in model.j
					if (P_param[i, j, t] != 1)
					)
				for t in model.t
				]) # (6)
		else:
			constrain([
				P[t] == prod(
					P_param[i, j, t] * y[i, j, t] + (1 - y[i, j, t])
					for i in model.i
					for j in model.j
					if (P_param[i, j, t] != 1)
					)
				for t in model.t
				]) # (6)
		constrain([
			S[t] == S[t - 1] - KI[t] * P[t] * S[t - 1] * I[t - 1]
			for t in t_except_first
			]) # (1)
		constrain([
			I[t] == I[t - 1] + KI[t] * P[t] * S[t - 1] * I[t - 1] -
			KR[t] * I[t - 1] - KD[t] * I[t - 1] for t in t_except_first
			]) # (2)
		if allow_multiple_policies_per_period:
			constrain([
				sum(y[i, j, t]
					for j in model.j) <= 1
				for i in model.i
				for t in model.t
				]) # (7)
		else:
			constrain([
				sum(y[i, j, t]
					for j in model.j
					for i in model.i) <= 1
				for t in model.t
				]) # (7) at most one policy can be chosen each period
		# add_constraints(constr,[y[i,j,t]<= 1 for i in model.i for j in model.j for t in model.t]) # (8)
		constrain([
			z[i, j, t] >= y[i, j, t] - y[i, j, t - 1] for i in model.i
			for j in model.j for t in t_except_first
			]) # (9)
		constrain([z[i, j, 0] >= y[i, j, 0]
					for i in model.i
					for j in model.j]) # (9)
		# easiest way to make some policies "not exist" is to set them to zero if costs are infinite or utility is zero (P==1)
		for i in model.i:
			for j in model.j:
				for t in model.t:
					if (C_setup[i, j, t] == np.inf) or (C_policy[
						i, j, t] == np.inf) or (P_param[i, j, t] == 1):
						model.y[i, j, t].fix(0)
		# add_constraints(constr,[model.y[i,j,t]==0 for i in model.i for j in model.j for t in model.t if ((C_setup[i,j,t] == np.inf) or (C_policy[i,j,t] == np.inf) or (P_param[i,j,t]==1))])
	if force_no_policy and efficient:
		print("EFFICIENTLY FORCING NO POLICY")
		constrain([
			S[t] == S[t - 1] - KI[t] * S[t - 1] * I[t - 1]
			for t in t_except_first
			]) # (1)
		constrain([
			I[t] == I[t - 1] + KI[t] * S[t - 1] * I[t - 1] - KR[t] * I[t - 1] -
			KD[t] * I[t - 1] for t in t_except_first
			]) # (2)

	if force_no_policy and (not efficient):
		print("INEFFICIENTLY FORCING NO POLICY")
		for k, component in model.y.items():
			component.fix(0)
		for k, component in model.P.items():
			component.fix(1)

	model.get_cumulative_disease_cost = get_cumulative_disease_cost
	model.get_cumulative_policy_cost = get_cumulative_policy_cost
	model.get_objective_cumulative_TT_periods = get_objective_cumulative_TT_periods
	model.obj = pyo.Objective(
		expr=get_objective_cumulative_TT_periods(model, T), sense=pyo.minimize
		)

	# if force_no_policy:
	# 	add_constraints(constr,[y[i,j,t] == 0 for i in model.i for j in model.j for t in model.t])
	return model
