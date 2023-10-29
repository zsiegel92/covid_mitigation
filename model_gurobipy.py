import numpy as np
from math import prod
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from time import time
import gurobipy as gp
from gurobipy import GRB
from itertools import product

from model import get_SIRD_model
from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo


def get_pyomo_model_from_gurobi(model, argnames, modelvars, pyomo_model):
	# return model, modelvars
	for varname, vardict in modelvars.items():
		if hasattr(pyomo_model, varname):
			for ind, var in vardict.items():
				getattr(pyomo_model, varname)[ind].fix(var.X)
	return pyomo_model


def get_gurobi_solution(model, argnames, modelvars, pyomo_model, solver_params={}):
	start = time()
	model.optimize()
	end = time()
	timeToSolve = end - start
	objVal = model.getObjective().getValue()
	sol = None
	sol = {
		'Solver': [{
		'Status': model.status
			}],
		'Problem': [{
		'Lower bound': model.ObjBound,
		'Upper bound': objVal
			}]
		}
	pyomo_model = get_pyomo_model_from_gurobi(model, argnames, modelvars, pyomo_model)
	return objVal, timeToSolve, pyomo_model, sol, model, solver_params


def fix(var, val):
	var.lb = val
	var.ub = val


def get_model_and_solve_gurobi(
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
	**solver_params
	):
	argnames = locals().copy()
	print(solver_params)
	argnames['bilinear'] = False
	model = gp.Model('disaster_mitigation')

	modeli = list(pyo.RangeSet(0, m - 1))
	modelj = list(pyo.RangeSet(0, n - 1))
	modelt = list(pyo.RangeSet(0, T - 1))
	modelijt = list(product(modeli, modelj, modelt))
	t0 = modelt[0]
	t_except_first = list(t for t in modelt if t != t0)

	S = model.addVars(modelt, lb=0, ub=S0 + I0, obj=0, vtype=GRB.CONTINUOUS, name="S")
	I = model.addVars(modelt, lb=0, ub=S0 + I0, obj=0, vtype=GRB.CONTINUOUS, name="I")
	R = model.addVars(modelt, lb=0, ub=S0 + I0, obj=0, vtype=GRB.CONTINUOUS, name="R")
	D = model.addVars(modelt, lb=0, ub=S0 + I0, obj=0, vtype=GRB.CONTINUOUS, name="D")
	d = model.addVars(modelt, lb=0, ub=S0 + I0, obj=0, vtype=GRB.CONTINUOUS, name="d")
	y = model.addVars(modelijt, lb=0, ub=1, vtype=GRB.BINARY, name="y")
	P = model.addVars(modelt, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="P")
	Interactions = model.addVars(
		modelt, lb=0, ub=(S0 + I0)**2, vtype=GRB.CONTINUOUS, name="Interactions"
		)
	inside_log_policy_prob = model.addVars(
		modelijt, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="inside_log_policy_prob"
		)
	logP = model.addVars(modelt, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, name="logP")
	log_policy_prob = model.addVars(
		modelijt, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, name="log_policy_prob"
		)

	for t in modelt:
		logP[t].start = 0
		P[t].start = 1

	for (i, j, t) in modelijt:
		y[i, j, t].start = 0
		inside_log_policy_prob[i, j, t].start = 1
		log_policy_prob[i, j, t].start = 0

	fix(S[t0], S0)
	fix(I[t0], I0)
	fix(R[t0], R0)
	fix(D[t0], D0)
	fix(d[t0], 0)

	model.addConstrs(R[t] == R[t - 1] + KR[t] * I[t - 1] for t in t_except_first)
	model.addConstrs(R[t] == R[t - 1] + KR[t] * I[t - 1] for t in t_except_first) # (3)
	model.addConstrs(d[t] == KD[t] * I[t - 1] for t in t_except_first) # (4)
	model.addConstrs(D[t] == D[t - 1] + d[t] for t in t_except_first) # (5)

	model.addConstrs(
		S[t] == S[t - 1] - KI[t] * P[t] * Interactions[t - 1] for t in t_except_first
		) # (1)
	model.addConstrs(
		I[t] == I[t - 1] + KI[t] * P[t] * Interactions[t - 1] - KR[t] * I[t - 1] - KD[t] * I[t - 1]
		for t in t_except_first
		) # (2)
	model.addConstrs(Interactions[t] == S[t] * I[t] for t in modelt)

	model.addConstrs(
		inside_log_policy_prob[i, j, t] == P_param[i, j, t] * y[i, j, t] + (1 - y[i, j, t])
		for i in modeli for j in modelj for t in modelt if (P_param[i, j, t] != 1)
		)
	model.addConstrs(
		logP[t] == gp.quicksum(
		log_policy_prob[i, j, t] for i in modeli for j in modelj if (P_param[i, j, t] != 1)
			) for t in modelt
		) # (6)

	for t in modelt:
		model.addGenConstrLog(P[t], logP[t])
		for i in modeli:
			for j in modelj:
				if P_param[i, j, t] != 1:
					model.addGenConstrLog(inside_log_policy_prob[i, j, t], log_policy_prob[i, j, t])

	# model.addConstrs[ log_policy_prob[i,j,t] == pyo.log(inside_log_policy_prob[i,j,t]) for i in modeli for j in modelj for t in modelt if (P_param[i,j,t] !=1))
	# model.addConstrs(logP[t]== pyo.log(P[t])for t in modelt)

	if allow_multiple_policies_per_period:
		model.addConstrs(
			gp.quicksum(y[i, j, t] for j in modelj) <= 1 for i in modeli for t in modelt
			) # (7)
	else:
		model.addConstrs(
			gp.quicksum(y[i, j, t] for j in modelj for i in modeli) <= 1 for t in modelt
			) # (7) at most one policy can be chosen each period

	# easiest way to make some policies "not exist" is to set them to zero if costs are infinite or utility is zero (P==1)
	for i in modeli:
		for j in modelj:
			for t in modelt:
				if (C_setup[i, j, t] == np.inf) or (C_policy[i, j, t] == np.inf) or (P_param[i, j, t] == 1):
					fix(y[i, j, t], 0)

	setup_cost = gp.quicksum(
		C_setup[i, j, t] * y[i, j, t] for i in modeli
		for j in modelj for t in modelt if (C_setup[i, j, t] != np.inf)
		)
	if cost_per_susceptible_only:
		variable_cost = gp.quicksum(
			C_policy[i, j, t] * S[t] * y[i, j, t] for i in modeli
			for j in modelj for t in modelt if (C_policy[i, j, t] != np.inf)
			)
	else:
		variable_cost = gp.quicksum(
			C_policy[i, j, t] * (N / individual_multiplier) * y[i, j, t] for i in modeli
			for j in modelj for t in modelt if (C_policy[i, j, t] != np.inf)
			)
	policy_cost = (1 / cost_multiplier) * (setup_cost + variable_cost)
	disease_cost = (1 / cost_multiplier) * gp.quicksum(
		C_infected[t] * I[t] + C_death * d[t] for t in modelt
		)
	obj = disease_cost + policy_cost
	model.setObjective(obj, GRB.MINIMIZE)
	model.setParam('NonConvex', 2)
	model.setParam('MIPGap', solver_params['optGap'])
	localsdict = locals()
	modelvars = {
		k: localsdict[k]
		for k in (
		"S", "I", "R", "D", "d", "y", "P", "Interactions", "inside_log_policy_prob", "logP", "log_policy_prob"
			)
		}

	# pyomo_model = get_SIRD_model(m,n,T, N, KI, KR, KD, I0, S0, P_param, C_policy, C_setup, C_infected, C_death,individual_multiplier, days_per_period, cost_multiplier,policies,allow_multiple_policies_per_period=allow_multiple_policies_per_period,cost_per_susceptible_only=cost_per_susceptible_only,use_logarithm=use_logarithm, force_no_policy=force_no_policy,time_varying_costs_and_probabilities=time_varying_costs_and_probabilities)

	pyomo_model = get_SIRD_model(
		**{k: v
		for k, v in argnames.items()
		if k not in ('solver_params', 'bilinear')}
		)
	return get_gurobi_solution(model, argnames, modelvars, pyomo_model, solver_params)


def get_cumulative_disease_cost_gurobipy(model, TT):
	return (1 / model.cost_multiplier) * sum(
		model.C_infected[t] * model.I[t] + model.C_death * model.d[t] for t in range(TT)
		)


def get_cumulative_policy_cost_gurobipy(model, TT):
	setup_cost = 0
	variable_cost = 0
	setup_cost = sum(
		model.C_setup[i, j, t] * model.y[i, j, t] for i in model.i
		for j in model.j for t in range(TT) if (model.C_setup[i, j, t] != np.inf)
		)
	if model.cost_per_susceptible_only:
		variable_cost = sum(
			model.C_policy[i, j, t] * model.S[t] * model.y[i, j, t] for i in model.i
			for j in model.j for t in range(TT) if (model.C_policy[i, j, t] != np.inf)
			)
	else:
		variable_cost = sum(
			model.C_policy[i, j, t] * (model.N / model.individual_multiplier) * model.y[i, j, t]
			for i in model.i for j in model.j for t in range(TT) if (model.C_policy[i, j, t] != np.inf)
			)
	return (1 / model.cost_multiplier) * (setup_cost + variable_cost)


def get_objective_cumulative_TT_periods_gurobipy(model, TT):
	disease_cost = get_cumulative_disease_cost_gurobi_capable(model, TT)
	if model.force_no_policy:
		return disease_cost
	else:
		policy_cost = get_cumulative_policy_cost_gurobi_capable(model, TT)
		return disease_cost + policy_cost
