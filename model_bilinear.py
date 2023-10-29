import numpy as np
from math import prod
import pandas as pd
import pyomo.environ as pyo
from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo
from general_utils import to_base_b, from_base_b,is_pareto_inefficient, is_pareto_efficient,assign,convert_int_to_policy_assortment, convert_policy_assortment_to_int


yInitVal = 1

def get_cumulative_disease_cost_bilinear(model,TT):
	return (1/model.cost_multiplier)*sum(model.C_infected[t]*model.I[t] + model.C_death*model.d[t] for t in range(TT))


def get_cumulative_policy_cost_bilinear(model,TT):
	setup_cost = 0
	variable_cost = 0
	if model.time_varying_costs_and_probabilities:
		get_param_time_varying = lambda param_matrix, assortment_index, t: param_matrix[assortment_index,t]
	else:
		get_param_time_varying = lambda param_matrix, assortment_index, t: param_matrix[assortment_index]

	setup_cost = sum(get_param_time_varying(model.C_setup_assortment,assortment_index,t)*model.y[assortment_index,t] for assortment_index in model.policy_assortments for t in range(TT) if get_param_time_varying(model.C_setup_assortment,assortment_index,t) != np.inf)
	if model.cost_per_susceptible_only:
		variable_cost = sum(get_param_time_varying(model.C_policy_assortment,assortment_index,t)*model.S[t]*model.y[assortment_index,t] for assortment_index in model.policy_assortments for t in range(TT) if get_param_time_varying(model.C_policy_assortment,assortment_index,t) != np.inf)
	else:
		variable_cost = sum(get_param_time_varying(model.C_policy_assortment,assortment_index,t)*(model.N/model.individual_multiplier)*model.y[assortment_index,t] for assortment_index in model.policy_assortments for t in range(TT) if get_param_time_varying(model.C_policy_assortment,assortment_index,t) != np.inf)

	return (1/model.cost_multiplier)*(setup_cost + variable_cost)




def get_objective_cumulative_TT_periods_bilinear(model,TT):
	disease_cost = get_cumulative_disease_cost_bilinear(model,TT)
	if model.force_no_policy:
		return disease_cost
	else:
		if model.allow_multiple_policies_per_period:
			policy_cost = get_cumulative_policy_cost_bilinear(model,TT)
		else:
			policy_cost = get_cumulative_policy_cost_one_policy_per_period_bilinear(model,TT)
		return disease_cost + policy_cost

def get_cumulative_policy_cost_one_policy_per_period_bilinear(model,TT):
	setup_cost = 0
	variable_cost = 0

	setup_cost = sum(model.policies[i].C_setup_[0]*model.y[i,t] for i in model.i for t in range(TT))

	if model.cost_per_susceptible_only:
		variable_cost = sum(model.policies[i].C_[0]*model.S[t]*model.y[i,t] for i in model.i for t in range(TT))

	else:
		variable_cost = sum(model.policies[i].C_[0]*(model.N/model.individual_multiplier)*model.y[i,t] for i in model.i for t in range(TT))
	return (1/model.cost_multiplier)*(setup_cost + variable_cost)

def get_SIRD_one_policy_per_period(model):
	constr = model.constr
	S = model.S
	I = model.I
	R = model.R
	D = model.D
	d = model.d
	model.y = pyo.Var(model.i*model.t, domain=pyo.Binary, initialize=yInitVal)
	add_constraints(model.constr,[model.P[t] == sum(model.policies[i].P_[0]*model.y[i,t] for i in model.i) for t in model.t])
	add_constraints(model.constr,[sum(model.y[i,t] for i in model.i) == 1 for t in model.t]) #exactly one policy assortment is used per period (possibly the "no policy" assortment)
	add_constraints(model.constr,[S[t] == S[t-1] - model.KI[t]*model.P[t]*model.Interactions[t-1] for t in model.t_except_first ]) # (1)
	add_constraints(model.constr,[I[t] == I[t-1] + model.KI[t]*model.P[t]*model.Interactions[t-1] - model.KR[t]*I[t-1] - model.KD[t]*I[t-1] for t in model.t_except_first]) # (2)
	add_constraints(model.constr,[model.Interactions[t] == S[t] * I[t] for t in model.t])
	model.obj = pyo.Objective(expr=get_objective_cumulative_TT_periods_bilinear(model,model.T),sense=pyo.minimize)
	return model

def get_SIRD_model_bilinear(m,n,T, N, KI, KR, KD, I0, S0,D0, R0, P_param, C_policy, C_setup, C_infected, C_death,individual_multiplier, days_per_period, cost_multiplier, policies, allow_multiple_policies_per_period=True,cost_per_susceptible_only=False,use_logarithm=False, force_no_policy=False,prune=False,time_varying_costs_and_probabilities=False):
	varnames = locals().copy() #stores *args and **kwargs

	model = pyo.ConcreteModel()

	# stores all arguments of this function as attributes of model for access later on
	for k,v in varnames.items():
		setattr(model,k,v) #

	def convert_int_to_policy_assortment_specific(assortment_index):
		return convert_int_to_policy_assortment(assortment_index,m,n)

	def convert_policy_assortment_to_int_specific(policy_assortment):
		return convert_policy_assortment_to_int(policy_assortment,n)

	model.convert_int_to_policy_assortment_specific = convert_int_to_policy_assortment_specific
	model.convert_policy_assortment_to_int_specific = convert_policy_assortment_to_int_specific
	model.bilinear=True

	model.i = pyo.RangeSet(0,m-1)
	model.j = pyo.RangeSet(0,n-1)
	model.t = pyo.RangeSet(0,T-1)
	t0 = model.t.first() # do not have to start indices at 0.
	model.t_except_first = list(t for t in model.t if t != t0) #periods except first period

	model.policy_assortments = pyo.RangeSet(0,((n+1)**(m))-1) #each of m policies has n levels
	# model.sparse_index_set = policies.sparse_index_set

	model.S = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=S0,bounds=(0,S0+I0))
	model.I = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0,bounds=(0,S0+I0))
	model.R = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0,bounds=(0,S0+I0))
	model.D = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0,bounds=(0,S0+I0))
	model.d = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0,bounds=(0,S0+I0))
	model.constr = pyo.ConstraintList()
	S = model.S
	I = model.I
	R = model.R
	D = model.D
	d = model.d

	constr = model.constr

	S[t0].fix(S0)
	I[t0].fix(I0)
	R[t0].fix(R0)
	D[t0].fix(D0)
	d[t0].fix(0)

	add_constraints(constr,[R[t] == R[t-1] + KR[t]*I[t-1] for t in model.t_except_first]) # (3)
	add_constraints(constr,[d[t] == KD[t]*I[t-1] for t in model.t_except_first]) # (4)
	add_constraints(constr,[D[t] == D[t-1] + d[t] for t in model.t_except_first]) # (5)

	if not force_no_policy:
		model.P = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=1,bounds=(0,1))
		model.Interactions = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0,bounds=(0,(S0+I0)**2))
		add_constraints(constr,[S[t] == S[t-1] - KI[t]*model.P[t]*model.Interactions[t-1] for t in model.t_except_first ]) # (1)
		add_constraints(constr,[I[t] == I[t-1] + KI[t]*model.P[t]*model.Interactions[t-1] - KR[t]*I[t-1] - KD[t]*I[t-1] for t in model.t_except_first]) # (2)
		add_constraints(constr,[model.Interactions[t] == S[t] * I[t] for t in model.t])

		if not model.allow_multiple_policies_per_period:
			return get_SIRD_one_policy_per_period(model)

		print("Constructing assortment cost and probability arrays")
		policies.ensure_assortment_params(model.time_varying_costs_and_probabilities)
		for k in ('C_setup_assortment','C_policy_assortment','P_assortment',):
			setattr(model,k,getattr(policies,k))

		print(f"Creating ({(n+1)}^{m}-1)*{T} = {(n+1)**m-1}*{T} = {(((n+1)**(m))-1)*T} binary variables in matrix")
		model.policy_assortments.display()
		model.t.display()

		if model.time_varying_costs_and_probabilities:
			get_param_mat = lambda param_matrix: param_matrix[:,0]
			set_param_val = lambda param_matrix,assortment_index, t, val: assign(param_matrix,(assortment_index,t),val)
			get_param_val = lambda param_matrix, assortment_index, t: param_matrix[assortment_index,t]
		else:
			get_param_mat = lambda param_matrix: param_matrix
			set_param_val = lambda param_matrix,assortment_index, t, val: assign(param_matrix, (assortment_index,), val)
			get_param_val = lambda param_matrix, assortment_index, t: param_matrix[assortment_index]

		if prune:
			print("PRUNING")
			to_prune = is_pareto_inefficient(np.stack([get_param_mat(model.C_setup_assortment),get_param_mat(model.C_policy_assortment), get_param_mat(model.P_assortment)],axis=-1))
			print("Determined inefficient policy assortments...")
			for assortment_index in to_prune:
				for t in model.t:
					set_param_val(model.C_setup_assortment,assortment_index,t,np.inf)
					set_param_val(model.C_policy_assortment,assortment_index,t,np.inf)
					set_param_val(model.P_assortment,assortment_index,t,1)
					# model.y[assortment_index,t].fix(0)
			print(f"Pruned {len(to_prune)} (unique: {len(set(to_prune))}) out of {model.C_setup_assortment.shape[0]} possible policies (some of which are not valid)")

		model.y = pyo.Var(model.policy_assortments*model.t, domain=pyo.Binary, initialize=yInitVal)
		add_constraints(constr,[model.P[t] == sum(get_param_val(model.P_assortment,assortment_index,t) *model.y[assortment_index,t] for assortment_index in model.policy_assortments) for t in model.t])
		add_constraints(constr,[sum(model.y[assortment_index,t] for assortment_index in model.policy_assortments) == 1 for t in model.t]) #exactly one policy assortment is used per period (possibly the "no policy" assortment)

	else:
		model.policy_assortments = pyo.RangeSet(0,(n**(m+1))-1) #each of m policies has n levels
		model.P = pyo.Param(model.t,default=1)
		add_constraints(constr,[S[t] == S[t-1] - KI[t]*model.P[t]*S[t-1]*I[t-1] for t in model.t_except_first ]) # (1)
		add_constraints(constr,[I[t] == I[t-1] + KI[t]*model.P[t]*S[t-1]*I[t-1] - KR[t]*I[t-1] - KD[t]*I[t-1] for t in model.t_except_first]) # (2)

	model.get_cumulative_disease_cost = get_cumulative_disease_cost_bilinear
	model.get_cumulative_policy_cost = get_cumulative_policy_cost_bilinear
	model.get_objective_cumulative_TT_periods = get_objective_cumulative_TT_periods_bilinear
	model.obj = pyo.Objective(expr=get_objective_cumulative_TT_periods_bilinear(model,T),sense=pyo.minimize)

	return model
