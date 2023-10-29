import numpy as np
from math import prod
import pandas as pd
import pyomo.environ as pyo
from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo
from general_utils import to_base_b, from_base_b,is_pareto_inefficient, is_pareto_efficient,assign,convert_int_to_policy_assortment, convert_policy_assortment_to_int


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
		policy_cost = get_cumulative_policy_cost_bilinear(model,TT)
		return disease_cost + policy_cost



def get_SIRD_model_bilinear(m,n,T, N, KI, KR, KD, I0, S0, P_param, C_policy, C_setup, C_infected, C_death,individual_multiplier, days_per_period, cost_multiplier, policies, allow_multiple_policies_per_period=False,cost_per_susceptible_only=False,use_logarithm=False, force_no_policy=False,prune=True,time_varying_costs_and_probabilities=False):
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
	t_except_first = list(t for t in model.t if t != t0) #periods except first period

	model.policy_assortments = pyo.RangeSet(0,((n+1)**(m))-1) #each of m policies has n levels
	# model.sparse_index_set = policies.sparse_index_set


	model.S = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=N)
	model.I = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.R = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.D = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.d = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)
	model.constr = pyo.ConstraintList()
	S = model.S
	I = model.I
	R = model.R
	D = model.D
	d = model.d

	constr = model.constr
	if not force_no_policy:
		print("Constructing assortment cost and probability arrays")

		# policies.ensure_assortment_params(model.time_varying_costs_and_probabilities)
		# for k in ('C_setup_assortment','C_policy_assortment','P_assortment',):
		# 	setattr(model,k,getattr(policies,k))

		if model.time_varying_costs_and_probabilities:
			policies.ensure_assortment_params(model.time_varying_costs_and_probabilities)

			model.C_setup_assortment = np.array(
				[[sum(model.C_setup[i,j,t] for i,j in enumerate(model.convert_int_to_policy_assortment_specific(assortment_index)) if j>=0) for t in model.t] for assortment_index in model.policy_assortments]
			) #sum defaults to 0 for empty iterable
			model.C_policy_assortment = np.array([[sum(model.C_policy[i,j,t] for i,j in enumerate(model.convert_int_to_policy_assortment_specific(assortment_index)) if j>=0) for t in model.t] for assortment_index in model.policy_assortments]) #sum defaults to 0 for empty iterable
			model.P_assortment = np.array([[pyo.prod(P_param[i,j,t] for i,j in enumerate(convert_int_to_policy_assortment_specific(assortment_index)) if j>=0) for t in model.t] for assortment_index in model.policy_assortments]) #pyo.prod defaults to 1 for empty iterable
		else:
			model.C_setup_assortment = np.array(
				[sum(model.C_setup[i,j,0] for i,j in enumerate(model.convert_int_to_policy_assortment_specific(assortment_index)) if j>=0) for assortment_index in model.policy_assortments]
			) #sum defaults to 0 for empty iterable
			model.C_policy_assortment = np.array([sum(model.C_policy[i,j,0] for i,j in enumerate(model.convert_int_to_policy_assortment_specific(assortment_index)) if j>=0) for assortment_index in model.policy_assortments]) #sum defaults to 0 for empty iterable
			model.P_assortment = np.array([pyo.prod(P_param[i,j,0] for i,j in enumerate(convert_int_to_policy_assortment_specific(assortment_index)) if j>=0) for assortment_index in model.policy_assortments]) #prod defaults to 1 for empty iterable

		print(f"Creating ({(n+1)}^{m}-1)*{T} = {(n+1)**m-1}*{T} = {(((n+1)**(m))-1)*T} binary variables in matrix")
		model.policy_assortments.display()
		model.t.display()
		# model.y = pyo.Var(model.policy_assortments*model.t, domain=pyo.Binary, initialize=0)



		if model.time_varying_costs_and_probabilities:
			get_param_mat = lambda param_matrix: param_matrix[:,0]
			set_param_val = lambda param_matrix,assortment_index, t, val: assign(param_matrix,(assortment_index,t),val)
			get_param_val = lambda param_matrix, assortment_index, t: param_matrix[assortment_index,t]
		else:
			get_param_mat = lambda param_matrix: param_matrix
			set_param_val = lambda param_matrix,assortment_index, t, val: assign(param_matrix, (assortment_index,), val)
			get_param_val = lambda param_matrix, assortment_index, t: param_matrix[assortment_index]

		# model.y = pyo.Var(model.policy_assortments*model.t, domain=pyo.Binary, initialize=0)
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

			# model.sparse_int_indices = is_pareto_efficient(np.stack([get_param_mat(m.C_setup_assortment),get_param_mat(m.C_policy_assortment), get_param_mat(m.P_assortment)],axis=-1))

			def initialize_pruned_sparse_indices(m):
				sparse_int_indices = is_pareto_efficient(np.stack([get_param_mat(m.C_setup_assortment),get_param_mat(m.C_policy_assortment), get_param_mat(m.P_assortment)],axis=-1))

				keeper_indices = [convert_int_to_policy_assortment_specific(assortment_index) for assortment_index in sparse_int_indices]
				return keeper_indices

			model.sparse_tuple_indices = pyo.Set(initialize=initialize_pruned_sparse_indices)
			model.sparse_tuple_indices.construct()
			model.y = pyo.Var(model.sparse_tuple_indices*model.t, domain=pyo.Binary, initialize=0)


			model.P = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=1)
			model.Interactions = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)


			# model.sparse_tuple_indices convert_policy_assortment_to_int_specific

			add_constraints(constr,[model.P[t] == sum(get_param_val(model.P_assortment,convert_policy_assortment_to_int_specific(assortment),t)*model.y[assortment,t] for assortment in model.sparse_tuple_indices) for t in model.t])


			# add_constraints(constr,[model.P[t] == sum(model.P_assortment[(assortment_index:=convert_policy_assortment_to_int_specific(assortment)),t]*model.y[assortment,t] for assortment in model.sparse_tuple_indices) for t in model.t])
			add_constraints(constr,[sum(model.y[assortment,t] for assortment in model.sparse_tuple_indices) == 1 for t in model.t]) #exactly one policy assortment is used per period (possibly the "no policy" assortment)

			add_constraints(constr,[S[t] == S[t-1] - KI[t]*model.P[t]*model.Interactions[t-1] for t in t_except_first ]) # (1)
			add_constraints(constr,[I[t] == I[t-1] + KI[t]*model.P[t]*model.Interactions[t-1] - KR[t]*I[t-1] - KD[t]*I[t-1] for t in t_except_first]) # (2)

			add_constraints(constr,[model.Interactions[t] == S[t] * I[t] for t in model.t])


		else:
			model.y = pyo.Var(model.policy_assortments*model.t, domain=pyo.Binary, initialize=0)
			model.P = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=1)
			model.Interactions = pyo.Var(model.t,domain=pyo.NonNegativeReals,initialize=0)

			add_constraints(constr,[model.P[t] == sum(model.P_assortment[assortment_index,t]*model.y[assortment_index,t] for assortment_index in model.policy_assortments) for t in model.t])
			add_constraints(constr,[sum(model.y[assortment,t] for assortment in model.policy_assortments) == 1 for t in model.t]) #exactly one policy assortment is used per period (possibly the "no policy" assortment)

			add_constraints(constr,[S[t] == S[t-1] - KI[t]*model.P[t]*model.Interactions[t-1] for t in t_except_first ]) # (1)
			add_constraints(constr,[I[t] == I[t-1] + KI[t]*model.P[t]*model.Interactions[t-1] - KR[t]*I[t-1] - KD[t]*I[t-1] for t in t_except_first]) # (2)

			add_constraints(constr,[model.Interactions[t] == S[t] * I[t] for t in model.t])



	else:
		model.policy_assortments = pyo.RangeSet(0,(n**(m+1))-1) #each of m policies has n levels
		model.P = pyo.Param(model.t,default=1)
		add_constraints(constr,[S[t] == S[t-1] - KI[t]*model.P[t]*S[t-1]*I[t-1] for t in t_except_first ]) # (1)
		add_constraints(constr,[I[t] == I[t-1] + KI[t]*model.P[t]*S[t-1]*I[t-1] - KR[t]*I[t-1] - KD[t]*I[t-1] for t in t_except_first]) # (2)


	model.obj = pyo.Objective(expr=get_objective_cumulative_TT_periods_bilinear(model,T),sense=pyo.minimize)

	S[t0].fix(S0)
	I[t0].fix(I0)
	R[t0].fix(0)
	D[t0].fix(0)
	d[t0].fix(0)
	# constr.add(S[t0] == S0)
	# constr.add(I[t0] == I0)
	# constr.add(R[t0] == 0)
	# constr.add(D[t0] == 0)
	# constr.add(d[t0] == 0)

	add_constraints(constr,[R[t] == R[t-1] + KR[t]*I[t-1] for t in t_except_first]) # (3)
	add_constraints(constr,[d[t] == KD[t]*I[t-1] for t in t_except_first]) # (4)
	add_constraints(constr,[D[t] == D[t-1] + d[t] for t in t_except_first]) # (5)
	# if force_no_policy:
	# 	add_constraints(constr,[model.y[i,j,t] == 0 for i in model.i for j in model.j for t in model.t])
	return model
