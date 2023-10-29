import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
from time import time
import numpy as np

import pyomo
import pyomo.environ as pyo
from pyomo.environ import RangeSet, Set

from pyomo.solvers.plugins.solvers.GAMS import GAMSDirect
import subprocess

from pyomo_utils import solve_gams_direct, add_constraints, solve_executable, solve_pyomo
from model import get_SIRD_model, no_policy_model, get_params
from model_bilinear import get_SIRD_model_bilinear, convert_int_to_policy_assortment, convert_policy_assortment_to_int


def test_gurobi_bilinear():
	m = 2 # i = {1,...,m} policies
	n = 1 # j = {1,...,n} levels
	T = 150 # t = {1,...,T} periods
	individual_multiplier = 1000
	days_per_period=7
	N = 300000
	KI0 = 0.0000006
	KD0 = 0.015
	KR0 = 0.03

	solver = "Gurobi"
	solve_from_binary = False
	system_params = dict(m=m,n=n,T=T,individual_multiplier=individual_multiplier,days_per_period=days_per_period,N=N,)
	instance_params = dict(KI0=KI0,KR0=KR0,KD0=KD0,C1=20,C2=10,P1=0.5,P2=0.7,c_setup=100000,c_infected=10000,c_death=20000,num_infected0=100)
	param_kwargs = dict(**system_params, **instance_params)
	model_kwargs = dict(allow_multiple_policies_per_period=True,cost_per_susceptible_only=False, use_logarithm=False,)
	df_control = no_policy_model(**param_kwargs)

	sols = {}
	for force_no_policy in (True,False):
		solver_params = dict(
								solver=solver,
								tee=True,
								keepfiles=True,
								warmstart=True,
								solve_from_binary=solve_from_binary
							)
		print("\n\n")
		print(solver, solve_from_binary)
		model = get_SIRD_model_bilinear(*get_params(**param_kwargs),**model_kwargs,force_no_policy=force_no_policy)

		sols[solver, solve_from_binary,force_no_policy] = solve_pyomo(model,**solver_params)

		# solver,solve_from_binary = "BARON", False
		objVal,_,model = sols[solver,solve_from_binary,False]
		objVal_no_policy,_,model_no_policy = sols[solver,solve_from_binary,True]
		df = get_results_df(model,system_params)
		df_no_policy = get_results_df(model_no_policy,system_params)
		plot_results(df,df_no_policy,objVal,objVal_no_policy,param_kwargs,solver=solver)


# ("COUENNE", True),  ("BONMIN", True),("BONMIN", False),("DICOPT", False),
def test_solvers(solvers = (("BARON",False))):
	m = 2 # i = {1,...,m} policies
	n = 1 # j = {1,...,n} levels
	T = 150 # t = {1,...,T} periods
	individual_multiplier = 1000
	# cost_multiplier = 100
	days_per_period=7

	# From EM & PM spreadsheet
	N = 300000

	KI0 = 0.0000006 #.0000003 * 10000
	KD0 = 0.015 #0.004006376
	KR0 = 0.03# 0.035535187


	solvers = ( ("BARON", False),)
	system_params = dict(m=m,n=n,T=T,individual_multiplier=individual_multiplier,days_per_period=days_per_period,N=N,)
	instance_params = dict(KI0=KI0,KR0=KR0,KD0=KD0,C1=20,C2=10,P1=0.5,P2=0.7,c_setup=100000,c_infected=10000,c_death=20000,num_infected0=100)
	param_kwargs = dict(**system_params, **instance_params)
	model_kwargs = dict(allow_multiple_policies_per_period=True,cost_per_susceptible_only=False, use_logarithm=False,)
	df_control = no_policy_model(**param_kwargs)

	sols = {}
	for solver, solve_from_binary in solvers:
		for force_no_policy in (True,False):
			solver_params = dict(
									solver=solver,
									tee=True,
									keepfiles=True,
									warmstart=True,
									solve_from_binary=solve_from_binary
								)
			print("\n\n")
			print(solver, solve_from_binary)
			model = get_SIRD_model(*get_params(**param_kwargs),**model_kwargs,force_no_policy=force_no_policy)

			sols[solver, solve_from_binary,force_no_policy] = solve_pyomo(model,**solver_params)

		# solver,solve_from_binary = "BARON", False
		objVal,_,model = sols[solver,solve_from_binary,False]
		objVal_no_policy,_,model_no_policy = sols[solver,solve_from_binary,True]
		df = get_results_df(model,system_params)
		df_no_policy = get_results_df(model_no_policy,system_params)
		plot_results(df,df_no_policy,objVal,objVal_no_policy,param_kwargs,solver=solver)



	if True:
		solver,solve_from_binary, force_no_policy = "BARON", False, False
		objVal, timeToSolve,model = sols[solver,solve_from_binary,force_no_policy]
		print(f"solver: {solver}, solve_from_binary: {solve_from_binary}, Forced no policy: {force_no_policy}. Objective: {objVal}")
		df = get_results_df(model,system_params)

		solver,solve_from_binary, force_no_policy = "BARON", False, True
		objVal_no_policy, timeToSolve_no_policy,model_no_policy = sols[solver,solve_from_binary,force_no_policy]
		df_no_policy = get_results_df(model_no_policy,system_params)
		print(df_no_policy.S)
