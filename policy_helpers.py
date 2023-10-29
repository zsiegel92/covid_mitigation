import numpy as np
import pandas as pd

import math

from collections import UserList
import pyomo.environ as pyo
from general_utils import to_base_b, from_base_b, is_pareto_inefficient, is_pareto_efficient, assign, convert_int_to_policy_assortment, convert_policy_assortment_to_int

is_iterable = lambda obj: hasattr(obj, '__iter__') or hasattr(obj, '__getitem__')


class LatexPolicy:
	def __init__(self, policy):
		self.policy = policy
		self.name = f"``{policy.name}\""
		self.A = self.latecize(policy.C_setup_)
		self.B = self.latecize(policy.C_switch_)
		self.C = self.latecize(policy.C_)
		self.P = self.latecize(policy.P_)
		# P_, C_, C_setup, C_switch, Name

	def scientific_notation(self, num):
		base10 = math.log10(abs(num))
		if base10 < 0:
			base10 = -1 * abs(math.floor(base10))
		else:
			base10 = abs(math.floor(base10))
		main = num / (10**base10)
		if main == int(main):
			main = int(main)
		return (main, base10)

	def format_num(self, num):
		if type(num) == int and num % 10 == 0 and num != 0:
			sci_not = self.scientific_notation(num)
			mathrm = "\\mathrm"
			form = f"{sci_not[0]}{mathrm}{{e}}{{{sci_not[1]}}}"
		else:
			form = f"{num}"
		if form.startswith("0."):
			form = form.lstrip("0")
		return form

	def latecize(self, param_list):
		return "$[" + ",".join([self.format_num(num) for num in param_list]) + "]$"


class Policy():
	# C_setup can be a list of costs or a scalar if setup costs are the same at every level
	# should either give time-varying parameters or a time parameter
	def __init__(
		self,
		C,
		P,
		C_setup,
		C_switch,
		T=None,
		name='',
		emptyname_is_empty=False,
		assortment_composition=None
		):
		try:
			self.C_ = C
			self.P_ = P
			self.C_setup_ = C_setup
			self.C_switch_ = C_switch
			self.C = np.array(C, dtype=float)
			self.P = np.array(P, dtype=float)
			self.n = self.C.shape[0]
			assert self.C.shape == self.P.shape
			assert (T is not None) or (
				len(self.C.shape) == 2
				) # should either give time-varying parameters or a time parameter

			if is_iterable(C_setup):
				self.C_setup = np.array(C_setup, dtype=float)
				self.C_switch = np.array(C_switch, dtype=float)
				assert self.C_setup.shape == self.C.shape
				assert self.C_switch.shape == self.C.shape
			else:
				self.C_setup = np.array([C_setup for j in range(len(C))], dtype=float)
				self.C_switch = np.array([C_switch for j in range(len(C))], dtype=float)
			self.name = name
			self.emptyname_is_empty = emptyname_is_empty
			if not name and emptyname_is_empty:
				self.name = 'No policy used'
			self.assortment_composition = assortment_composition
			self.T = T
			self.padded = False
		except Exception as e:
			print(f"Error initializing policy {name}")
			raise e

	def pad(self, n):
		np.pad([1, 2, 3], (0, 4), mode="constant", constant_values=(-1, 5))
		if len(self.C) < n:
			self.C = np.pad(
				self.C, (0, n - len(self.C)), mode="constant", constant_values=(-1, np.inf)
				)
			self.C_setup = np.pad(
				self.C_setup, (0, n - len(self.C_setup)),
				mode="constant",
				constant_values=(-1, np.inf)
				)
			self.C_switch = np.pad(
				self.C_switch, (0, n - len(self.C_switch)),
				mode="constant",
				constant_values=(-1, np.inf)
				)
			self.P = np.pad(self.P, (0, n - len(self.P)), mode="constant", constant_values=(-1, 1))
		self.padded = True
		# while len(self.C) < n:
		# 	self.C.append(np.inf)
		# 	self.C_setup.append(np.inf)
		# 	self.P.append(1)

	# Should occur after self.pad()
	def make_time_varying(self):
		assert self.padded
		self.C = np.stack([self.C.copy() for t in range(self.T)], axis=-1)
		self.P = np.stack([self.P.copy() for t in range(self.T)], axis=-1)
		self.C_setup = np.stack([self.C_setup.copy() for t in range(self.T)], axis=-1)
		self.C_switch = np.stack([self.C_switch.copy() for t in range(self.T)], axis=-1)

	def clone(self, only_level=None):
		if only_level is not None:
			return Policy(
				C=[self.C_[only_level]],
				P=[self.P_[only_level]],
				C_setup=[self.C_setup_[only_level]],
				C_switch=[self.C_switch_[only_level]],
				T=self.T,
				name=f"{self.name} level {only_level}",
				assortment_composition=self.assortment_composition
				)
		return Policy(
			C=self.C_,
			P=self.P_,
			C_setup=self.C_setup_,
			C_switch=self.C_switch_,
			T=self.T,
			name=self.name,
			assortment_composition=self.assortment_composition
			)

	def reprself(self):
		return '\n\t'.join((
			f'\t{self.name}:', f'C={self.C_}', f'A={self.C_setup_}', f'B={self.C_switch_}',
			f'P={self.P_}'
			))

	def __repr__(self):
		return '\n\t'.join((
			f'\t{self.name}:', f'C={self.C_}', f'A={self.C_setup_}', f'B={self.C_switch_}',
			f'P={self.P_}', f'T={self.T}'
			))

	@property
	def latex(self):
		return LatexPolicy(self)


class PolicyList(UserList):
	def initialize_sparse_index_set_before_padding(self):
		def initialize_assortments(model):
			return np.prod(np.array([pyo.RangeSet(0, policy.n) for policy in self], dtype='object'))

		self.sparse_index_set = pyo.Set(dimen=self.m, initialize=initialize_assortments)
		self.sparse_index_set.construct()
		# self.sparse_index_set.discard((0,2)) # interface example

	def pad_policies(self):
		# self.initialize_sparse_index_set_before_padding()
		for policy in self:
			assert isinstance(policy, Policy)
			assert policy.T == self.T
			policy.pad(self.n)
			policy.make_time_varying()

	def __init__(self, *args, originList=None, printing=True):
		super().__init__(args) # constructs self as list-like
		args = sorted(args, key=lambda policy: policy.n, reverse=True)
		self.m = len(args)
		self.n = max(len(policy.P) for policy in args)
		self.T = self[0].T

		self.num_assortments = np.prod([policy.n + 1 for policy in self])
		self.num_assortment_indices = (self.n + 1)**self.m
		self.pad_policies()

		self.P_param = np.full((self.m, self.n, self.T), 1, dtype=float)
		self.C_policy = np.full((self.m, self.n, self.T), np.inf)
		self.C_setup = np.full((self.m, self.n, self.T), np.inf)
		self.C_switch = np.full((self.m, self.n, self.T), np.inf)
		for i in range(self.m):
			for j in range(self.n):
				for t in range(self.T):
					self.P_param[i, j, t] = self[i].P[j, t]
					self.C_policy[i, j, t] = self[i].C[j, t]
					self.C_setup[i, j, t] = self[i].C_setup[j, t]
					self.C_switch[i, j, t] = self[i].C_switch[j, t]
		for i, policy in enumerate(self):
			if not policy.name:
				policy.name = f'Policy {i+1}' #index at 1
		if printing:
			print(
				f"Considering policies ({self.summary()}) with {self.num_assortments} assortments and {self.num_assortment_indices} assortment indices!"
				)
		if originList is not None:
			self.originList = originList
			self.num_assortments = originList.num_assortments
			self.num_assortment_indices = originList.num_assortment_indices
		else:
			self.originList = self

		self.assortment_compositions = [{
			i + 1: j + 1 for i, j in enumerate(policy.assortment_composition) if j > -1
			} for policy in self if policy.assortment_composition is not None]
		self.is_pruned = False

	def clone_some(self, inds, levels=False):
		# l = [policy.clone() for i,policy in enumerate(self) if i in inds]
		# pl = PolicyList(*l,originList=self.originList)
		try:
			if is_iterable(inds[0]):
				levels = True
				print("Passed multiple-level indices to clone_some. Specifying levels.")
		except:
			pass
		if levels:
			return PolicyList(
				*[self[i].clone(only_level=j) for i, j in inds], originList=self.originList
				).with_attributes_set(is_pruned=self.is_pruned)
		return PolicyList(
			*[self[i].clone() for i in inds], originList=self.originList
			).with_attributes_set(is_pruned=self.is_pruned)

	def convert_int_to_policy_assortment_specific(self, assortment_index):
		return convert_int_to_policy_assortment(assortment_index, self.m, self.n)

	def convert_policy_assortment_to_int_specific(self, policy_assortment):
		return convert_policy_assortment_to_int(policy_assortment, self.n)

	def ensure_assortment_params(self, time_varying_costs_and_probabilities=False):
		if hasattr(self, 'policy_assortments') and hasattr(self, 'C_setup_assortment') and hasattr(
			self, 'C_policy_assortment'
			) and hasattr(self, 'P_assortment') and hasattr(self, 'C_switch_assortment'):
			return

		self.time_varying_costs_and_probabilities = time_varying_costs_and_probabilities
		self.policy_assortments = pyo.RangeSet(0, ((self.n + 1)**(self.m)) - 1)
		if self.time_varying_costs_and_probabilities:
			self.C_setup_assortment = np.array([[
				sum(
					self.C_setup[i, j, t] for i, j in enumerate(
						self.convert_int_to_policy_assortment_specific(assortment_index)
						) if j >= 0
					) for t in range(self.T)
				] for assortment_index in self.policy_assortments]
												) #sum defaults to 0 for empty iterable
			self.C_policy_assortment = np.array([[
				sum(
					self.C_policy[i, j, t] for i, j in enumerate(
						self.convert_int_to_policy_assortment_specific(assortment_index)
						) if j >= 0
					) for t in range(self.T)
				] for assortment_index in self.policy_assortments]
												) #sum defaults to 0 for empty iterable
			self.C_switch_assortment = np.array([[
				sum(
					self.C_switch[i, j, t] for i, j in enumerate(
						self.convert_int_to_policy_assortment_specific(assortment_index)
						) if j >= 0
					) for t in range(self.T)
				] for assortment_index in self.policy_assortments]
												) #sum defaults to 0 for empty iterable
			self.P_assortment = np.array([[
				pyo.prod(
					self.P_param[i, j, t] for i, j in enumerate(
						self.convert_int_to_policy_assortment_specific(assortment_index)
						) if j >= 0
					) for t in range(self.T)
				] for assortment_index in self.policy_assortments]
											) #pyo.prod defaults to 1 for empty iterable
		else:
			self.C_setup_assortment = np.array([
				sum(
					self.C_setup[i, j, 0] for i, j in enumerate(
						self.convert_int_to_policy_assortment_specific(assortment_index)
						) if j >= 0
					) for assortment_index in self.policy_assortments
				]) #sum defaults to 0 for empty iterable
			self.C_policy_assortment = np.array([
				sum(
					self.C_policy[i, j, 0] for i, j in enumerate(
						self.convert_int_to_policy_assortment_specific(assortment_index)
						) if j >= 0
					) for assortment_index in self.policy_assortments
				]) #sum defaults to 0 for empty iterable
			self.C_switch_assortment = np.array([
				sum(
					self.C_switch[i, j, 0] for i, j in enumerate(
						self.convert_int_to_policy_assortment_specific(assortment_index)
						) if j >= 0
					) for assortment_index in self.policy_assortments
				]) #sum defaults to 0 for empty iterable
			self.P_assortment = np.array([
				pyo.prod(
					self.P_param[i, j, 0] for i, j in enumerate(
						self.convert_int_to_policy_assortment_specific(assortment_index)
						) if j >= 0
					) for assortment_index in self.policy_assortments
				]) #prod defaults to 1 for empty iterable
		self.keeper_assortments = None

	def pruned(self):
		print(
			f"PRUNING: There are {np.prod([policy.n+1 for policy in self]):,} policy assortments and {(self.n +1)**self.m:,} policy assortment indices"
			)
		self.ensure_assortment_params(time_varying_costs_and_probabilities=False)
		if self.time_varying_costs_and_probabilities:
			keepers = is_pareto_efficient(
				np.stack([
					self.C_setup_assortment[:, 0],
					self.C_policy_assortment[:, 0],
					self.C_switch_assortment[:, 0],
					self.P_assortment[:, 0],
					],
							axis=-1)
				)
			self.keeper_assortments = keepers
			policies = [
				Policy(
					C=[self.C_policy_assortment[assortment_index, 0]],
					P=[self.P_assortment[assortment_index, 0]],
					C_setup=[self.C_setup_assortment[assortment_index, 0]],
					C_switch=[self.C_switch_assortment[assortment_index, 0]],
					T=self.T,
					name='+'.join(
						f'{self[policy_index].name}({j+1}/{self[policy_index].n})'
						for policy_index, j in enumerate(
							self.convert_int_to_policy_assortment_specific(assortment_index)
							)
						if j > -1
						)
					)
				for assortment_index in keepers
				]
			# name='+'.join(self[policy_index].name for policy_index in self.convert_int_to_policy_assortment_specific(assortment_index))  )
		else:
			keepers = is_pareto_efficient(
				np.stack([
					self.C_setup_assortment,
					self.C_policy_assortment,
					self.C_switch_assortment,
					self.P_assortment,
					],
							axis=-1)
				)
			self.keeper_assortments = keepers
			# for assortment_index in keepers:
			# 	for policy_index in self.convert_int_to_policy_assortment_specific(assortment_index):
			# 		print(policy_index)
			policies = [
				Policy(
					C=[self.C_policy_assortment[assortment_index]],
					P=[self.P_assortment[assortment_index]],
					C_setup=[self.C_setup_assortment[assortment_index]],
					C_switch=[self.C_switch_assortment[assortment_index]],
					T=self.T,
					name='+'.join(
						f'{self[policy_index].name}({j+1}/{self[policy_index].n})'
						for policy_index, j in enumerate(
							self.convert_int_to_policy_assortment_specific(assortment_index)
							)
						if j > -1
						),
					assortment_composition=self.convert_int_to_policy_assortment_specific(
						assortment_index
						),
					emptyname_is_empty=True
					)
				for assortment_index in keepers
				]

		pruned_policy_list = PolicyList(*policies, originList=self)
		pruned_policy_list.is_pruned = True
		print(f"Returning {len(pruned_policy_list)} pruned policy assortments")
		return pruned_policy_list

	def assortment_df(self):
		print(
			f"GETTING ALL POLICY ASSORTMENTS: There are {np.prod([policy.n+1 for policy in self]):,} policy assortments and {(self.n +1)**self.m:,} policy assortment indices"
			)
		self.ensure_assortment_params(time_varying_costs_and_probabilities=False)
		if self.time_varying_costs_and_probabilities:
			df = pd.DataFrame(
				dict(
					C_policy=self.C_policy_assortment[:, 0],
					C_setup=self.C_setup_assortment[:, 0],
					C_switch=self.C_switch_assortment[:, 0],
					P=self.P_assortment[:, 0]
					)
				)
		else:
			df = pd.DataFrame(
				dict(
					C_policy=self.C_policy_assortment,
					C_setup=self.C_setup_assortment,
					C_switch=self.C_switch_assortment,
					P=self.P_assortment
					)
				)
		df['efficient'] = False
		if hasattr(self, 'keeper_assortments') and self.keeper_assortments is not None:
			df.loc[self.keeper_assortments, 'efficient'] = True
		return df

	def new_T(self, new_T_value):
		policies = [
			Policy(
				C=policy.C_,
				P=policy.P_,
				C_setup=policy.C_setup_,
				C_switch=policy.C_switch_,
				T=new_T_value,
				name=policy.name,
				emptyname_is_empty=policy.emptyname_is_empty,
				assortment_composition=policy.assortment_composition
				) for policy in self
			]
		newPolicyList = PolicyList(*policies, originList=self.originList, printing=False)
		newPolicyList.is_pruned = self.is_pruned
		# newPolicyList.assortment_compositions = self.assortment_compositions
		return newPolicyList

	def with_attributes_set(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)
		return self

	def summary(self):
		return ", ".join([f"{policy.name} (0-{len(policy.C_)})" for policy in self])

	def get_policy_by_name(self, name):
		for policy in self:
			if policy.name == name:
				return policy
		print(f"No policy with name '{name}' in PolicyList!")
		raise KeyError(name)

	def __repr__(self):
		return f'\n\n{len(self)} Policies:\n' + '\n'.join([
			f"{i}. {policy.reprself()}" for i, policy in enumerate(self)
			]) + '\n\n'
