import numpy as np
import casadi


class Problem_Maker(casadi.Opti):


	def __init__(self):
		super().__init__()
		self.discrete = [] #stores variables in order of creation. Req'd for Bonmin
		self.discrete_ = {} #stores variables by casadi.Opti.variable.name()

	def get_variables_mat(self,dims,disc):
		arr = np.empty(dims,dtype=casadi.Opti)
		for index,_ in np.ndenumerate(arr):
			arr[index] = self.variable()
			self.discrete_[arr[index].name()] = disc
		self.discrete.extend([disc for i in range(np.prod(arr.shape))])
		return arr

	def extract_values(self,solution,var_matrix):
		vals = np.empty(var_matrix.shape)
		for index,_ in np.ndenumerate(var_matrix):
			vals[index] = solution.value(var_matrix[index])
			if self.discrete_[var_matrix[index].name()]:
				vals[index] = np.round(vals[index]).astype(int)
		return vals

	def set_solver(self):
		# Casadi plugin options
		# https://web.casadi.org/python-api/
		p_options = {"discrete":self.discrete }
		# ,"expand":True
		# Solver options (IPOPT/BONMIN)
		# https://www.coin-or.org/Bonmin/option_pages/options_list_bonmin.html
		# https://coin-or.github.io/Ipopt/OPTIONS.html
		# s_options = {'file_print_level' : 9,'output_file' : 'output.txt', 'expect_infeasible_problem' : 'no' , 'print_user_options' : 'yes'}
		s_options = {'expect_infeasible_problem' : 'no'}
		# ,'print_options_documentation' : 'yes'
		# , 'print_advanced_options' : 'yes',
		# 'print_options_documentation' : True
		# s_options = {"max_iter": 1000000000,'tol': .00000001}
		# s_options = {'iteration_limit': 100000000, 'time_limit': 100000000000}
		self.solver('bonmin',p_options,s_options)
