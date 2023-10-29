import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from model import get_objective_cumulative_TT_periods, get_cumulative_disease_cost, get_cumulative_policy_cost
from model_bilinear import get_objective_cumulative_TT_periods_bilinear, get_cumulative_disease_cost_bilinear, get_cumulative_policy_cost_bilinear, get_cumulative_policy_cost_one_policy_per_period_bilinear

from model_gurobi_capable import get_cumulative_disease_cost_gurobi_capable, get_cumulative_policy_cost_gurobi_capable, get_objective_cumulative_TT_periods_gurobi_capable, get_SIRD_model_gurobi_capable

# https://stackoverflow.com/questions/55427836/how-to-render-a-latex-matrix-by-using-matplotlib
# mpl.rcParams['font.size'] = 20
# mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


# rcParams['font.monospace'] = ['Tahoma', 'DejaVu Sans',
#                                'Lucida Grande', 'Verdana']
def to_bmatrix(array2d):
	# mat = "\\\\".join(" & ".join(f"{el}" for el in row) for row in array2d)
	# matstring = f"$\\begin{{matrix}}{mat}\\end{{matrix}}$"
	# return r' {} '.format(matstring)
	return str(array2d).replace("\n", "")


def get_results_df(model, vaccination=False):
	populations = ["S", "I", "R", "D", "d", "P"]
	if vaccination:
		populations.extend(['v', 'S_antivax', 'S_unvax', 'S_vax', 'I_vax', 'I_novax'])
	df = pd.DataFrame({
		k: [(model.individual_multiplier if k != "P" else 1) * (
			getattr(model, k)[t].value
			if hasattr(getattr(model, k)[t], 'value') else getattr(model, k)[t]
			) for t in model.t] for k in populations if hasattr(model, k)
		})
	df.insert(0, "t", list(model.t))
	df.set_index('t')
	df['tot'] = df['S'] + df['I'] + df['R'] + df['D']
	if hasattr(model, 'y'):
		if model.allow_multiple_policies_per_period:
			if model.bilinear:
				y = np.full((model.m, model.n, model.T), 0)
				for assortment_index in model.policy_assortments:
					for t in model.t:
						if model.y[assortment_index, t].value == 1:
							policy_assortment = model.convert_int_to_policy_assortment_specific(
								assortment_index
								)
							for i, j in enumerate(policy_assortment):
								if j >= 0:
									y[i, j, t] = 1
			else:
				y = np.full((model.m, model.n, model.T), 0, dtype=float)
				for (i, j, t), val in model.y.extract_values().items():
					y[i, j, t] = model.y[i, j, t].value
		else:
			# or use if model.policies.is_pruned
			if model.policies[0].assortment_composition is not None:
				y = np.full((model.policies.originList.m, model.policies.originList.n, model.T),
							0,
							dtype=float)
				ydim = model.y.dim()
				if ydim == 2:
					for i in model.i:
						for t in model.t:
							if model.y[i, t].value == 1:
								for ii, jj in enumerate(model.policies[i].assortment_composition):
									if jj > -1:
										y[ii, jj, t] = 1
				elif ydim == 3:
					for i in model.i:
						for t in model.t:
							if model.y[i, 0, t].value == 1:
								for ii, jj in enumerate(model.policies[i].assortment_composition):
									if jj > -1:
										y[ii, jj, t] = 1
			else:
				y = np.full((model.m, model.n, model.T), 0, dtype=float)
				for (i, j, t), val in model.y.extract_values().items():
					y[i, j, t] = model.y[i, j, t].value

		print(f"Used any policy: {any(model.y.extract_values().values())}")
		for i in range(y.shape[0]):
			df[f'y_{i}'] = -1
			for j in range(y.shape[1]):
				df[f'y_{i}_{j}'] = [
					(y[i, j, t].value if hasattr(y[i, j, t], 'value') else y[i, j, t])
					for t in model.t
					]
				for t in model.t:
					if (y[i, j, t].value if hasattr(y[i, j, t], 'value') else y[i, j, t]) == 1:
						df.loc[t, f'y_{i}'] = j

		df['policy'] = [
			' '.join(
				f"({i+1},{j+1})"
				for i in range(y.shape[0])
				for j in range(y.shape[1])
				if df.loc[t, f'y_{i}_{j}'] == 1
				)
			for t in model.t
			]
		df['policy'] = ['Policy: ' + policy if policy else 'Do nothing.' for policy in df['policy']]

		policy_changes = [df['policy'][0]]
		policy_change_periods = [0]
		for t in model.t:
			if df['policy'][t] != policy_changes[-1]:
				policy_changes.append(df['policy'][t])
				policy_change_periods.append(t)
		# remove "0-0" policy period because there is always no policy in period 0
		if len(policy_change_periods) > 1 and policy_change_periods[1] == 1:
			policy_change_periods.pop(0)

		df['policy_change'] = 0
		df.loc[policy_change_periods, 'policy_change'] = 1

	if model.bilinear:
		cumulative_cost_function = get_objective_cumulative_TT_periods_bilinear
		cumulative_disease_cost_function = get_cumulative_disease_cost_bilinear
		if model.allow_multiple_policies_per_period:
			cumulative_policy_cost_function = get_cumulative_policy_cost_bilinear
		else:
			cumulative_policy_cost_function = get_cumulative_policy_cost_one_policy_per_period_bilinear
	else:
		if hasattr(model, 'inside_log_policy_prob'):
			cumulative_cost_function = get_objective_cumulative_TT_periods_gurobi_capable
			cumulative_disease_cost_function = get_cumulative_disease_cost_gurobi_capable
			cumulative_policy_cost_function = get_cumulative_policy_cost_gurobi_capable
		else:
			cumulative_cost_function = get_objective_cumulative_TT_periods
			cumulative_disease_cost_function = get_cumulative_disease_cost
			cumulative_policy_cost_function = get_cumulative_policy_cost
	df['CC'] = [
		model.cost_multiplier * pyo.value(cumulative_cost_function(model, TT))
		for TT in range(1, model.T + 1)
		]

	df['CC_disease'] = [
		model.cost_multiplier * pyo.value(cumulative_disease_cost_function(model, TT))
		for TT in range(1, model.T + 1)
		]

	if model.force_no_policy:
		df['CC_policy'] = 0
	else:
		df['CC_policy'] = [
			model.cost_multiplier * pyo.value(cumulative_policy_cost_function(model, TT))
			for TT in range(1, model.T + 1)
			]

	for k1, k2 in {'C_period': 'CC', 'C_policy_period': 'CC_policy',
					'C_disease_period': 'CC_disease'}.items():
		df[k1] = 0
		df.loc[0, k1] = df.loc[0, k2]
		for t in range(1, df.shape[0]):
			df.loc[t, k1] = df.loc[t, k2] - df.loc[t - 1, k2]

	return df


def plot_results(
	sol,
	sol_no_policy,
	param_kwargs,
	solver_params,
	model_kwargs,
	policies,
	plot_cost_scale=100,
	figsize=(35, 32),
	height_ratios=[2, 1],
	tight_layout_padding=3,
	hspace=0.8,
	plotting_costs=True,
	one_table_row=False,
	table_fontsize=8,
	auto_set_fontsize=False,
	extra='',
	extra_extra=None,
	title_extra=None,
	last_row_fontsize=None,
	first_col_fontsize=None,
	first_row_fontsize=None,
	xticks_at_policy_changes=False,
	legend_fontsize=8,
	title_fontsize=14,
	figsize_scale_factor=1,
	linewidth=1,
	markersize=1,
	extra_objVal=(),
	zero_switching_costs=False,
	vaccination=False,
	):
	# fig,axes = plt.subplots(2,1,figsize=(18,14)) #(width, height)
	solver = solver_params['solver']
	# fig,axes = plt.subplots(2,1,figsize=figsize) #(width, height)
	# ax1=axes[0]
	# ax2=axes[1]

	gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=height_ratios)
	fig = plt.figure(figsize=[figdim * figsize_scale_factor for figdim in figsize]) #(width, height)
	ax1 = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])

	fig.subplots_adjust(hspace=hspace)
	# for ax in axes.flatten():
	ax1.set_aspect('auto')
	ax2.set_aspect('auto')
	add_lines(
		sol,
		sol_no_policy,
		param_kwargs,
		solver_params,
		model_kwargs,
		policies,
		ax1,
		plotting_costs=plotting_costs,
		title_extra=title_extra,
		xticks_at_policy_changes=xticks_at_policy_changes,
		legend_fontsize=legend_fontsize,
		title_fontsize=title_fontsize,
		linewidth=linewidth,
		markersize=markersize,
		extra_objVal=extra_objVal,
		zero_switching_costs=zero_switching_costs,
		vaccination=vaccination,
		)
	add_table(
		sol,
		policies,
		ax2,
		plot_cost_scale=plot_cost_scale,
		one_table_row=one_table_row,
		table_fontsize=table_fontsize,
		auto_set_fontsize=auto_set_fontsize,
		last_row_fontsize=last_row_fontsize,
		first_col_fontsize=first_col_fontsize,
		first_row_fontsize=first_row_fontsize
		)
	# fig.subplots_adjust(hspace=0.5)
	if tight_layout_padding:
		plt.tight_layout(pad=tight_layout_padding)
	# fig.subplots_adjust(hspace=0.5)
	plt.savefig(
		f"figures/system_state_vs_time_T{sol.df['t'].max()+1}_{solver}{f'_{extra}' if extra else ''}{f'_{extra_extra}' if extra_extra else ''}.pdf"
		)
	# return table,table_d, nRows, nCols


def add_lines(
	sol,
	sol_no_policy,
	param_kwargs,
	solver_params,
	model_kwargs,
	policies,
	ax1,
	plotting_costs=True,
	title_extra=None,
	xticks_at_policy_changes=False,
	legend_fontsize=8,
	title_fontsize=14,
	linewidth=1,
	markersize=1,
	extra_objVal=(),
	zero_switching_costs=False,
	vaccination=False,
	):
	df = sol.df
	objVal = sol.objVal
	df_no_policy = sol_no_policy.df #get_results_df(sol_no_policy.model, vaccination=vaccination) #sometimes vaccination columns oddly not appearing...
	objVal_no_policy = sol_no_policy.objVal

	solver = solver_params['solver']
	m, n, T = policies.m, policies.n, policies.T
	# all_markers = [',',  'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X','.',]
	populations = {'Infected': "I", 'Cumulative Deaths': "D", "Recovered": "R", "Susceptible": "S"}

	if vaccination:
		# model = sol.model
		# model_no_policy = sol_no_policy.df
		# df['V'] = [sum(model.v[tt].value for tt in range(t + 1)) for t in model.t]
		# df_no_policy['V'] = [
		# 	sum(model_no_policy.v[tt].value for tt in range(t + 1)) for t in model_no_policy.t
		# 	]
		df['V'] = df.v.cumsum()
		df_no_policy['V'] = df_no_policy.v.cumsum()
		populations['Vaccinated'] = 'V'
	# breakpoint()
	for population in populations:
		# breakpoint()
		print(f"Plotting {population}")
		sns.lineplot(
			data=df_no_policy,
			x='t',
			y=populations[population],
			label=f"{population} - No intervention",
			ax=ax1,
			legend=False,
			linewidth=linewidth
			) #,palette=[rgb],
		sns.lineplot(
			data=df,
			x='t',
			y=populations[population],
			label=population,
			ax=ax1,
			legend=False,
			linewidth=linewidth
			) #,palette=[rgb],
		lines = {
			line.get_label(): line for line in ax1.lines
			} # ax1.lines gets shuffled each time, have to find correct line
		line_color = lines[population].get_color() #.lstrip("#")
		control_line = lines[f'{population} - No intervention']
		control_line.set_color(line_color)
		control_line.set_linestyle(":")

	if plotting_costs:
		cost_ax = ax1.twinx()
		cc_label = f"Cumulative Cost Total"
		cc_policy_label = f"Cumulative Costs of Interventions"

		cost_marker = "o" #"$/$"
		policy_cost_marker = "^" #$\\backslash$"
		plot_cost_multiplier = param_kwargs['cost_multiplier'] #can be anything
		sns.lineplot(
			x=df['t'],
			y=df['CC_policy'] / plot_cost_multiplier,
			label=cc_policy_label,
			ax=cost_ax,
			legend=False
			)
		sns.lineplot(
			x=df['t'],
			y=df['CC'] / plot_cost_multiplier,
			label=cc_label,
			ax=cost_ax,
			legend=False,
			linewidth=linewidth
			)
		sns.lineplot(
			x=df_no_policy['t'],
			y=df_no_policy['CC'] / plot_cost_multiplier,
			label=f"{cc_label} - No intervention",
			ax=cost_ax,
			legend=False,
			linewidth=linewidth
			) #,palette=[rgb],

		# https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Colors/Color_picker_tool
		# cost_color = (38/256,140/256,32/256, 1)
		cost_color = lambda aalpha: (38 / 256, 140 / 256, 32 / 256, aalpha)
		cost_line_alpha = 0.55
		cost_ax.yaxis.label.set_color(cost_color(1))
		cost_ax.tick_params(axis='y', colors=cost_color(1))
		# cost_ax.plot(df['t'],df['CC']/plot_cost_multiplier,label=cc_label,marker="x",color="black",markersize=5,zorder=100)
		# cost_ax.plot(df_no_policy['t'],df_no_policy['CC']/plot_cost_multiplier,label=f"{cc_label} - No intervention",marker="x",color="black",markersize=5,zorder=100)

		xcoords = (df['t'].iloc[-1], df['t'].iloc[-1], df_no_policy['t'].iloc[-1])
		ycoords = (
			df['CC_policy'].iloc[-1] / plot_cost_multiplier, df['CC'].iloc[-1] /
			plot_cost_multiplier, df_no_policy['CC'].iloc[-1] / plot_cost_multiplier
			)
		texts = (
			f"$\${int(df['CC_policy'].iloc[-1]/plot_cost_multiplier):,}\\cdot 10^{{{int(np.log10(plot_cost_multiplier))}}}$",
			f"$\${int(df['CC'].iloc[-1]/plot_cost_multiplier):,}\\cdot 10^{{{int(np.log10(plot_cost_multiplier))}}}$",
			f"$\${int(df_no_policy['CC'].iloc[-1]/plot_cost_multiplier):,}\\cdot 10^{{{int(np.log10(plot_cost_multiplier))}}}$",
			)
		for xx, yy, ss in zip(xcoords, ycoords, texts):
			cost_ax.text(xx, yy, ss, fontdict=dict(fontsize=8, color=cost_color(1)))

		lines = {line.get_label(): line for line in cost_ax.lines}
		for k, line in lines.items():
			line.set_marker(cost_marker)
			line.set_markersize(5 * markersize)
			line.set_color(cost_color(cost_line_alpha))
			line.set_markeredgecolor(cost_color(1))
			line.set_markerfacecolor(cost_color(0))
			# line.set_alpha(cost_line_alpha)
		# line_color = lines[cc_label].get_color()
		# lines[f'{cc_label} - No intervention'].set_color(line_color)
		lines[f'{cc_label} - No intervention'].set_linestyle(":")
		lines[cc_policy_label].set_marker(policy_cost_marker)

		cost_ax.set_ylabel(f"Cumulative Cost (in $\$10^{{{int(np.log10(plot_cost_multiplier))}}}$)")
		cost_ax.yaxis.set_major_formatter('${x:,.0f}')
		# cost_ax.yaxis.set_tick_params(which='major',labelleft=False, labelright=True)

		handles, labels = ax1.get_legend_handles_labels()
		unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
		handles, labels = cost_ax.get_legend_handles_labels()
		unique = unique + [(h, l)
							for i, (h, l) in enumerate(zip(handles, labels))
							if l not in labels[:i]]
		cost_ax.legend(*zip(*unique), fontsize=legend_fontsize, loc='upper left')
		# ax1.legend(fontsize=8)
	else:
		ax1.legend(fontsize=legend_fontsize, loc='upper left')

	policy_change_periods = np.flatnonzero(df['policy_change'].values)
	policy_changes = df['policy'][policy_change_periods].values
	ylim = ax1.get_ylim()[1]
	for policy_change in policy_change_periods:
		if policy_change != 0:
			ax1.axvline(policy_change, 0, ylim, linestyle=":", alpha=0.4, zorder=-1)
	current_xticks = list(ax1.get_xticks())
	if xticks_at_policy_changes:
		for period in policy_change_periods:
			buffersize = len(str(period)) - 1
			add_tick = True
			for buffer in range(-buffersize, buffersize + 1):
				if period - buffer in current_xticks:
					add_tick = False
			if add_tick:
				current_xticks.append(period)
		current_xticks = sorted(current_xticks)
		ax1.set_xticks(current_xticks)
	ax1.set_xlim(left=0, right=T + 1)

	title_rows = []
	title_rows.append(
		f"Objective: \$${int(objVal*param_kwargs['cost_multiplier']):,}$; without intervention: \$${int(objVal_no_policy*param_kwargs['cost_multiplier']):,}$ (Desired optimality gap: ${solver_params['optGap']*100:.0f}\%$; actual: ${100*(sol.ub-sol.lb)/sol.ub:.0f}\%$. Lower Bound: \${int(sol.lb)*param_kwargs['cost_multiplier']:,}. Time to solve: {int(sol.timeToSolve)}s)"
		)

	# title2= f"$C$: {to_bmatrix(policies.C_policy[:,:,0])},$P$: {to_bmatrix(policies.P_param[:,:,0])},$(C^I,C^D,C^{{setup}})=(\${param_kwargs['c_infected']:,},\${param_kwargs['c_death']:,},\${policies.C_setup[0,0,0]:,})$"
	costRow = f"$C^I=\${param_kwargs['c_infected']:,},C^D=\${param_kwargs['c_death']:,}$"
	if zero_switching_costs:
		costRow = costRow + ". Zero switching costs."
	title_rows.append(costRow)

	title_rows.append(
		f"One Period={param_kwargs['days_per_period']} days (costs scaled by ${param_kwargs['cost_multiplier']:,}$ during optimization)"
		)

	if title_extra:
		if extra_objVal:
			title_extra = title_extra + f". {extra_objVal[0]}: \$${int(extra_objVal[1]*param_kwargs['cost_multiplier']):,}$"
		title_rows.append(title_extra)
	titletext = "\n".join(title_rows)
	ax1.set_title(titletext, fontsize=title_fontsize)
	ax1.set_ylabel("Number Individuals")
	ax1.set_xlabel("Period")


def add_table(
	sol,
	policies,
	ax2,
	plot_cost_scale=100,
	one_table_row=False,
	table_fontsize=8,
	auto_set_fontsize=False,
	last_row_fontsize=None,
	first_col_fontsize=None,
	first_row_fontsize=None
	):
	df = sol.df

	if not sol.model.allow_multiple_policies_per_period:
		policies = sol.model.policies.originList

	m, n, T = policies.m, policies.n, policies.T

	policy_change_periods = np.flatnonzero(df['policy_change'].values)
	policy_changes = df['policy'][policy_change_periods].values

	nChanges = len(policy_changes)

	ax2.axis('off')
	nRows = m + 1
	nCols = nChanges
	cols = [
		f"{policy_change_periods[t]}\n-{policy_change_periods[t+1]-1}"
		if t != nChanges - 1 else f"{policy_change_periods[t]}\n-{T-1}" for t in range(nChanges)
		]
	# row_labels = [f"Level of Policy {i+1}" for i in range(m)]
	chars_per_entry = 5
	row_labels = [
		"\n".join((
			f"{i}. {policies[i].name}",
			f"A: \$[{','.join([f'{int(policies[i].C_setup[j,0]/plot_cost_scale)}'.ljust(chars_per_entry) for j in range(policies[i].C_setup.shape[0]) if policies[i].C_setup[j,0]<np.inf])}]$\\cdot 10^{{{int(np.log10(plot_cost_scale))}}}$",
			f"B: \$[{','.join([f'{int(policies[i].C_switch[j,0]/plot_cost_scale)}'.ljust(chars_per_entry) for j in range(policies[i].C_switch.shape[0]) if policies[i].C_switch[j,0]<np.inf])}]$\\cdot 10^{{{int(np.log10(plot_cost_scale))}}}$",
			# f"$C^{{setup}}$: {[policies[i].C_setup[j,0] for j in range(policies[i].C_setup.shape[0]) if policies[i].C_setup[j,0]<np.inf]}",
			# f"$C^{{setup}}$: {policies[i].C_setup[0,0]}",
			f"C: \$[{','.join([f'{int(policies[i].C[j,0]/plot_cost_scale)}'.ljust(chars_per_entry) for j in range(policies[i].C.shape[0]) if policies[i].C[j,0]<np.inf])}]$\\cdot 10^{{{int(np.log10(plot_cost_scale))}}}$",
			f"P:  [{','.join([f'{policies[i].P[j,0]:.2}'.lstrip('0').ljust(chars_per_entry) for j in range(policies[i].P.shape[0]) if policies[i].P[j,0]<1])}]",
			)) for i in range(m)
		]

	content = [[
		str(df[f'y_{i}'][t] + 1) if df[f'y_{i}'][t] != -1 else '' for t in policy_change_periods
		] for i in range(m)]

	interval_value_keys = {
		'C_period': "Cost Per Period: TOTAL",
		'C_policy_period': "Cost Per Period: POLICY",
		'C_disease_period': "Cost Per Period: DISEASE",
		"P": "Probability Factor"
		}
	row_labels += ["\n".join(interval_value_keys.values())]

	interval_values = {
		k: [
			sum(df[k][tt] for tt in range(t, policy_change_periods[t_index + 1])) /
			(policy_change_periods[t_index + 1] - t) if t_index != nChanges - 1 else sum(
				df[k][tt] for tt in range(t, df.shape[0])
				) / (df.shape[0] - t) for t_index, t in enumerate(policy_change_periods)
			] for k in interval_value_keys.keys() if k != 'P'
		}

	interval_values['P'] = [df.P[t] for t in policy_change_periods]
	content += [[
		"\n".join(
			f'${thing:,.2}'
			for thing in (interval_values[k][t_index]
							for k in interval_values
							if k != "P")
			) + f'\n{interval_values["P"][t_index]:.3f}'
		for t_index, t in enumerate(policy_change_periods)
		]]

	# C_period
	# C_policy_period
	# C_disease_period

	plt.subplots_adjust(bottom=0.3)
	table = ax2.table(
		cellText=content,
		# rowColours=colors,
		rowLabels=row_labels,
		colLabels=cols,
		loc='center',
		cellLoc='left'
		)
	table_d = table.get_celld()

	last_row = max([i for (i, j) in table_d])
	first_col = min([j for (i, j) in table_d])
	first_row = min([i for (i, j) in table_d])
	table.auto_set_font_size(auto_set_fontsize)
	if not auto_set_fontsize:
		table.set_fontsize(table_fontsize)
		for (i, j) in table_d:
			if i == last_row and last_row_fontsize is not None:
				table_d[i, j].set_fontsize(last_row_fontsize)
			if i == first_row and first_row_fontsize is not None:
				table_d[i, j].set_fontsize(first_row_fontsize)
			if j == first_col and first_col_fontsize is not None:
				table_d[i, j].set_fontsize(first_col_fontsize)

	# min_width = table_d[1,0].get_width()/2
	min_width = .02
	max_width = table_d[1, 0].get_width() * 6
	base_w = (table.get_celld()[1, 0].get_width()) * nChanges / T
	for j in range(nCols):
		if j < len(policy_change_periods) - 1:
			col_mult = policy_change_periods[j + 1] - policy_change_periods[j]
		else:
			col_mult = T - 1 - policy_change_periods[j]
		for i in range(nRows + 1):
			table_d[i, j].set_width(min(max(min_width, base_w * col_mult), max_width))
	for i in range(nRows + 1):
		maxRows = max((table[i, j].get_text().get_text().count("\n") + 1)
						for j in range(-1, nCols)
						if (i, j) in table_d)
		for j in range(-1, nCols):
			if (i, j) in table_d:
				table_d[i, j].set_height(table_d[i, j].get_height() * maxRows)
				table_d[i, j].set_text_props(fontfamily='monospace')
				table_d[i, j].set_text_props(fontweight="bold")


class Solution:
	def __init__(
		self,
		objVal,
		timeToSolve,
		model,
		sol,
		otherModel=None,
		solver_params={},
		extra_metadata={},
		get_DF=True,
		solution_path=[],
		description='',
		vaccination=False,
		):
		self.objVal = objVal
		self.timeToSolve = timeToSolve
		self.model = model
		self.sol = sol
		try:
			self.lb = sol['Problem'][0]['Lower bound']
			self.ub = sol['Problem'][0]['Upper bound']
			self.status = sol['Solver'][0]['Status']
		except Exception as e:
			print("Error construction Solution")
			print(e)
			self.lb = 1
			self.ub = 1
			self.status = None
		if get_DF:
			self.df = get_results_df(self.model, vaccination=vaccination)
		else:
			self.df = None
		self.otherModel = otherModel
		self.extra_metadata = extra_metadata
		self.solver_params = solver_params
		try:
			self.termination_condition = sol.Solver.termination_condition.value
			self.timed_out = (sol.Solver.termination_condition.value == 'maxTimeLimit')
			# sol.Problem.values() # these seem to be lazily loaded...
			# sol.Solution.values() # these seem to be lazily loaded...
			# sol.Solver.values() # these seem to be lazily loaded...
			# print(f"Problem=\n{sol.Problem.__repr__()}")
			# print(f"Solution=\n{sol.Solution.__repr__()}")
			# print(f"Solver=\n{sol.Solver.__repr__()}")
		except:
			self.termination_condition = None
			self.timed_out = False
		self.multipliers = None
		self.solution_path = solution_path
		self.description = description

	def without_model(self):
		self.model = None
		return self

	def to_dict(self, include_model_stuff=False, description=False, cumulative_stats=False):
		if include_model_stuff:
			self.extra_metadata = {
				**dict(
					m=self.model.m,
					n=self.model.n,
					T=self.model.T,
					nVariables=self.model.nvariables(),
					nConstraints=self.model.nconstraints(),
					),
				**self.extra_metadata,
				}
			self.extra_metadata['nPolicies'] = self.model.policies.num_assortments
			# try:
			# 	self.extra_metadata['nPolicies'] = self.model.policies.num_assortments
			# except:
			# 	print("FAILED TO GET 'num_assortments'!")
		if cumulative_stats:
			if (not hasattr(self, 'df')) or (self.df is None):
				self.df = get_results_df(self.model)
			colmap = dict(
				cumulative_cost='CC',
				cumulative_policy_cost='CC_policy',
				cumulative_disease_cost='CC_disease',
				cumulative_deaths='D',
				cumulative_recovered='R'
				)
			self.extra_metadata = {
				**{k: self.df[v].values[-1] for k, v in colmap.items()},
				**self.extra_metadata
				}
		if description:
			self.extra_metadata['description'] = self.description
		return dict(
			objVal=self.objVal,
			timeToSolve=self.timeToSolve,
			LB=self.lb,
			UB=self.ub,
			status=str(self.status),
			**self.solver_params,
			**self.extra_metadata
			)

	def drop_model(self):
		self.model = None


def plot_time_trials(
	trial_sols,
	param_kwargs,
	solver_params,
	model_kwargs,
	policies,
	extra='',
	extra_extra=None,
	policy_descr=None
	):
	optGap = solver_params['optGap']
	df = pd.DataFrame(
		dict(
			T=trial_sols.keys(),
			timeToSolve=[sol.timeToSolve for sol in trial_sols.values()],
			nChanges=[sol.df['policy_change'].sum() for sol in trial_sols.values()]
			)
		)
	sns.lineplot(x='T', y='timeToSolve', data=df)
	sns.scatterplot(x='T', y='timeToSolve', size='nChanges', data=df)
	# plt.show(block=False)
	title = f"Time to Solve to ${optGap*100:.1f}\%$ Optimality"
	if policy_descr:
		title += f"\n{policy_descr}"
	plt.title(title, fontsize=12)
	solver = solver_params['solver']
	plt.savefig(f"figures/time_to_solve_{solver}_{optGap}{extra}.pdf")
