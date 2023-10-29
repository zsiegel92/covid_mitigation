import pandas as pd
import numpy as np
import jinja2

from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from tabulation_utils import prioritize_value_one, mask_indices, replace_all, limit_to_trials_with_only_one_parameter_varied

# from jinja2 import Environment, PackageLoader, select_autoescape

# col: col.replace("_", " ").replace("lagrangian", "lagr").replace("solution", "soln").replace(
# 	"objVal", "obj"
# 	).replace("desired", "des").replace('cumulative ', '') for col in df.columns

overleaf = "/Users/zach/UCLA Anderson Dropbox/Zach Siegel/UCLA_classes/research/covid_mitigation/writing/covid_mitigation"

colname_replacements = [("lagrangian_objVal", "lagrangian_heuristic_objVal"), ("_", " "),
						("lagrangian", "lagr"), ("solution", "soln"), ("objVal", "obj"),
						("desired", "des"), ('cumulative ', ''), ("trial ", "")]
# colname_replacements = [("_", " "), ("lagrangian", "lagr"), ("solution", "soln"), ("objVal", "obj"),
# ("desired", "des")]

# Tvals = (2, 5, 10, 20)
# Tvals = (20, 50, 150)
Tvals = (150, 100, 50)
# Tvals = (10, 12)
# Tvals = (10, 12, 15, 20)
# Tvals = (10, 12, 15)
# Tvals = (2, 5, 10, 20)
df = pd.read_csv(
	# "lagrangian_heuristic_comparison_time_03_11_2021_22_46_(T=20_30_50_150).csv"
	f'output/lagrangian_heuristic_comparison_(T={"_".join(str(T) for T in Tvals)}).csv'
	)
df = limit_to_trials_with_only_one_parameter_varied(df)

# df = pd.read_csv(
# 	f'output/backups/lagrangian_heuristic_comparison_(T={"_".join(str(T) for T in Tvals)}).csv'
# 	)
df.sort_values(
	by=[
		"T",
		"m",
		"n",
		# "trial_cost_multiplier",
		"trial_effect_multiplier",
		],
	inplace=True,
	# key=lambda col: col.replace(1, -1000),
	)
df.sort_values(
	by=[
		"T",
		"m",
		"n",
		"trial_cost_multiplier",
		# "trial_effect_multiplier",
		],
	inplace=True,
	key=lambda col: col.replace(1, -1000),
	)
# https://stackoverflow.com/questions/67459709/pandas-how-to-sort-values-by-2-different-columns-using-2-different-keys
# df.groupby(["T", "m", "n", "trial_cost_multiplier"]
# 			).apply(lambda x: x.sort_values("trial_effect_multiplier", ascending=True)).reset_index(
# 			level=0, drop=True
# 			)
# df = mask_indices(df, ["T", "trial_cost_multiplier", "trial_effect_multiplier"])
df.index = range(1, len(df) + 1)
df['Trial'] = df.index

indices_with_new_times = []
T = None
for i, row in df.iterrows():
	if row['T'] != T:
		T = row['T']
		indices_with_new_times.append(row['Trial'])

cumulative_cols = [col for col in df.columns if 'cumulative' in col]

# [
# 	'no_policy_cumulative_cost', 'no_policy_cumulative_policy_cost',
# 	'no_policy_cumulative_disease_cost', 'no_policy_cumulative_deaths',
# 	'no_policy_cumulative_recovered', 'solver_cumulative_cost', 'solver_cumulative_policy_cost',
# 	'solver_cumulative_disease_cost', 'solver_cumulative_deaths', 'solver_cumulative_recovered',
# 	'lagrangian_cumulative_cost', 'lagrangian_cumulative_policy_cost',
# 	'lagrangian_cumulative_disease_cost', 'lagrangian_cumulative_deaths', 'lagrangian_cumulative_recovered'
# 	]

df = df[[
	'Trial',
	'T',
	'trial_cost_multiplier',
	'trial_effect_multiplier',
	'm',
	'n',
	"no_policy_objVal",
	"solver_objVal",
	"lagrangian_objVal",
	"lagrangian_LB",
	"lagrangian_LB_guaranteed",
	"solver_LB",
	"solver_optGap",
	"lagrangian_optGap",
	"lagrangian_desired_L1_optGap",
	"lagrangian_desired_L2_optGap",
	"lagrangian_desired_optGap",
	"solver_desired_optGap",
	"lagrangian_L1_objVal",
	"lagrangian_L2_objVal",
	"solver_timeToSolve",
	"lagrangian_timeToSolve",
	"lagrangian_timeToSolve_L1_total",
	"lagrangian_timeToSolve_L2_total",
	'nConstraints',
	'nPolicies',
	'nVariables',
	'quadratic_heuristic_objVal',
	'quadratic_approx_objVal',
	'simple_index_blocksize_1_objVal',
	'simple_index_blocksize_7_objVal',
	# "lagrangian_solution_path",
	] + cumulative_cols]

df['solver_vs_lagrangian_lb_gap'] = abs((df['solver_objVal'] - df['lagrangian_LB'])) / abs(
	df['lagrangian_LB']
	)
df['lagrangian_vs_lagrangian_lb_gap'] = abs((df['lagrangian_objVal'] - df['lagrangian_LB'])) / abs(
	df['lagrangian_LB']
	)
df['solver_vs_lagrangian_lb_gap_conservative'] = (
	df['solver_objVal'] - df['lagrangian_LB_guaranteed']
	) / df['lagrangian_LB_guaranteed']
# df = pd.read_csv('lagrangian_heuristic_comparison_(T=20_30_50_150_200).csv')

for col in (
	'T',
	"no_policy_objVal",
	"solver_objVal",
	"lagrangian_objVal",
	"lagrangian_LB",
	"lagrangian_LB_guaranteed",
	"solver_LB",
	# "solver_optGap",
	# "lagrangian_optGap",
	# "lagrangian_desired_L1_optGap",
	# "lagrangian_desired_L2_optGap",
	# "lagrangian_desired_optGap",
	# "solver_desired_optGap",
	"lagrangian_L1_objVal",
	"lagrangian_L2_objVal"
	"solver_timeToSolve",
	"lagrangian_timeToSolve",
	"lagrangian_timeToSolve_L1_total",
	"lagrangian_timeToSolve_L2_total",
	'm',
	'n',
	'nConstraints',
	'nPolicies',
	'nVariables',
	"lagrangian_L1_objVal",
	"lagrangian_L2_objVal",
	'quadratic_heuristic_objVal',
	'quadratic_approx_objVal',
	'simple_index_blocksize_1_objVal',
	'simple_index_blocksize_7_objVal',
	):
	if col in df:
		try:
			df[col] = df[col].astype(int)
		except:
			print(f"COULD NOT CONVERT COLUMN {col} TO INTEGER!")

df['solver_vs_lagrangian_lb_gap'] = df['solver_vs_lagrangian_lb_gap'].map(lambda x: f"{x:.6f}")

# def format_number(x):
# 	parts =

# 	return f"{(parts:=f"{x:.2e}".replace('e+', 'e').split('e'))[0].ljust(4,' ')}e{parts[1]}"

for col in cumulative_cols:
	df[col] = df[col].map(
		lambda x: f"{(parts:=f'{x:.2e}'.replace('e+', 'e').split('e'))[0].ljust(4,' ')}e{parts[1]}"
		)
# del df['lagrangian_solution_path']
optGap_names = [
	"lagrangian_desired_L1_optGap",
	"lagrangian_desired_L2_optGap",
	"lagrangian_desired_optGap",
	"solver_desired_optGap",
	'solver_optGap',
	"lagrangian_optGap",
	]

optGaps = {optGap_name: df[optGap_name].min() for optGap_name in optGap_names}
for optGap_name in optGaps:
	assert df[optGap_name].max() == df[optGap_name].min() # these should be scalar, not trial-varying

col_mapper = {col: replace_all(col, colname_replacements) for col in df.columns}

optGap_string = "; ".join(
	f"``\\emph{{{replace_all(optGap_name, colname_replacements)}}}'' : {val:.02f}"
	for optGap_name, val in optGaps.items()
	) + "."
df0 = df[[
	'Trial',
	'T',
	'trial_cost_multiplier',
	'trial_effect_multiplier',
	'm',
	'n',
	'nConstraints',
	'nPolicies',
	'nVariables',
	]]

df1 = df[[
	'Trial',
	"no_policy_objVal",
	"solver_objVal",
	"lagrangian_objVal",
	'simple_index_blocksize_1_objVal',
	'simple_index_blocksize_7_objVal',
	'quadratic_heuristic_objVal',
	# 'quadratic_approx_objVal',
	"lagrangian_LB",
	# "lagrangian_LB_guaranteed",
	"solver_LB",
	# "solver_optGap",
	# "lagrangian_optGap",
	"solver_vs_lagrangian_lb_gap",
	"lagrangian_vs_lagrangian_lb_gap",
	# 'solver_vs_lagrangian_lb_gap_conservative',
	# "lagrangian_desired_L1_optGap",
	# "lagrangian_desired_L2_optGap",
	# "lagrangian_desired_optGap",
	# "solver_desired_optGap",
	# "lagrangian_L1_objVal",
	# "lagrangian_L2_objVal",
	]]
df2 = df[[
	'Trial',
	"solver_timeToSolve",
	"lagrangian_timeToSolve",
	"lagrangian_timeToSolve_L1_total",
	"lagrangian_timeToSolve_L2_total",
	]]

df3 = df[['Trial'] + cumulative_cols]
df3 = df3[[
	col for col in df3.columns if ('recovered' not in col) and
	(col not in ["no_policy_cumulative_policy_cost", "no_policy_cumulative_disease_cost"])
	]]
# df3 columns:
df3 = df3[[
	'Trial',
	"no_policy_cumulative_cost",
	"no_policy_cumulative_deaths",
	"solver_cumulative_cost",
	"solver_cumulative_deaths",
	"solver_cumulative_disease_cost",
	"solver_cumulative_policy_cost",
	"lagrangian_cumulative_cost",
	"lagrangian_cumulative_deaths",
	"lagrangian_cumulative_disease_cost",
	"lagrangian_cumulative_policy_cost",
	]]

for i, row in df1.iterrows():
	if row['solver_objVal'] < 100:
		for k, v in row.items():
			if 'solver' in k:
				row[k] = '--'
		df1.loc[i] = row

col_mapper = {col: replace_all(col, colname_replacements) for col in df.columns}

df = df.rename(columns=col_mapper)
df0 = df0.rename(columns=col_mapper)
df0.set_index(df0.columns.tolist(), inplace=True)
df1 = df1.rename(columns=col_mapper)
df2 = df2.rename(columns=col_mapper)
df3 = df3.rename(columns=col_mapper)

for dataframe in (df1, df2, df3):
	dataframe.set_index("Trial", inplace=True)
# print(df.to_latex(index=True))


def add_hlines_at_trials(df_rendered, trials):
	lines = df_rendered.split('\n')
	for trial in trials[1:]:
		splitlines = [line.split('&') for line in lines]
		for i, splitline in enumerate(splitlines):
			if splitline[0].strip() == str(trial):
				break
		lines.insert(i, '\\midrule')
	return '\n'.join(lines)


df0_rendered = df0.to_latex(index=True, sparsify=True)
print(df0_rendered)
one_col_format = f"p{{{0.9/(len(df1.columns)):.2}\\linewidth}}"
df1_col_format = "l" + one_col_format * (len(df1.columns))
df1_rendered = df1.to_latex(index=True, column_format=df1_col_format)
print(df1_rendered)
df2_rendered = df2.to_latex(index=True)
print(df2_rendered)
df3_n_cols = len(df3.columns)
one_col_format = f"p{{{0.7/(len(df3.columns)):.2}\\linewidth}}"
df3_col_format = "l" + "|" + one_col_format*2 + "|" + one_col_format*4 + "|" + one_col_format*4
# df3_col_format = "l" + f"p{{{0.7/(len(df3.columns)):.2}\\linewidth}}" * (len(df3.columns))
df3_rendered = df3.to_latex(
	index=True, column_format=df3_col_format, float_format=lambda s: "{:,.3g}".format(s)
	)
print(df3_rendered)

df0.to_csv('output/table_a.csv')
df1.to_csv('output/table_b.csv')
df2.to_csv('output/table_c.csv')
df3.to_csv('output/table_d.csv')

df0.to_csv(f'{overleaf}/content_to_share/table_a.csv')
df1.to_csv(f'{overleaf}/content_to_share/table_b.csv')
df2.to_csv(f'{overleaf}/content_to_share/table_c.csv')
df3.to_csv(f'{overleaf}/content_to_share/table_d.csv')

with (open("latex_trial_template.tex", "r")) as f:
	template_str = f.read()

df0_rendered = add_hlines_at_trials(df0_rendered, indices_with_new_times)
df1_rendered = add_hlines_at_trials(df1_rendered, indices_with_new_times)
df2_rendered = add_hlines_at_trials(df2_rendered, indices_with_new_times)
df3_rendered = add_hlines_at_trials(df3_rendered, indices_with_new_times)

rendered = jinja2.Template(template_str).render(
	df0=df0_rendered,
	df1=df1_rendered,
	df2=df2_rendered,
	df3=df3_rendered,
	optGap_string=optGap_string,
	)

with open(f"{overleaf}/lagrangian_table_figure.tex", "w") as f:
	f.write(rendered)


def combine_columns_add_key(df, old_col1, old_col2, new_col, key_col, key1, key2):
	value_vars = [old_col1, old_col2]
	id_vars = [col for col in df.columns if col not in value_vars]
	df_new = pd.melt(
		df, id_vars=id_vars, value_vars=value_vars, var_name=key_col, value_name=new_col
		)
	return df_new


timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
identifier0 = f'(T={"_".join(str(T) for T in Tvals)})'
identifier = f'{identifier0}_(time_{timestamp})'

plotting = False
if plotting:
	if True:
		fig, ax = plt.subplots(figsize=(10, 10))
		ax = sns.lineplot(
			data=df.loc[df['cost multiplier'] == 1],
			x='effect multiplier',
			y='solver vs lagr lb gap',
			hue='T',
			legend=True,
			ax=ax
			)
		ax.set_title(
			f"Relative Gap between Solver and Lagrangian Lower Bound vs. Effect Multiplier"
			)
		plt.savefig(
			f'figures/processing_lagrangian_results/solver_vs_lagr_lb_gap_vs_effect_multiplier_{identifier0}.pdf'
			)
		plt.savefig(
			f'figures/processing_lagrangian_results/solver_vs_lagr_lb_gap_vs_effect_multiplier_{identifier}.pdf'
			)
		plt.savefig(f"{overleaf}/figures/solver_vs_lagr_lb_gap_vs_effect_multiplier.pdf")

	if True:
		fig, ax = plt.subplots(figsize=(10, 10))
		ax = sns.lineplot(
			data=df.loc[df['effect multiplier'] == 1],
			x='cost multiplier',
			y='solver vs lagr lb gap',
			hue='T',
			legend=True,
			ax=ax
			)
		ax.set_title(f"Relative Gap between Solver and Lagrangian Lower Bound vs. Cost Multiplier")
		plt.savefig(
			f'figures/processing_lagrangian_results/solver_vs_lagr_lb_gap_vs_cost_multiplier_{identifier0}.pdf'
			)
		plt.savefig(
			f'figures/processing_lagrangian_results/solver_vs_lagr_lb_gap_vs_cost_multiplier_{identifier}.pdf'
			)
		plt.savefig(f"{overleaf}/figures/solver_vs_lagr_lb_gap_vs_cost_multiplier.pdf")

	if True:
		df2 = combine_columns_add_key(
			df, 'lagr heuristic obj', 'solver obj', 'objective', 'result type',
			'lagrangian heuristic', 'solver'
			)

		fig, ax = plt.subplots(figsize=(10, 10))
		ax = sns.lineplot(
			data=df2.loc[df2['cost multiplier'] == 1],
			x='effect multiplier',
			y='objective',
			hue='T',
			style='result type',
			legend=True,
			ax=ax
			)
		ax.set_title(f"Objective vs. Effect Multiplier")
		plt.savefig(
			f'figures/processing_lagrangian_results/objective_vs_effect_multiplier_{identifier0}.pdf'
			)
		plt.savefig(
			f'figures/processing_lagrangian_results/objective_vs_effect_multiplier_{identifier}.pdf'
			)
		plt.savefig(f"{overleaf}/figures/objective_vs_effect_multiplier.pdf")

	if True:
		df2 = combine_columns_add_key(
			df, 'lagr heuristic obj', 'solver obj', 'objective', 'result type',
			'lagrangian heuristic', 'solver'
			)

		fig, ax = plt.subplots(figsize=(10, 10))
		ax = sns.lineplot(
			data=df2.loc[df2['effect multiplier'] == 1],
			x='cost multiplier',
			y='objective',
			hue='T',
			style='result type',
			legend=True,
			ax=ax
			)
		ax.set_title(f"Objective vs. Cost Multiplier")
		plt.savefig(
			f'figures/processing_lagrangian_results/objective_vs_cost_multiplier_{identifier0}.pdf'
			)
		plt.savefig(
			f'figures/processing_lagrangian_results/objective_vs_cost_multiplier_{identifier}.pdf'
			)
		plt.savefig(f"{overleaf}/figures/objective_vs_cost_multiplier.pdf")
# def add_lines(
# 	sol,
# 	sol_no_policy,
# 	param_kwargs,
# 	solver_params,
# 	model_kwargs,
# 	policies,
# 	ax1,
# 	plotting_costs=True,
# 	title_extra=None,
# 	xticks_at_policy_changes=False,
# 	legend_fontsize=8,
# 	title_fontsize=14,
# 	):
# 	df = sol.df
# 	objVal = sol.objVal
# 	df_no_policy = sol_no_policy.df
# 	objVal_no_policy = sol_no_policy.objVal

# 	solver = solver_params['solver']
# 	m, n, T = policies.m, policies.n, policies.T
# 	# all_markers = [',',  'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X','.',]
# 	populations = {'Infected': "I", 'Cumulative Deaths': "D", "Recovered": "R", "Susceptible": "S"}
# 	for population in populations:
# 		sns.lineplot(
# 			data=df_no_policy,
# 			x='t',
# 			y=populations[population],
# 			label=f"{population} - No intervention",
# 			ax=ax1,
# 			legend=False
# 			) #,palette=[rgb],
# 		sns.lineplot(
# 			data=df, x='t', y=populations[population], label=population, ax=ax1, legend=False
# 			) #,palette=[rgb],
# 		lines = {
# 			line.get_label(): line for line in ax1.lines
# 			} # ax1.lines gets shuffled each time, have to find correct line
# 		line_color = lines[population].get_color() #.lstrip("#")
# 		control_line = lines[f'{population} - No intervention']
# 		control_line.set_color(line_color)
# 		control_line.set_linestyle(":")

# 	if plotting_costs:
# 		cost_ax = ax1.twinx()
# 		cc_label = f"Cumulative Cost Total"
# 		cc_policy_label = f"Cumulative Costs of Interventions"

# 		cost_marker = "o" #"$/$"
# 		policy_cost_marker = "^" #$\\backslash$"
# 		plot_cost_multiplier = param_kwargs['cost_multiplier'] #can be anything
# 		sns.lineplot(
# 			x=df['t'],
# 			y=df['CC_policy'] / plot_cost_multiplier,
# 			label=cc_policy_label,
# 			ax=cost_ax,
# 			legend=False
# 			)
# 		sns.lineplot(
# 			x=df['t'], y=df['CC'] / plot_cost_multiplier, label=cc_label, ax=cost_ax, legend=False
# 			)
# 		sns.lineplot(
# 			x=df_no_policy['t'],
# 			y=df_no_policy['CC'] / plot_cost_multiplier,
# 			label=f"{cc_label} - No intervention",
# 			ax=cost_ax,
# 			legend=False
# 			) #,palette=[rgb],

# 		# https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Colors/Color_picker_tool
# 		# cost_color = (38/256,140/256,32/256, 1)
# 		cost_color = lambda aalpha: (38 / 256, 140 / 256, 32 / 256, aalpha)
# 		cost_line_alpha = 0.55
# 		cost_ax.yaxis.label.set_color(cost_color(1))
# 		cost_ax.tick_params(axis='y', colors=cost_color(1))
# 		# cost_ax.plot(df['t'],df['CC']/plot_cost_multiplier,label=cc_label,marker="x",color="black",markersize=5,zorder=100)
# 		# cost_ax.plot(df_no_policy['t'],df_no_policy['CC']/plot_cost_multiplier,label=f"{cc_label} - No intervention",marker="x",color="black",markersize=5,zorder=100)

# 		xcoords = (df['t'].iloc[-1], df['t'].iloc[-1], df_no_policy['t'].iloc[-1])
# 		ycoords = (
# 			df['CC_policy'].iloc[-1] / plot_cost_multiplier, df['CC'].iloc[-1] /
# 			plot_cost_multiplier, df_no_policy['CC'].iloc[-1] / plot_cost_multiplier
# 			)
# 		texts = (
# 			f"$\${int(df['CC_policy'].iloc[-1]/plot_cost_multiplier):,}\\cdot 10^{{{int(np.log10(plot_cost_multiplier))}}}$",
# 			f"$\${int(df['CC'].iloc[-1]/plot_cost_multiplier):,}\\cdot 10^{{{int(np.log10(plot_cost_multiplier))}}}$",
# 			f"$\${int(df_no_policy['CC'].iloc[-1]/plot_cost_multiplier):,}\\cdot 10^{{{int(np.log10(plot_cost_multiplier))}}}$",
# 			)
# 		for xx, yy, ss in zip(xcoords, ycoords, texts):
# 			cost_ax.text(xx, yy, ss, fontdict=dict(fontsize=8, color=cost_color(1)))

# 		lines = {line.get_label(): line for line in cost_ax.lines}
# 		for k, line in lines.items():
# 			line.set_marker(cost_marker)
# 			line.set_markersize(5)
# 			line.set_color(cost_color(cost_line_alpha))
# 			line.set_markeredgecolor(cost_color(1))
# 			line.set_markerfacecolor(cost_color(0))
# 			# line.set_alpha(cost_line_alpha)
# 		# line_color = lines[cc_label].get_color()
# 		# lines[f'{cc_label} - No intervention'].set_color(line_color)
# 		lines[f'{cc_label} - No intervention'].set_linestyle(":")
# 		lines[cc_policy_label].set_marker(policy_cost_marker)
# 		cost_ax.set_ylabel(f"Cumulative Cost (in $\$10^{{{int(np.log10(plot_cost_multiplier))}}}$)")
# 		cost_ax.yaxis.set_major_formatter('${x:,.0f}')
# 		# cost_ax.yaxis.set_tick_params(which='major',labelleft=False, labelright=True)

# 		handles, labels = ax1.get_legend_handles_labels()
# 		unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
# 		handles, labels = cost_ax.get_legend_handles_labels()
# 		unique = unique + [(h, l)
# 							for i, (h, l) in enumerate(zip(handles, labels))
# 							if l not in labels[:i]]
# 		cost_ax.legend(*zip(*unique), fontsize=legend_fontsize, loc='upper left')
# 		# ax1.legend(fontsize=8)
# 	else:
# 		ax1.legend(fontsize=legend_fontsize, loc='upper left')

# 	policy_change_periods = np.flatnonzero(df['policy_change'].values)
# 	policy_changes = df['policy'][policy_change_periods].values
# 	ylim = ax1.get_ylim()[1]
# 	for policy_change in policy_change_periods:
# 		if policy_change != 0:
# 			ax1.axvline(policy_change, 0, ylim, linestyle=":", alpha=0.4, zorder=-1)
# 	current_xticks = list(ax1.get_xticks())
# 	if xticks_at_policy_changes:
# 		for period in policy_change_periods:
# 			buffersize = len(str(period)) - 1
# 			add_tick = True
# 			for buffer in range(-buffersize, buffersize + 1):
# 				if period - buffer in current_xticks:
# 					add_tick = False
# 			if add_tick:
# 				current_xticks.append(period)
# 		current_xticks = sorted(current_xticks)
# 		ax1.set_xticks(current_xticks)
# 	ax1.set_xlim(left=0, right=T + 1)

# 	title_rows = []
# 	title_rows.append(
# 		f"Objective: \$${int(objVal*param_kwargs['cost_multiplier']):,}$; without intervention: \$${int(objVal_no_policy*param_kwargs['cost_multiplier']):,}$ (Desired optimality gap: ${solver_params['optGap']*100:.0f}\%$; actual: ${100*(sol.ub-sol.lb)/sol.ub:.0f}\%$. Lower Bound: \${int(sol.lb)*param_kwargs['cost_multiplier']:,}. Time to solve: {int(sol.timeToSolve)}s)"
# 		)

# 	# title2= f"$C$: {to_bmatrix(policies.C_policy[:,:,0])},$P$: {to_bmatrix(policies.P_param[:,:,0])},$(C^I,C^D,C^{{setup}})=(\${param_kwargs['c_infected']:,},\${param_kwargs['c_death']:,},\${policies.C_setup[0,0,0]:,})$"

# 	title_rows.append(f"$C^I=\${param_kwargs['c_infected']:,},C^D=\${param_kwargs['c_death']:,}$")

# 	title_rows.append(
# 		f"One Period={param_kwargs['days_per_period']} days (costs scaled by ${param_kwargs['cost_multiplier']:,}$ during optimization)"
# 		)
# 	if title_extra:
# 		title_rows.append(title_extra)
# 	titletext = "\n".join(title_rows)
# 	ax1.set_title(titletext, fontsize=title_fontsize)
# 	ax1.set_ylabel("Number Individuals")
# 	ax1.set_xlabel("Period")
