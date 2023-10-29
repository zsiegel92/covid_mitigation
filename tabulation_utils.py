import pandas as pd
import numpy as np
import jinja2


def prioritize_value_one(col_series):
	return pd.core.series.Series([-np.inf if y == 1 else y for y in col_series])


def mask_indices(df, index_cols, prioritize_ones=False):
	x = df.copy()
	x.reset_index()
	# put index columns first
	for i, colname in enumerate(index_cols):
		x['tmp'] = x[colname]
		del x[colname]
		x.insert(i, colname, x['tmp'])
		del x['tmp']
	x.sort_values(
		by=index_cols, key=prioritize_value_one if prioritize_ones else None, inplace=True
		)
	# clear columns
	n_ind = len(index_cols)
	for clearcol in range(1, n_ind):
		# print(clearcol)
		repcols = index_cols[:-clearcol]
		# print(repcols)
		# print(index_cols[-clearcol - 1])
		clearcolname = repcols[-1]
		x.loc[x[repcols].duplicated(), clearcolname] = ''
	return x


def replace_all(string, list_of_2_tuples):
	if len(list_of_2_tuples) < 1:
		return string
	return replace_all(
		string.replace(list_of_2_tuples[0][0], list_of_2_tuples[0][1]), list_of_2_tuples[1:]
		)


def limit_to_trials_with_only_one_parameter_varied(df):
	df = df[(df['trial_effect_multiplier'] == 1) | (df['trial_cost_multiplier'] == 1)]
	return df
