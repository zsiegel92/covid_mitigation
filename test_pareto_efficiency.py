import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from general_utils import is_pareto_inefficient, is_pareto_efficient, test_is_pareto_efficient

if __name__ == "__main__":
	costs = np.random.rand(1000, 3)
	nVals = list(reversed(range(1, 100000, 100)))
	nTrials = 5
	df = pd.DataFrame(
		dict(
			n=nVals,
			n_efficient=[
				np.mean([len(is_pareto_efficient(np.random.rand(n, 3)))
							for _ in range(nTrials)])
				for n in nVals
				],
			)
		)
	df['ratio'] = df['n_efficient'] / df['n']

	plot = sns.lineplot(data=df, x="n", y="ratio")
	plot.set(yscale='log')
	plot.set(xscale='log')
	plt.show(block=False)
