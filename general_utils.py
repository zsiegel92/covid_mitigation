import numpy as np


def _to_base_b(n, b):
	if b == 1:
		return [1] * n
	q = n // b
	r = n % b
	if q == 0:
		return [r]
	else:
		return to_base_b(q, b) + [r]


def from_base_b(l, b):
	return sum(place_value * (b**i) for i, place_value in enumerate(reversed(l)))


def pad_l(l, padLength=0):
	return [0] * (padLength - len(l)) + l


def to_base_b(n, b, padLength=0):
	return pad_l(_to_base_b(n, b), padLength=padLength)


# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_inefficient(costs):
	print(f"Finding efficient rows of matrix of size {costs.shape}")
	is_efficient = np.arange(costs.shape[0])
	n_points = costs.shape[0]
	next_point_index = 0 # Next index in the is_efficient array to search for
	while next_point_index < len(costs):
		nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
		nondominated_point_mask[next_point_index] = True
		is_efficient = is_efficient[nondominated_point_mask] # Remove dominated points
		costs = costs[nondominated_point_mask]
		next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
	is_inefficient_mask = np.ones(n_points, dtype=bool)
	is_inefficient_mask[is_efficient] = False
	return np.nonzero(is_inefficient_mask)[0]


def is_pareto_efficient(costs):
	print(f"Finding efficient rows of matrix of size {costs.shape}")
	is_efficient = np.arange(costs.shape[0])
	n_points = costs.shape[0]
	next_point_index = 0 # Next index in the is_efficient array to search for
	while next_point_index < len(costs):
		nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
		nondominated_point_mask[next_point_index] = True
		is_efficient = is_efficient[nondominated_point_mask] # Remove dominated points
		costs = costs[nondominated_point_mask]
		next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
	return is_efficient


def test_is_pareto_inefficient():
	xx = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 4], [5, 1, -1]]) #False, True, True, True, False -> 1,2,3
	assert all(is_pareto_inefficient(xx) == [1, 2, 3])

	assert all(is_pareto_efficient(xx) == [0, 4])
	xx = np.array([[1, 2], [3, 4], [2, 1], [1, 1], [1, 1]]) # True, True, True, False -> 0, 1, 2
	assert all(is_pareto_inefficient(xx) == [0, 1, 2, 4]) # tie -> inefficient
	assert (is_pareto_efficient(xx) == [3])

	xx = np.random.rand(100, 5)
	assert len(is_pareto_efficient(xx)) + len(is_pareto_inefficient(xx)) == xx.shape[0]
	assert set(is_pareto_efficient(xx)).union(set(is_pareto_inefficient(xx))) == set(range(xx.shape[0]))
	print(f"TEST is_pareto_inefficient: PASS")


def test_base_conversion():
	bases = list(range(1, 4))
	for b in bases:
		for num in range(max(b**5, 12)):
			l = to_base_b(num, b, padLength=5)
			inv = from_base_b(l, b)
			assert inv == num
			print(f"{num} in base {b} is {l} : (inverted: {inv})")


def assign(mat, indices, val):
	mat[indices] = val


# returns a 1Xm list of policy levels for assortment. The i'th element is the level of policy i. Value j indicates level j, -1 indicates no usage
def convert_int_to_policy_assortment(assortment_index, m, n):
	return list(j - 1 for j in reversed(to_base_b(assortment_index, n + 1, padLength=m)))


# inverse of convert_int_to_policy_assortment_specific if n matches
def convert_policy_assortment_to_int(policy_assortment, n):
	return from_base_b(list(reversed([j + 1 for j in policy_assortment])), n + 1)


if __name__ == "__main__":
	test_base_conversion()
	test_is_pareto_inefficient()
