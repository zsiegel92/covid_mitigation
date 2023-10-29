from sympy import IndexedBase, symbols, expand, simplify, diff, ln, latex

T = 5

KI = .0000006
KR = .03
KD = .015

# x = IndexedBase('x')
# I = IndexedBase('I')
# R = IndexedBase('R')
# d = IndexedBase('d')
# D = IndexedBase('D')
P = IndexedBase('P')

S0, I0, R0, D0, d0 = symbols('S0 I0 R0 D0 d0')
S = [S0] * T
I = [I0] * T
R = [R0] * T
D = [D0] * T
d = [d0] * T
S[0] = S0
I[0] = I0
R[0] = R0
D[0] = D0
d[0] = d0
for t in range(1, T):
	print(f"t={t}")
	S[t] = simplify(expand(S[t - 1] - KI * P[t] * S[t - 1] * I[t - 1]))
	I[t] = simplify(
		expand(I[t - 1] + KI * P[t] * S[t - 1] * I[t - 1] - KR * I[t - 1] - KD * I[t - 1])
		)
	R[t] = simplify(expand(R[t - 1] + KR * I[t - 1]))
	d[t] = simplify(expand(KD * I[t - 1]))
	D[t] = simplify(expand(D[t - 1] + d[t]))

c_infected = 10000
c_death = 10000000
llambda = IndexedBase('lam')

objective = sum(c_infected * I[t] + c_death * d[t] - llambda[t] * ln(P[t]) for t in range(T))

objective_gradient_wrt_P = [diff(objective, P[t]) for t in range(T)]

print(latex(objective))

latex_grad = [latex(expr) for expr in objective_gradient_wrt_P]
rows = []
for t, expr in enumerate(latex_grad):
	rows.append(
		f"\\begin{{equation*}}\n\\frac{{\\partial}}{{\\partial P_{t}}} = {expr}\n\\end{{equation*}}"
		)

print("\n".join(rows[0:3]))

# grad_rows = '\\\\\n'.join(latex_grad)
# latex_vector = f"\\begin{{bmatrix}}\n{grad_rows}\n\\end{{bmatrix}}"
# print(latex_vector)
