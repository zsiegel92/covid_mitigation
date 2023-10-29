from pyomo_utils import solve_gurobi_direct,solve_gams_direct, solve_executable, solve_pyomo

from math import prod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np

import pyomo.environ as pyo

if True:
    model = pyo.ConcreteModel()
    # use model.i * model.j for cartesian product
    model.i = pyo.RangeSet(0,1)
    model.j = pyo.RangeSet(0,2)
    env = pyo.RangeSet(0,1)

    a = [350, 600]
    b = [325, 300, 275]
    d = [[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]]

    model.x = pyo.Var(model.i, model.j, domain=pyo.NonNegativeIntegers)

    model.obj = pyo.Objective(expr=0.09 * sum(d[i][j] * model.x[i,j] for i in model.i for j in model.j))
    model.obj.expr += 10 #extend objective

    def c1_rule(model, i):
        return sum(model.x[i,j] for j in model.j) <= a[i]
    model.c1 = pyo.Constraint(model.i, rule=c1_rule)

    def c2_rule(model, j):
        return sum(model.x[i,j] for i in model.i) >= b[j]
    model.c2 = pyo.Constraint(model.j, rule=c2_rule)


if True:
    model2 = pyo.ConcreteModel()
    # use model2.i * model2.j for cartesian product
    model2.i = pyo.RangeSet(0,1)
    model2.j = pyo.RangeSet(0,2)
    env = pyo.RangeSet(0,1)

    a = [350, 600]
    b = [325, 300, 275]
    d = [[2.5, 1.7, 1.8], [2.5, 1.8, 1.4]]

    model2.x = pyo.Var(model2.i, model2.j, domain=pyo.NonNegativeIntegers)

    model2.obj = pyo.Objective(expr=0.09 * sum(d[i][j] * model2.x[i,j] for i in model2.i for j in model2.j))
    model2.obj.expr += 10 #extend objective

    def c1_rule(model2, i):
        return sum(model2.x[i,j] for j in model2.j) <= a[i]
    model2.c1 = pyo.Constraint(model2.i, rule=c1_rule)

    def c2_rule(model2, j):
        return sum(model2.x[i,j] for i in model2.i) >= b[j]
    model2.c2 = pyo.Constraint(model2.j, rule=c2_rule)


# model.c = ConstraintList()
# model.c.add()
solver1= "COUENNE"
print(solver1)
sol = solve_executable(model,solver=solver1)
print(model.obj.expr())
# sol.write()
print(model.x.extract_values())


solver2 = "Gurobi"
print(solver2)
sol2 = solve_gurobi_direct(model2)
print(model2.obj.expr())
# sol2.write()
print(model2.x.extract_values())

print(solver2, " through solve_pyomo")
sol3 = solve_pyomo(model2,solver=solver2)
print(model2.obj.expr())
print(model2.x.extract_values())

# model.A = pyo.RangeSet(5)
# model.B = pyo.RangeSet(5)
# model.xx = pyo.Var(model.A,model.B,domain=pyo.NonNegativeReals)
# list(model.xx.index_set())
# model.ccc = pyo.ConstraintList()
# for a in model.A:
#     for b in model.B:
#         model.ccc.add(model.xx[a,b] == 1)



# model.yy = pyo.Var(model.A * model.B,domain=pyo.NonNegativeReals)
# print(list(model.yy.index_set()))

