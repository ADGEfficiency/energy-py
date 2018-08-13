
import pulp

from pulp import LpProblem, LpMinimize, LpVariable, LpStatus

prob = LpProblem('cost minimization', LpMinimize)

boiler_1 = LpVariable('boiler_1', 0, 100)
boiler_2 = LpVariable('boiler_2', 0, 100)

prob += boiler_1 * 2 + boiler_2 * 3

#  steam balance
prob += boiler_1 + boiler_2 == 50

prob.writeLP('chp.lp')

prob.solve()

print(LpStatus[prob.status])

for v in prob.variables():
    print('{} {}'.format(v.name, v.varValue))
