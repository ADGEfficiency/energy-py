import pulp

from pulp import LpProblem, LpMinimize, LpVariable, LpStatus

"""

obj needs all the costs
- gas
- elect

balances need 
- steam
- elect

"""


class GasTurbine(object):

    def __init__(
            self,
            prob,
            size,
            name,
    ):
        self.prob = prob
        self.size = size

        self.effy = {
            'electrical': 0.28,
            'thermal': 0.4
        }

        #  MW
        self.load = LpVariable(name, 0, 1.0)

    def steam_generated(self):
        heat_generated = self.size * self.load * (1 / self.effy['electrical']) * self.effy['thermal']
        return heat_generated * (1 / enthalpy) * 3.6

    def gas_burnt(self):
        return self.size * self.load * (1 / self.effy['electrical']) * gas_price

    def power_generated(self):
        return self.load * self.size


class Boiler(object):

    def __init__(
            self,
            prob,
            size,
            name,
            min_turndown=0.0
            parasitics=0.0,
    ):
        self.prob = prob
        self.effy = {
            'thermal': 0.8
        }

        #  t/h
        self.load = LpVariable(name, min_turndown, size)

        self.parasitics = parasitics

    def steam_generated(self):
        return self.load

    def gas_burnt(self):
        #  https://www.tlv.com/global/TI/calculator/steam-table-temperature.html
        #  30 barG, 250 C vapour - liquid enthalpy at 100C
        #  MJ/kg = kJ/kg * MJ/kJ

        #  MW = t/h * kg/t * hr/sec * MJ/kg / effy
        return self.load * (1/3.6) * enthalpy * (1/self.effy['thermal']) * gas_price

    def power_generated(self):
        return self.parasitics

enthalpy =  (2851.34 - 418.991) / 1000
gas_price = 20

prob = LpProblem('cost minimization', LpMinimize)

assets = [
    GasTurbine(prob=prob, size=10, name='gt1'),
    Boiler(prob=prob, size=10, name='blr1')
]

#  need to form objective function first
prob += sum([asset.gas_burnt() for asset in assets]) 

#  then add constraints
prob += sum([asset.steam_generated() for asset in assets]) == 10

prob.writeLP('chp.lp')

prob.solve()

print(LpStatus[prob.status])

for v in prob.variables():
    print('{} {}'.format(v.name, v.varValue))
