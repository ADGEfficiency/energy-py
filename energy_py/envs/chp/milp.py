import pulp

from pulp import LpProblem, LpMinimize, LpVariable, LpStatus


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
            'electrical': 0.28
            'thermal': 0.4
        }

        #  MW
        self.load = LpVariable(name, 50, 100)

        heat_generated = self.size * self.load * (1 / self.effy['thermal'])
        self.steam_generated = heat_generated * (1 / enthalpy) * 3.6

    def gas_burnt(self):
        self.prob += self.size * self.load * (1 / self.effy['electrical']) * gas_price


class Boiler(object):

    def __init__(
            self,
            prob,
            size,
            name,
            min_turndown=0.0
    ):
        self.prob = prob
        self.effy = {
            'thermal': 0.8
        }

        #  t/h
        self.steam_generated = LpVariable(name, min_turndown, size)

    def gas_burnt(self):
        #  https://www.tlv.com/global/TI/calculator/steam-table-temperature.html
        #  30 barG, 250 C vapour - liquid enthalpy at 100C
        #  MJ/kg = kJ/kg * MJ/kJ

        #  MW = t/h * kg/t * hr/sec * MJ/kg / effy
        self.prob += self.load * (1/3.6) * enthalpy * (1/self.effy['thermal']) * gas_price

enthalpy =  (2851.34 - 418.991) / 1000
gas_price = 20

prob = LpProblem('cost minimization', LpMinimize)

assets = [
    GasTurbine(prob=prob, size=50, name='gt1'),
    Boiler(prob=prob, size=100, name='blr1')

]

steam_generated = sum([asset.steam_generated for asset in assets])

prob += steam_generated == 50

prob.writeLP('chp.lp')

prob.solve()

print(LpStatus[prob.status])

for v in prob.variables():
    print('{} {}'.format(v.name, v.varValue))
