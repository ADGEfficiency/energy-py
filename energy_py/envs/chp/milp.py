import pulp

from pulp import LpProblem, LpMinimize, LpVariable, LpStatus

"""
all efficiencies are HHV

you cant use divide by with pulp
have to do obj first then do constraints

obj needs all the costs
- gas
- elect

balances need
- steam
- elect

"""

class GasTurbine(object):
    lb = 50
    ub = 100

    def __init__(
            self,
            prob,
            size,
            name,
    ):
        self.prob = prob

        self.size = size

        self.cont = LpVariable(name, 0, self.ub)
        self.binary = LpVariable('{}_bin'.format(name), lowBound=0, upBound=1, cat='Integer')

        self.load = self.cont * (1/100)

        self.effy = {
            'electrical': 0.28,
            'thermal': 0.4
        }

    def steam_generated(self):
        """ t/h """
        heat_generated = self.size * self.load * (1 / self.effy['electrical']) * self.effy['thermal']
        return heat_generated * (1 / enthalpy) * 3.6

    def gas_burnt(self):
        """  MW HHV """
        return self.size * self.load * (1 / self.effy['electrical'])

    def power_generated(self):
        """  MW """
        return self.load * self.size


class Boiler(object):

    def __init__(
            self,
            prob,
            size,
            name,
            min_turndown=0.0,
            parasitics=0.0
    ):
        self.prob = prob

        self.lb = min_turndown
        self.ub = size

        self.cont = LpVariable(name, 0, size)
        self.binary = LpVariable('{}_bin'.format(name), lowBound=0, upBound=1, cat='Integer')
        self.load = self.cont

        self.effy = {
            'thermal': 0.8
        }

        self.parasitics = parasitics

    def steam_generated(self):
        """ t/h """
        return self.load

    def gas_burnt(self):
        """ MW HHV """
        #  https://www.tlv.com/global/TI/calculator/steam-table-temperature.html
        #  30 barG, 250 C vapour - liquid enthalpy at 100C
        #  MJ/kg = kJ/kg * MJ/kJ

        #  MW = t/h * kg/t * hr/sec * MJ/kg / effy
        return self.load * (1/3.6) * enthalpy * (1/self.effy['thermal'])

    def power_generated(self):
        """ MW """
        return self.parasitics


enthalpy =  (2851.34 - 418.991) / 1000
gas_price = 20
electricity_price = 10

prob = LpProblem('cost minimization', LpMinimize)

assets = [
    GasTurbine(prob=prob, size=10, name='gt1'),
    Boiler(prob=prob, size=100, name='blr1')
]

#  need to form objective function first
prob += sum([asset.gas_burnt() for asset in assets]) * gas_price \
    - sum([asset.power_generated() for asset in assets]) * electricity_price

prob += sum([asset.steam_generated() for asset in assets]) == 100, 'steam_balance'

net_grid = LpVariable('net_power_to_site', -100, 100)
prob += sum([asset.power_generated() for asset in assets]) + net_grid == 100, 'power_balance'

#  asset constraints
for asset in assets:
    prob += asset.cont - asset.ub * asset.binary <= 0
    prob += asset.lb * asset.binary - asset.cont <= 0

prob.writeLP('chp.lp')

prob.solve()

print(LpStatus[prob.status])

for v in prob.variables():
    print('{} {}'.format(v.name, v.varValue))
