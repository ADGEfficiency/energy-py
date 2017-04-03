class gas_turbine(object):

    def __init__(self, size, name):
        self.size = float(size)  # power output in MWe
        self.name = str(name)
        self.variables = [
                        {'Name': self.name + ' Load',
                        'Current': 0, 'Min': 50, 'Max': 100, 'Init': 50, 'Radius': 20},
                        {'Name': self.name + ' On/Off',
                        'Current': 0, 'Min': 0, 'Max': 1, 'Init': 1}]
        self.reset()
        self.update()

    def update(self):
        self.on_off = int(self.variables[1]['Current'])
        self.load = float(self.variables[0]['Current']/100) * self.on_off  # %
        self.power_output = float(self.size * self.load)  # MWe
        elect_effy = self.load * 0.120 + 0.196667  # % HHV
        thermal_effy = self.load * 0.10 + 0.401667  # % HHV
        self.gas_burnt = float(self.power_output / elect_effy)  # MW HHV
        self.HG_heat_output = float(self.gas_burnt * thermal_effy)  # MW
        self.LG_heat_output = 0  # MW
        self.unrecoverable_heat = 0  # MW
        self.cooling_output = 0  # MW

    def reset(self):
        for var in self.variables:
            var['Current'] = var['Init']

class gas_engine(object):

    def __init__(self, size, name):
        self.size = float(size)  # power output in MWe
        self.name = str(name)
        self.variables = [
            {'Name': self.name + ' Load',
            'Current': 0, 'Min': 50, 'Max': 100, 'Init': 50, 'Radius': 20},
            {'Name': self.name + ' On/Off',
            'Current': 0, 'Min': 0, 'Max': 1, 'Init': 1}]
        self.reset()
        self.update()

    def update(self):
        self.on_off = int(self.variables[1]['Current'])
        self.load = float(self.variables[0]['Current']/100) * self.on_off  # %
        self.power_output = float(self.size * self.load)  # MWe
        elect_effy = self.load * 0.08 + 0.3
        HG_effy = self.load * 0 + 0.2
        LG_effy = self.load * 0.04 + 0.16
        self.gas_burnt = float(self.power_output / elect_effy)  # MW HHV
        self.HG_heat_output = float(self.gas_burnt * HG_effy)  # MW
        self.LG_heat_output = float(self.gas_burnt * LG_effy)  # MW
        self.unrecoverable_heat = 0  # MW
        self.cooling_output = 0  # MW

    def reset(self):
        for var in self.variables:
            var['Current'] = var['Init']
