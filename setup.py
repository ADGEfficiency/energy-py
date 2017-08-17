from setuptools import setup

"""
The setup.py script for energy_py
"""

setup(name='energy_py',

      version='2.0',

      description='Reinforcement learning for energy systems',

      author='Adam Green',

      author_email='adam.green@adgefficiency.com',

      url='http://adgefficiency.com/',

      packages=['energy_py',
                'energy_py.agents',
                'energy_py.agents.naive',
                'energy_py.agents.policy_based',
                'energy_py.agents.value_based',
                'energy_py.envs',
                'energy_py.envs.precool',
                'energy_py.envs.battery',
		        'energy_py.main',
		        'energy_py.main.scripts',
		        'energy_py.main.notebooks'],

      package_data = {'':['*.csv']},
      install_requires=[]
      )
