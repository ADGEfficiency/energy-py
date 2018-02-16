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
                'energy_py.envs',
                'energy_py.envs.flex',
                'energy_py.envs.battery',
                'energy_py.experiments',
                'energy_py.scripts'],

      package_data = {'':['*.csv']},
      install_requires=[]
      )
