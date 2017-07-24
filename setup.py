from setuptools import setup

setup(name='energy_py',
      version='2.0',
      description='Reinforcement learning for energy systems',
      author='Adam Green',
      author_email='adam.green@adgefficiency.com',
      url='http://adgefficiency.com/',
      packages=['energy_py',
                'energy_py.agents',
                'energy_py.envs',
                'energy_py.main'],
      install_requires=['numpy',
                        'pandas',
                        'tensorflow-gpu',
                        'keras'])
