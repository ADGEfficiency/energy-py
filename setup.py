from setuptools import setup, find_packages

"""
The setup.py script for energy_py
"""

setup(name='energy_py',

      version='0.2',

      description='reinforcement learning for energy systems',

      author='Adam Green',

      author_email='adam.green@adgefficiency.com',

      url='http://adgefficiency.com/',

      packages=find_packages(exclude=['tests', 'tests.*']),

      package_data = {'':['*.csv']},
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      install_requires=[]
      )
