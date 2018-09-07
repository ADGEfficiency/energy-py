from setuptools import setup, find_packages


setup(
    name='energy_py',

    version='0.2',
    description='reinforcement learning for energy systems',
    author='Adam Green',
    author_email='adam.green@adgefficiency.com',
    url='http://adgefficiency.com/',

    packages=find_packages(exclude=['tests', 'tests.*']),
    package_dir={'energy_py': 'energy_py'},
    package_data={'energy_py': ['experiments/datasets/example/*.csv']},

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[]
)
