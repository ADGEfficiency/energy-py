from setuptools import setup, find_packages

setup(
    name='energypy',
    version='0.3.0',

    description='reinforcement learning for energy systems',
    author='Adam Green',
    author_email='adam.green@adgefficiency.com',
    url='http://www.adgefficiency.com/',

    packages=find_packages(exclude=['tests', 'tests.*']),
    package_dir={'energypy': 'energypy'},
    package_data={'energypy': ['experiments/datasets/example/*.csv']},

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=['Click'],
    entry_points='''
            [console_scripts]
            ep-expt=energypy.experimentsv2.first_look:cli
        '''
)
