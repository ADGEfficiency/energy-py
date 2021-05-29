from setuptools import setup, find_packages


setup(
    name='energypy',
    version='0.3.0',
    packages=find_packages(),
    install_requires=['Click'],
    entry_points={
        'console_scripts': [
            'energypy=energypy.main:cli'
        ],
    }
)
