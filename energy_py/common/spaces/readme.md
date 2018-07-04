energy_py Space objects - inspired by the OpenAI gym Spaces.

Compatability with gym spaces is ideal as it allows energy_py agents
to be used with gym environments.

The energy_py GlobalSpace is the equivilant of the gym TupleSpace.

As GlobalSpaces are created by the environment, discrete representations of
the spaces that form the GlobalSpace are available on demand via

GlobalSpace.discretize(n_discr=10)
sample = GlobalSpace.sample_discrete()
