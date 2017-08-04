


class Base_Agent(object):
    """
    the energy_py base agent class

    The methods of this class are:
        policy
        learning
    """

    def __init__(self):

        return None

    def _policy(self, observation): raise NotImplementedError
    def _learning(self, observation): raise NotImplementedError

    def policy(self, observation):
        """
        """
        return self._policy(observation)

    def learning():
        """
        """
        return self._learning()
