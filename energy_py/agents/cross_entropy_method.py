

class CEM(object):
    """
    Implementation of the cross entropy method.

    https://gist.github.com/domluna/022e73fd5128b05bdd96d118b5131631

    https://gist.github.com/kashif/5dfa12d80402c559e060d567ea352c06
    """

    def __init__(self,
                 env,
                 discount,
                 brain_path,

                 policy,
                 lr):
