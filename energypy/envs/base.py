

class AbstractEnv:

    def reset(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def setup_test(self):
        raise NotImplementedError()
