"""
Objects to decay parameters used in experiments - for example to decay
epsilon in an e-greedy policy
"""

def test_linear_decay():

    ld = LinearDecayer(10, 20, 1.0, 0.0)

    assert ld.value(0) == 1.0
    assert ld.value(10) == 1.0

    assert ld.value(11) == 1.0 - (1/20)
    assert ld.value(15) == 1.0 - (5/20)

    assert ld.value(31) == 0.0


class LinearDecayer(object):

    def __init__(self, predecay_steps, decay_steps, start_val, end_val):

        self.predecay_steps = int(predecay_steps)
        self.decay_steps = int(decay_steps)

        self.start_val = float(start_val)
        self.end_val = float(end_val)

        self.decay_end = self.predecay_steps + self.decay_steps

        self.val_delta = self.start_val - self.end_val

    def value(self, step):

        if step < self.predecay_steps:
            return self.start_val

        else:
            decay_step = step - self.predecay_steps
            fraction = min(decay_step / self.decay_steps, 1.0)
            return self.start_val - fraction * self.val_delta


class PropertyDecayer(object):

    def __init__(self, initial):
        self._val = initial

    @property
    def value(self):
        return self._val

    @value.getter
    def value(self):
        self._val -= 1
        return self._val


if __name__ == '__main__':
    ld = LinearDecayer(10, 5, 0.9, 0.1)

    for step in range(20):
        print(ld.value(step))

    test_linear_decay()

    prop = PropertyDecayer(1.0)
