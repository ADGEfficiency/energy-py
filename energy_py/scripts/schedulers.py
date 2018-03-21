"""
Objects to schedule parameters used in experiments - for example to decay
epsilon in an e-greedy policy
"""


def test_linear_decay():

    sch_args = {'pre_step': 10,
                'sched_step': 20,
                'initial': 1.0,
                'final': 0.0}

    ld = LinearScheduler(**sch_args)

    vals = [ld() for _ in range(50)]

    assert vals[0] == 1.0
    assert vals[10] == 1.0

    assert vals[11] == 1.0 - (1/20)
    assert vals[15] == 1.0 - (5/20)

    assert vals[31] == 0.0

def test_linear_increase():

    sch_args = {'pre_step': 20,
                'sched_step': 20,
                'initial': 0.4,
                'final': 1.2}

    ld = LinearScheduler(**sch_args)

    vals = [ld() for _ in range(50)]

    assert vals[0] == 0.4
    assert vals[20] == 0.4

    assert vals[21] == 0.4 + (1.2-0.4)*(1/20)

    assert vals[29] == 0.4 + (1.2-0.4)*(9/20)
    assert vals[39] == 0.4 + (1.2-0.4)*(19/20)
    assert vals[41] == 1.2
    assert vals[-1] == 1.2

class LinearScheduler(object):

    def __init__(self, sched_step, initial, final, pre_step=0):

        self.pre_step = int(pre_step)
        self.sched_step = int(sched_step)

        self.initial = float(initial)
        self.final = float(final)

        self._val = initial
        self.step = 0

        self.coeff = (self.initial - self.final) / self.sched_step

    @property
    def value(self):
        return self._val

    @value.getter
    def value(self):
        if self.step > self.pre_step + self.sched_step:
            self._val = self.final

        elif self.step < self.pre_step:
            self._val = self.initial

        else:
            self._val = self.initial + self.coeff * (self.pre_step - self.step)

        self.step += 1
        return self._val

    def __call__(self):
        return self.value

if __name__ == '__main__':
    sch_args = {'pre_step': 10,
                'sched_step': 10,
                'initial': 0.9,
                'final': 0.1}
    ld = LinearScheduler(**sch_args)

    for step in range(20):
        print(ld())
