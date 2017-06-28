"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Code kindly supplied by Felipe Aguirre Martinez - many thanks!
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from functools import wraps
import os
import pickle
import time


class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str

    def __call__(self, func):
        @wraps(func)
        def timed_func(*args, **kwargs):
            try:
                self.restart()
                output = func(*args, **kwargs)
                print(
                    "Finished execution of {}() in {}".format(
                        func.__name__,
                        self.get_time()
                    )
                )
                return output
            except Exception as e:
                print(
                    "Failed execution of {}() after {}".format(
                        func.__name__,
                        self.get_time()
                    )
                )
                raise e

        return timed_func



def dump_pickle(obj, name):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


if __name__ == '__main__':

    from time import sleep

    timer = Timer()
    sleep(1.1)
    print(timer.get_time())

    @Timer()
    def func():
        sleep(1.1)

    @Timer()
    def failed_func():
        sleep(1.1)
        1/0

    func()
    failed_func()
