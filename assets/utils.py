"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Code kindly supplied by Felipe Aguirre Martinez - many thanks!
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import time
from functools import wraps


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
