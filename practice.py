import numpy as np
from collections import Counter


class parent:
    def __init__(self, hello):
        self.hi = hello

class child(parent):
    def __init__(self, hi):
        self.hello = hi
        super().__init__(
            hello=hi
        )


    def myfunc(self):


a = child('aaaaaa')
print(a.hi, a.hello)
