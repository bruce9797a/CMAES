import numpy as np


class Function():
    def __init__(self,dim,lb,ub):
        self.dimension = dim
        self.lower = lb
        self.upper = ub

    def evaluate(self,x):
        pass


class Rosenbrock(Function):
    def __init__(self,dim,lb,ub):
        super().__init__(dim, lb, ub)

    def evaluate(self,x):
        x = self.lower + x * (self.upper - self.lower)
        result = 0
        for i in range(self.dimension - 1):
            result += ( np.power((1 - x[i]), 2) + 100 * np.power( x[i + 1]-np.power(x[i], 2) , 2) )
        return result

