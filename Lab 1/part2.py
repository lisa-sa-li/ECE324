import numpy as np

class ElementwiseMultiply(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, input):
        if input.shape == self.weight.shape:
            return np.multiply(input, self.weight)

class AddBias(object):
    def __init__(self, bias):
        self.bias = bias

    def __call__(self, input):
        return np.add(self.bias, input)

class LeakyRelu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, input):
        return np.where(input < 0, input*self.alpha, input)

class Compose(object):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, input):
        tempFunction = input
        for func in self.layers:
            tempFunction = func(tempFunction)
        return tempFunction
