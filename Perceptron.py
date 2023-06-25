import numpy as np

class Perceptron():
    def __init__(self, weights : list, bias) -> None:
        self.weights = weights
        self.bias = bias

    def processInput(self, inputs: list):
        inputs = inputs[:2]
        return np.sum(np.dot(self.weights, inputs)) + self.bias
    
NAND = Perceptron([-2, -2], 3)
print(NAND.processInput([1,1, 3]))