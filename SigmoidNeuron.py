import numpy as np

class SigmoidNeuron():
    def __init__(self, weights : list, bias) -> None:
        self.weights = weights
        self.bias = bias

    def processInput(self, inputs: list):
        inputs = inputs[:len(self.weights)]
        x = np.sum(np.dot(self.weights, inputs)) + self.bias
        return 1/(1 + np.exp(-x))
    
NAND = SigmoidNeuron([-2, -2], 3)
print(NAND.processInput([1,1, 3]))