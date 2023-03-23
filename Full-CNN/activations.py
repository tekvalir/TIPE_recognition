from layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output
    def backward_propagation(self, output_error, learning_rate):
        return output_error*self.activation_prime(self.input)
    
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(-0.01*x, x)

def relu_prime(x):
    return (x>0)*1+(x<=0)*-0.01 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return (np.exp(-x)/np.power(1+np.exp(-x), 2))