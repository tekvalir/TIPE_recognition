import numpy as np

from network import Network
from layers import FCLayer
from activations import Activation
from activations import tanh,tanh_prime,relu,relu_prime,sigmoid,sigmoid_prime
from loss_functions import mse,mse_prime,binary_cross_entropy,binary_cross_entropy_prime
from sklearn import datasets

iris = datasets.load_iris()
print(iris['data'].shape)

x_train=np.reshape(iris['data'], (150,1,4))[:100]
y_train=np.reshape(iris['target'], (150,1,1))[:100]
print(y_train)

net = Network()
net.add(FCLayer(4,8))
net.add(Activation(tanh, tanh_prime))
net.add(FCLayer(8,1))
net.add(Activation(sigmoid,sigmoid_prime))



net.use(binary_cross_entropy, binary_cross_entropy_prime)
net.fit(x_train, y_train, epochs=100,learning_rate=0.1)

out = net.predict(x_train)
print(np.reshape(np.round(out).astype(int), (100)))
print(np.reshape(y_train, (100)))
print(np.reshape(y_train==np.round(out), (100)))