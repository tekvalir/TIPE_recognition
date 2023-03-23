import numpy as np

from network import Network
from layers import FCLayer
from activations import Activation
from activations import tanh,tanh_prime,sigmoid,sigmoid_prime,relu,relu_prime
from loss_functions import mse,mse_prime,binary_cross_entropy,binary_cross_entropy_prime

x_train=np.array([[[0,0]], [[0,1]], [[1,1]], [[1,0]]])
y_train=np.array([[[0]],[[1]], [[0]], [[1]]])

net = Network()
net.add(FCLayer(2,3))
net.add(Activation(tanh, tanh_prime))
net.add(FCLayer(3,1))
# net.add(Activation(sigmoid, sigmoid_prime))
net.add(Activation(tanh,tanh_prime))


# net.use(binary_cross_entropy, binary_cross_entropy_prime)
net.use(mse,mse_prime)
net.fit(x_train, y_train, epochs=1000,learning_rate=0.1)

out = net.predict(x_train)
print(out)