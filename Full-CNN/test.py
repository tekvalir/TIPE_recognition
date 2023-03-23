from layer import *
from layers import *
from activations import *
import numpy as np
from network import Network
from loss_functions import *

net = Network()
net.add(Conv2D(2,6))
net.add(Activation(sigmoid, sigmoid_prime))
net.add(Conv2D(6,1))
net.add(Activation(sigmoid, sigmoid_prime))

x_train=np.array([[[[[0]], [[0]]]], [[[[0]],[[1]]]], [[[[1]],[[1]]]], [[[[1]],[[0]]]]])
y_train=np.array([[[[0]]],[[[1]]], [[[0]]], [[[1]]]])

print(x_train[0].shape)

net.use(mse,mse_prime)
net.fit(x_train, y_train, epochs=1000,learning_rate=0.1)

out = net.predict(x_train)
print(out)

# print(test.weigths.shape)

# c=np.ones((3,3,3))

# c = convolution3D(c,c)
# print(c)
# print(c+5)

# c=np.array([0,5,3])
# e=np.array([[[1,5,0], [1,0,0],[1,0,0],[1,0,0]],[[1,0,0], [1,0,0],[1,0,8],[1,0,0]],[[1,0,0], [1,0,0],[1,0,0],[1,0,0]]])
# print(np.sum(e, axis=(0,1)))

# c=np.array([[11,12],[21,22]])
# print(decal(c,2,2))
# print(np.transpose(c, axes=(1,0)))
# print(np.transpose(np.transpose(c, axes=(1,0))[::-1,::-1], axes=(1,0)))
