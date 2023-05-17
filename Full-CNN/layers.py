from layer import Layer
import numpy as np

# class FCLayer(Layer):
#     def __init__(self, input_size, output_size):
#         self.weights = np.random.rand(input_size, output_size) - 0.5
#         self.bias = np.random.rand(1, output_size) - 0.5
#     def forward_propagation(self, input):
#         self.input = input
#         self.output = np.dot(self.input, self.weights) + self.bias
#         return self.output
#     def backward_propagation(self, output_error, learning_rate):
#         weights_error = np.dot(self.input.T, output_error)
#         input_error=np.dot(output_error, self.weights.T)
#         self.weights -= learning_rate*weights_error
#         self.bias -= learning_rate*output_error
#         return input_error

# class Conv2D(Layer):
#     def __init__(self, channels, filter):
#         self.filter=filter
#         self.channels = channels
#         self.weigths = np.random.rand(filter, 3, 3, channels)
#         self.bias = np.random.rand(filter)
#         self.params = filter*3*3*channels+filter
#     def forward_propagation(self, input):
#         self.input = input
#         self.output = np.zeros((self.filter,input.shape[0],input.shape[1]))
#         for i in range(self.filter):
#             self.output[i] = convolution3D_2axis(input, self.weigths[i])+self.bias[i]
#         return self.output
#     def backward_propagation(self, output_error, learning_rate):
#         self.bias -= learning_rate*np.sum(output_error, axis=(0,1))
#         dw = np.zeros_like(self.weigths)
#         for f in range (self.filter):
#             for c in range(self.channels):
#                 for k in range(3):
#                     for l in range(3):
#                         dw[f,k,l,c]=np.sum(output_error*decal(self.input,k,l))
#         dx=convolution3D_2axis(output_error,np.transpose(np.transpose(self.weigths, axes=(2,1))[::-1,::-1], axes=(2,1)))
#         self.weigths -= learning_rate*dw
#         return dx

def convolution3D_2axis(mat,ker):
    ''' Effectue la convolution du tableau numpy mat par le noyau ker de taille 3x3 '''
    mat_r = np.zeros((mat.shape[0]+2, mat.shape[1]+2, mat.shape[2]))
    mat_r[1:-1,1:-1]=mat
    ret = np.zeros(mat.shape[:2])
    for i in range (mat.shape[0]):
        for j in range (mat.shape[1]):
            ret[i,j]=np.sum(ker*mat_r[i:i+3,j:j+3])
    return ret

def decal(mat,k,l):
    ret = np.zeros_like(mat)
    ret[max(0,1-k):min(mat.shape[0],mat.shape[0]-k+1),max(0,1-l):min(mat.shape[1], mat.shape[1]-l+1)]=mat[max(0,k-1):min(mat.shape[0],mat.shape[0]+k-1),max(0,l-1):min(mat.shape[1],mat.shape[1]+l-1)]
    return ret

class Input(Layer):
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size
    def get_previouses(self):
        return []
    def forward_propagation(self, input):
        if input.shape != self.input_size:
            raise ValueError("Input shape is not correct")
        self.input = input
        self.output = input
        return self.output
    def backward_propagation(self, output_error, learning_rate):
        return output_error

class Conv2D(Layer):
    def __init__(self, filters: int, previous: Layer):
        self.previous = previous
        self.input_size = previous.output_size
        self.output_size = (previous.output_size[0], previous.output_size[1], filters)
        self.weights = np.random.rand(filters, 3, 3, previous.output_size[2])
        self.bias = np.random.rand(filters)
        self.params = filters*3*3*previous.output_size[2]+filters
    def get_previouses(self):
        return [self.previous]
    def forward_propagation(self, input):
        self.input = input
        self.output = np.zeros(self.output_size)
        for i in range(self.output_size[2]):
            self.output[:,:,i] = convolution3D_2axis(input, self.weights[i])+self.bias[i]
        return self.output
    def backward_propagation(self, output_error, learning_rate):
        self.bias -= learning_rate*np.sum(output_error, axis=(0,1))
        dw = np.zeros_like(self.weigths)
        for f in range (self.output_size[2]):
            for c in range(self.input_size[2]):
                for k in range(3):
                    for l in range(3):
                        dw[f,k,l,c]=np.sum(output_error*decal(self.input,k,l))
        dx=convolution3D_2axis(output_error,np.transpose(np.transpose(self.weigths, axes=(2,1))[::-1,::-1], axes=(2,1)))
        self.weigths -= learning_rate*dw
        return self.previous.backward_propagation(dx, learning_rate)