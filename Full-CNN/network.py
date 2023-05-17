from layer import Layer
from loss_functions import mse, mse_prime

# class Network:
#     def __init__(self):
#         self.layers =[]
#         self.loss = None
#         self.loss_prime = None
#     def add(self,layer: Layer):
#         self.layers.append(layer)
#     def use(self, loss, loss_prime):
#         self.loss = loss
#         self.loss_prime = loss_prime
#     def predict(self, input_data):
#         n = len(input_data)
#         result = []
#         for i in range (n):
#             output=input_data[i]
#             for layer in self.layers:
#                 output = layer.forward_propagation(output)
#             result.append(output)
#         return result
#     def fit(self, x_train, y_train, epochs, learning_rate):
#         n = len(x_train)
#         for i in range(epochs):
#             err=0
#             for j in range(n):
#                 output=x_train[j]
#                 for layer in self.layers:
#                     output=layer.forward_propagation(output)
#                 err+=self.loss(y_train[j], output)
#                 error = self.loss_prime(y_train[j],output)
#                 for layer in reversed(self.layers):
#                     error=layer.backward_propagation(error,learning_rate)
#             err/=n
#             print("epochs %d/%d error=%f" % (i+1,epochs,err))


class Network:
    def __init__(self, input: Layer, output: Layer):
        self.input = input
        self.output = output
        self.network={}
        pile=[output]
        while len(pile)>0:
            layer=pile.pop()
            for l in layer.get_previouses():
                if l not in self.network:
                    pile.append(l)
                    self.network[l]=[]
                self.network[l].append(layer)
    def predict(self, input_data):
        pass
    def fit(self, x_train, y_train, **kwargs):
        x_val = kwargs['x_val'] if 'x_val' in kwargs else None
        y_val = kwargs['y_val'] if 'y_val' in kwargs else None
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 10
        loss_fn = kwargs['loss_fn'] if 'loss_fn' in kwargs else mse
        loss_fn_prime = kwargs['loss_fn_prime'] if 'loss_fn_prime' in kwargs else mse_prime
        lr = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
        print("epochs=%d, loss_fn=%s, loss_fn_prime=%s, learning_rate=%f" % (epochs, loss_fn, loss_fn_prime, lr))
