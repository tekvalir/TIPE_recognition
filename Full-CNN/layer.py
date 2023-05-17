# class Layer:
#     def __init__(self):
#         self.input = None
#         self.output = None
#     def forward_propagation(self, input):
#         raise NotImplementedError
#     def backward_propagation(self, output_error, learning_rate):
#         raise NotImplementedError

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.input_size = None
        self.output_size = None
    def forward_propagation(self, input):
        raise NotImplementedError
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError