import numpy as np

def mse(y_fit, y_pred):
    return np.mean(np.power(y_fit-y_pred, 2))

def mse_prime(y_fit, y_pred):
    return 2*(y_pred-y_fit)/y_fit.size

def binary_cross_entropy(y_fit,y_pred):
    return -np.mean(y_fit*np.log(y_pred) + (1-y_fit)*np.log(1-y_pred))

def binary_cross_entropy_prime(y_fit,y_pred):
    return -((y_fit/y_pred)-((1-y_fit)/(1-y_pred)))
