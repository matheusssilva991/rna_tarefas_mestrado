import numpy as np
from utils.activations_fn import tanh, tanh_derivative


def make_mse_loss_for_neuron(x_data, y_data, z_data, activation_fn=tanh, activation_deriv=tanh_derivative):
    X = np.column_stack([x_data, y_data, np.ones(len(x_data))])
    y = z_data
    n = len(y)

    def loss_fn(w):
        a = X @ w
        y_pred = activation_fn(a)
        error = y - y_pred
        return np.mean(error**2)

    def grad_fn(w):
        a = X @ w
        y_pred = activation_fn(a)
        error = y - y_pred
        grad = (-2/n) * X.T @ (error * activation_deriv(a))
        return grad

    return loss_fn, grad_fn
