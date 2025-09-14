import numpy as np
from typing import Callable, Tuple

def make_mse_loss_function(x_data, y_data, z_data):
    X = np.column_stack([x_data ** 3, y_data ** 2, np.ones(len(x_data))])
    y = z_data
    n = len(z_data)

    def loss_function(w):
        predictions = X @ w
        error = y - predictions
        return (1/n) * np.sum(error**2)

    def loss_gradient(w):
        X_2 = X.T @ X
        grad = (-2 * X.T @ y) + (X_2 + X_2.T) @ w
        return 1/n * grad

    return loss_function, loss_gradient


def make_rmse_loss_function(x_data, y_data, z_data):
    mse_loss_fn, mse_loss_grad = make_mse_loss_function(x_data, y_data, z_data)

    def rmse_loss_function(w):
        return np.sqrt(mse_loss_fn(w))

    def rmse_loss_gradient(w):
        mse_value = mse_loss_fn(w)
        if mse_value == 0:
            return np.zeros_like(w)
        return (0.5 / np.sqrt(mse_value)) * mse_loss_grad(w)

    return rmse_loss_function, rmse_loss_gradient


def make_mae_loss_function(x_data, y_data, z_data):
    X = np.column_stack([x_data ** 3, y_data ** 2, np.ones(len(x_data))])
    y = z_data
    n = len(z_data)

    def loss_function(w):
        predictions = X @ w
        error = np.abs(y - predictions)
        return (1/n) * np.sum(error)

    def loss_gradient(w):
        predictions = X @ w
        error = y - predictions
        sign_error = np.sign(error)
        grad = -X.T @ sign_error
        return (1/n) * grad

    return loss_function, loss_gradient
