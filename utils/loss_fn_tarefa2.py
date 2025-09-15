import numpy as np # pyright: ignore[reportMissingImports]
from numpy.typing import NDArray # pyright: ignore[reportMissingImports]
from typing import Callable, Tuple


def make_mse_loss_function(x_data: NDArray, y_data: NDArray, z_data: NDArray) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    X = np.column_stack([x_data ** 3, y_data ** 2, np.ones(len(x_data))])
    y = z_data
    n = len(z_data)

    def loss_function(w: NDArray) -> float:
        predictions = X @ w
        error = y - predictions
        return (1/n) * np.sum(error**2)

    def loss_gradient(w: NDArray) -> NDArray:
        predictions = X @ w
        error = y - predictions
        grad = (-2/n) * (X.T @ error)
        return grad

    return loss_function, loss_gradient


def make_rmse_loss_function(x_data: NDArray, y_data: NDArray, z_data: NDArray) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    mse_loss_fn, mse_loss_grad = make_mse_loss_function(x_data, y_data, z_data)

    def rmse_loss_function(w: NDArray) -> float:
        return np.sqrt(mse_loss_fn(w))

    def rmse_loss_gradient(w: NDArray) -> NDArray:
        mse_value = mse_loss_fn(w)
        if mse_value == 0:
            return np.zeros_like(w)
        return (0.5 / np.sqrt(mse_value)) * mse_loss_grad(w)

    return rmse_loss_function, rmse_loss_gradient


def make_mae_loss_function(x_data: NDArray, y_data: NDArray, z_data: NDArray) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    X = np.column_stack([x_data ** 3, y_data ** 2, np.ones(len(x_data))])
    y = z_data
    n = len(z_data)

    def loss_function(w: NDArray) -> float:
        predictions = X @ w
        error = np.abs(y - predictions)
        return (1/n) * np.sum(error)

    def loss_gradient(w: NDArray) -> NDArray:
        predictions = X @ w
        error = y - predictions
        sign_error = np.sign(error)
        grad = -X.T @ sign_error
        return (1/n) * grad

    return loss_function, loss_gradient
