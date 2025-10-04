import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple

# ================= MSE ==================
def make_mse_loss_function(x_data: NDArray, y_data: NDArray, z_data: NDArray
) -> Tuple[Callable[[NDArray], float], Callable[[NDArray], NDArray]]:
    X = np.column_stack([x_data, y_data, np.ones(len(x_data))])
    y = z_data
    n = len(z_data)

    def loss_function(w: NDArray) -> float:
        predictions = X @ w
        error = y - predictions
        return np.mean(error**2)

    def loss_gradient(w: NDArray) -> NDArray:
        predictions = X @ w
        error = y - predictions
        grad = ((-2 / n) * X.T) @ error
        return grad

    return loss_function, loss_gradient


# ================= RMSE ==================
def make_rmse_loss_function(x_data: NDArray, y_data: NDArray, z_data: NDArray
) -> Tuple[Callable[[NDArray], float], Callable[[NDArray], NDArray]]:
    mse_loss_fn, mse_loss_grad = make_mse_loss_function(x_data, y_data, z_data)

    def rmse_loss_function(w: NDArray) -> float:
        return np.sqrt(mse_loss_fn(w))

    def rmse_loss_gradient(w: NDArray) -> NDArray:
        mse_value = mse_loss_fn(w)
        grad_mse = mse_loss_grad(w)
        return (0.5 / np.sqrt(mse_value + 1e-12)) * grad_mse  # +eps para evitar div/0

    return rmse_loss_function, rmse_loss_gradient


# ================= MAE ==================
def make_mae_loss_function(x_data: NDArray, y_data: NDArray, z_data: NDArray
) -> Tuple[Callable[[NDArray], float], Callable[[NDArray], NDArray]]:
    X = np.column_stack([x_data, y_data, np.ones(len(x_data))])
    y = z_data
    n = len(z_data)

    def loss_function(w: NDArray) -> float:
        predictions = X @ w
        error = np.abs(y - predictions)
        return np.mean(error)

    def loss_gradient(w: NDArray) -> NDArray:
        predictions = X @ w
        error = y - predictions
        grad = -(1/n) * X.T @ np.sign(error)
        return grad

    return loss_function, loss_gradient
