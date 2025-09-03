import numpy as np
from typing import Callable, Sequence, Tuple


def gradient_descendent(
    x0: Sequence[float | int] | np.ndarray,
    cost_fn: Callable[[np.ndarray], float],
    grad_cost_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    max_iter: int,
    tolerance: float,
    stopping_criterion: int = 1,
) -> Tuple[list[np.ndarray], list[float], int]:
    """
    Implementa o algoritmo de Gradiente Descendente para encontrar o mínimo de uma função.

    Parameters:
        x0: ponto inicial (lista, tupla ou np.ndarray).
        cost_fn: função de custo a ser minimizada.
        grad_cost_fn: gradiente da função de custo.
        alpha: taxa de aprendizado.
        max_iter: número máximo de iterações.
        tolerance: tolerância do critério de parada.
        stop_criterium:
            1 = baseado no gradiente,
            2 = baseado na mudança de x,
            3 = baseado na mudança da função de custo.

    Returns:
        x_values: list[np.ndarray]
        costs: list[float]
        num_iter: int
    """

    num_iter = 0
    xk = np.array(x0, dtype=np.float64)
    cost = cost_fn(xk)
    x_values = [xk.copy()]
    costs = [cost]
    stop_condition = False

    while num_iter < max_iter:
        # Calcular o gradiente no ponto Xk
        grad_xk = grad_cost_fn(xk)

        # Atualizar o ponto Xk
        xk = xk - alpha * grad_xk

        # Calcular o custo
        cost = cost_fn(xk)

        # Aumentar o iterador
        num_iter += 1

        # Critério de parada
        match stopping_criterion:
            case 1:
                if np.abs(grad_xk).max() <= tolerance:
                    stop_condition = True
            case 2:
                if np.abs(xk - x_values[-1]).max() < tolerance:
                    stop_condition = True
            case 3:
                if np.abs(cost - costs[-1]) < tolerance:
                    stop_condition = True
            case _:
                raise ValueError("Critério de parada inválido. Use 1, 2 ou 3.")

        # Armazenar o ponto e o custo
        x_values.append(xk.copy())
        costs.append(cost)

        if stop_condition:
            break

    return x_values, costs, num_iter


if __name__ == "__main__":
    def J(x: np.ndarray) -> float:
        return (x[0] - 3)**2 + (x[1] + 2)**2

    def grad_J(x: np.ndarray) -> np.ndarray:
        # gradiente: [2(x0 - 3), 2(x1 + 2)]
        return np.array([2 * (x[0] - 3), 2 * (x[1] + 2)], dtype=np.float64)

    x0 = [0.0, 0.0]
    x_values, costs, num_iter = gradient_descendent(
        x0, J, grad_J, alpha=0.1, max_iter=100, tolerance=1e-6
    )

    print(f"Ponto mínimo encontrado: {x_values[-1]}")
    print(f"Custo mínimo: {costs[-1]}")
    print(f"Número de iterações: {num_iter}")
