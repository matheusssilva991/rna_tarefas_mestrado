import numpy as np
from typing import Callable, Sequence, Tuple


def gradient_descendent(
    w0: Sequence[float | int] | np.ndarray,
    cost_fn: Callable[[np.ndarray], float],
    grad_cost_fn: Callable[[np.ndarray], np.ndarray],
    learning_rate: float,
    max_iter: int,
    tolerance: float,
    stopping_criterion: int = 1,
) -> Tuple[list[np.ndarray], list[float], int]:
    """
    Implementa o algoritmo de Gradiente Descendente para encontrar o mínimo de uma função.

    Parameters:
        w0: ponto inicial (lista, tupla ou np.ndarray).
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
        w_values: list[np.ndarray]
        costs: list[float]
        num_iter: int
    """

    num_iter = 0
    wk = np.array(w0, dtype=np.float64)
    cost = cost_fn(wk)
    w_values = [wk.copy()]
    costs = [cost]
    stop_condition = False

    while num_iter < max_iter:
        # Calcular o gradiente no ponto wk
        grad_wk = grad_cost_fn(wk)

        # Atualizar o ponto wk
        wk = wk - learning_rate * grad_wk

        # Calcular o custo
        cost = cost_fn(wk)

        # Aumentar o iterador
        num_iter += 1

        # Critério de parada
        match stopping_criterion:
            case 1:
                if np.abs(grad_wk).max() <= tolerance:
                    stop_condition = True
            case 2:
                if np.abs(wk - w_values[-1]).max() < tolerance:
                    stop_condition = True
            case 3:
                if np.abs(cost - costs[-1]) < tolerance:
                    stop_condition = True
            case _:
                raise ValueError("Critério de parada inválido. Use 1, 2 ou 3.")

        # Armazenar o ponto e o custo
        w_values.append(wk.copy())
        costs.append(cost)

        if stop_condition:
            break

    return w_values, costs, num_iter


if __name__ == "__main__":
    def J(w: np.ndarray) -> float:
        return (w[0] - 3)**2 + (w[1] + 2)**2

    def grad_J(w: np.ndarray) -> np.ndarray:
        # gradiente: [2(x0 - 3), 2(x1 + 2)]
        return np.array([2 * (w[0] - 3), 2 * (w[1] + 2)], dtype=np.float64)

    w0 = [0.0, 0.0]
    w_values, costs, num_iter = gradient_descendent(
        w0, J, grad_J, learning_rate=0.1, max_iter=100, tolerance=1e-6
    )

    print(f"Ponto mínimo encontrado: {w_values[-1]}")
    print(f"Custo mínimo: {costs[-1]}")
    print(f"Número de iterações: {num_iter}")
