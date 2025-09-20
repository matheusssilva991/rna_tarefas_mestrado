import numpy as np
from typing import Callable, Sequence, Tuple, Union


def gradient_descendent(
    w0: Sequence[float | int] | np.ndarray,
    cost_fn: Callable[[np.ndarray], float],
    grad_cost_fn: Callable[[np.ndarray], np.ndarray],
    learning_rate: float,
    max_iter: int,
    tolerance: float,
    stopping_criteria: Union[int, list[int]] = [1, 3],
) -> Tuple[list[np.ndarray], list[float], int]:
    """
    Implementa o algoritmo de Gradiente Descendente para encontrar o mínimo de uma função.

    Parameters:
        w0: ponto inicial (lista, tupla ou np.ndarray).
        cost_fn: função de custo a ser minimizada.
        grad_cost_fn: gradiente da função de custo.
        learning_rate: taxa de aprendizado.
        max_iter: número máximo de iterações.
        tolerance: tolerância do critério de parada.
        stopping_criteria:
            1 = baseado no gradiente,
            2 = baseado na mudança de w,
            3 = baseado na mudança da função de custo.
            Pode ser int ou lista de int (para múltiplos critérios).

    Returns:
        w_values: list[np.ndarray]
        costs: list[float]
        num_iter: int
    """

    # Permitir que o usuário passe um único int
    if isinstance(stopping_criteria, int):
        stopping_criteria = [stopping_criteria]

    num_iter = 0
    wk = np.array(w0, dtype=np.float64)
    cost = cost_fn(wk)
    w_values = [wk.copy()]
    costs = [cost]

    while num_iter < max_iter:
        # Calcular gradiente
        grad_wk = grad_cost_fn(wk)

        # Atualizar pesos
        wk = wk - learning_rate * grad_wk

        # Calcular novo custo
        cost = cost_fn(wk)

        # Iteração
        num_iter += 1

        # Critérios de parada
        stop_condition = False
        if 1 in stopping_criteria and np.abs(grad_wk).max() <= tolerance:
            stop_condition = True
        if 2 in stopping_criteria and np.abs(wk - w_values[-1]).max() < tolerance:
            stop_condition = True
        if 3 in stopping_criteria and np.abs(cost - costs[-1]) < tolerance:
            stop_condition = True

        # Armazenar histórico
        w_values.append(wk.copy())
        costs.append(cost)

        if stop_condition:
            break

    return w_values, costs, num_iter


if __name__ == "__main__":
    def J(w: np.ndarray) -> float:
        return (w[0] - 3)**2 + (w[1] + 2)**2

    def grad_J(w: np.ndarray) -> np.ndarray:
        return np.array([2 * (w[0] - 3), 2 * (w[1] + 2)], dtype=np.float64)

    w0 = [0.0, 0.0]
    w_values, costs, num_iter = gradient_descendent(
        w0, J, grad_J,
        learning_rate=0.1,
        max_iter=100,
        tolerance=1e-6,
        stopping_criteria=[1, 3]   # <-- usa múltiplos critérios
    )

    print(f"Ponto mínimo encontrado: {w_values[-1]}")
    print(f"Custo mínimo: {costs[-1]}")
    print(f"Número de iterações: {num_iter}")
