import jax.numpy as jnp
from jax import grad
from typing import Callable, Sequence, Tuple


def gradient_descendent(
    x0: Sequence[float | int] | jnp.ndarray,
    f: Callable[[jnp.ndarray], jnp.ndarray],
    alfa: float,
    max_iter: int,
    tolerance: float,
    stop_criterium: int = 1,
) -> Tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """
    Implementa o algoritmo de Gradiente Descendente para encontrar o mínimo de uma função.

    Parameters:
        x0: ponto inicial (lista, tupla ou jnp.ndarray).
        f: função a ser minimizada, que recebe jnp.ndarray e retorna escalar.
        alfa: taxa de aprendizado.
        max_iter: número máximo de iterações.
        tolerance: tolerância do critério de parada.
        stop_criterium:
            1 = baseado no gradiente,
            2 = baseado na mudança de x,
            3 = baseado na mudança da função de custo.

    Returns:
        xk (jnp.ndarray): ponto onde a função atinge (aprox.) o valor mínimo.
    """

    iter = 0
    xk = jnp.array(x0, dtype=jnp.float32)
    cost = f(xk)
    dfdx = grad(f)
    x_values = [xk.copy()]
    costs = [cost]

    while iter <= max_iter:
        # Calcular o gradiente no ponto Xk
        grad_xk = dfdx(xk)

        # Atualizar o ponto Xk
        xk = xk - alfa * grad_xk

        # Calcular o custo
        cost = f(xk)

        # Aumentar o iterador
        iter += 1

        # Critério de parada
        match stop_criterium:
            case 1:
                if jnp.abs(grad_xk).max() <= tolerance:
                    x_values.append(xk.copy())
                    costs.append(cost)
                    break
            case 2:
                if jnp.abs(xk - x_values[-1]).max() < tolerance:
                    x_values.append(xk.copy())
                    costs.append(cost)
                    break
            case 3:
                if jnp.abs(cost - costs[-1]) < tolerance:
                    x_values.append(xk.copy())
                    costs.append(cost)
                    break
            case _:
                raise ValueError("Critério de parada inválido. Use 1, 2 ou 3.")

        # Armazenar o ponto e o custo
        x_values.append(xk.copy())
        costs.append(cost)

    return x_values, costs


if __name__ == "__main__":
    def J(x: jnp.ndarray):
        return (x[0] - 3)**2 + (x[1] + 2)**2

    x0 = [0.0, 0.0]
    x_values, costs = gradient_descendent(x0, J, alfa=0.1, max_iter=100, tolerance=1e-6)
    print(f"Ponto mínimo encontrado: {x_values[-1]}")
