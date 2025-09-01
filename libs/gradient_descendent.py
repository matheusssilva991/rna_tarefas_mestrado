import jax.numpy as jnp
from jax import grad
from typing import Callable


def gradient_descendent(
    x0: list[float],
    f: Callable[[jnp.ndarray], float],
    alfa: float,
    max_iter: int,
    tolerance: float,
    stop_criterium: int = 1,
) -> jnp.ndarray:
    """
    Implementa o algoritmo de Gradiente Descendente para encontrar o mínimo de uma função.

    Args:
    - x0: ponto inicial (float)
    - f: função a ser minimizada (callable)
    - alfa: taxa de aprendizado (float)
    - max_iter: número máximo de iterações (int)
    - tolerance: critério de parada baseado na mudança do valor da função (float)
    Returns:
    - xk: ponto onde a função atinge o valor mínimo (float)
    """

    iter = 0
    cost = f(jnp.array(x0))
    cost_old = cost
    dfdx = grad(f)
    xk = jnp.array(x0)
    xk_old = xk.copy()

    while iter <= max_iter:
        # Calcular o vetor gradiente ponto X_k
        grad_xk = dfdx(xk)

        # Atualizar o valor de x
        xk = xk - alfa * grad_xk

        # Calcular o valor do custo
        cost = f(xk)

        # Incrementar o iterador
        iter += 1

        match stop_criterium:
            case 1:
                # Critério de parada baseado no valor do gradiente
                if jnp.abs(grad_xk).max() <= tolerance:
                    break
            case 2:
                # Critério de parada baseado na mudança do valor de x
                if jnp.abs(xk - xk_old).max() < tolerance:
                    break
            case 3:
                # Critério de parada baseado na mudança do valor da função de custo
                if jnp.abs(cost - cost_old) < tolerance:
                    break
            case _:
                raise ValueError("Critério de parada inválido. Use 1, 2 ou 3.")

        xk_old = xk.copy()
        cost_old = cost

    return xk


if __name__ == "__main__":
    # Função de exemplo: J(x, y) = (x - 3)^2 + (y + 2)^2
    def J(x):
        return (x[0] - 3)**2 + (x[1] + 2)**2

    x0 = [0.0, 0.0]
    learning_rate = 0.1
    max_iter = 100
    tolerance = 1e-6

    ponto_minimo = gradient_descendent(x0, J, learning_rate, max_iter, tolerance)
    print(f"Ponto mínimo encontrado: {ponto_minimo}")
