import jax.numpy as jnp
from jax import grad
from typing import Callable


def gradient_descendent(
    x0: float,
    f: Callable[[float], float],
    alfa: float,
    max_iter: int,
    tolerance: float,
    stop_criterium: int = 1,
) -> float:
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
    cost = f(x0)
    cost_old = cost
    dfdx = grad(f)
    xk = x0
    xk_old = x0

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
                if jnp.abs(grad_xk) <= tolerance:
                    break
            case 2:
                # Critério de parada baseado na mudança do valor de x
                if jnp.abs(xk - xk_old) < tolerance:
                    break
            case 3:
                # Critério de parada baseado na mudança do valor da função de custo
                if jnp.abs(cost - cost_old) < tolerance:
                    break
            case _:
                raise ValueError("Critério de parada inválido. Use 1 ou 2.")

        xk_old = xk
        cost_old = cost

    return xk


if __name__ == "__main__":
    # Exemplo de uso
    def funcao_exemplo(x):
        # 2x + 3 = 0 -> x = -3/2
        return x**2 + 3*x + 2

    ponto_inicial = 1.0
    taxa_aprendizado = 0.1
    maximo_iteracoes = 100
    tolerancia = 1e-6

    ponto_minimo = gradient_descendent(ponto_inicial, funcao_exemplo, taxa_aprendizado, maximo_iteracoes, tolerancia)
    print(f"Ponto mínimo encontrado: {ponto_minimo}")
