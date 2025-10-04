import numpy as np
from typing import Callable, Sequence, Tuple, Union


def levenberg_marquadt(
    w0: Sequence[float | int] | np.ndarray,
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    cost_fn: Callable[[np.ndarray], float],
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float = 1e-3,
    alpha_variability: int = 10,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
    stopping_criteria: Union[int, list[int]] = [1, 3],
) -> Tuple[list[np.ndarray], list[float], int]:

    wk = np.array(w0, dtype=np.float64)
    num_iter = 0
    cost = cost_fn(wk)
    w_values = [wk.copy()]
    costs = [cost]
    current_alpha = alpha

    while num_iter < max_iter:
        # Calcular jacobiano e resíduos
        J = jacobian_fn(wk)
        r = residuals_fn(wk)
        n = len(r)

        # Gradiente e aproximação da Hessiana
        E_w = -(1/n) * J.T @ r
        H_w = (1/n) * J.T @ J

        cost_aux = float('inf')
        max_inner_iter = 50
        inner_iter = 0

        while cost_aux >= cost and inner_iter < max_inner_iter:
            inner_iter += 1

            # Matriz identidade
            identity_matrix = np.eye(len(wk))

            try:
                # Atualizar pesos
                delta_wk = np.linalg.inv(H_w + current_alpha * identity_matrix) @ E_w
                w_aux = wk + delta_wk

                # calcular novo custo
                cost_aux = cost_fn(w_aux)

                # Ajustar current_alpha com base na melhoria do custo
                if cost_aux >= cost:
                    current_alpha *= alpha_variability
                else:
                    current_alpha /= alpha_variability
                    cost = cost_aux
                    wk = w_aux
                    w_values.append(wk.copy())
                    costs.append(cost)
                    num_iter += 1
                    break

            except np.linalg.LinAlgError:
                current_alpha *= alpha_variability
                print("Matriz singular, aumentando alpha.")
                continue

        # Verificar critério de parada baseado no gradiente
        if 1 in stopping_criteria and np.abs(E_w).max() <= tolerance:
            # print("Convergiu pelo critério do gradiente.")
            break
        # verificar critério de parada baseado na mudança em w
        if 2 in stopping_criteria and len(w_values) > 1 and np.abs(wk - w_values[-2]).max() < tolerance:
            # print("Convergiu pelo critério da mudança em w.")
            break
        # verificar critério de parada baseado na mudança no custo
        if 3 in stopping_criteria and len(costs) > 1 and np.abs(cost - costs[-2]) < tolerance:
            # print("Convergiu pelo critério da mudança no custo.")
            break

    return w_values, costs, num_iter


if __name__ == "__main__":
    # Dados artificiais
    np.random.seed(42)
    x_data = np.linspace(-2, 2, 30)
    y_data = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x_data, y_data)
    X, Y = X.ravel(), Y.ravel()

    # Pesos verdadeiros
    true_w = np.array([2.0, -1.0, 5.0])

    # Geração do z com ruído
    Z = true_w[0] * X**3 + true_w[1] * Y**2 + true_w[2]
    Z += np.random.normal(scale=0.5, size=Z.shape)  # ruído gaussiano

    # Definição das funções
    def residuals_fn(w: np.ndarray) -> np.ndarray:
        predictions = w[0]*X**3 + w[1]*Y**2 + w[2]
        return Z - predictions

    def cost_fn(w: np.ndarray) -> float:
        r = residuals_fn(w)
        return np.mean(r**2)

    def jacobian_fn(w: np.ndarray) -> np.ndarray:
        # Derivadas parciais em relação a cada peso
        J_a = -X**3
        J_b = -Y**2
        J_c = -np.ones_like(X)
        return np.vstack([J_a, J_b, J_c]).T

    # Chamada do LM
    w0 = np.array([0.5, 0.5, 0.5])  # chute inicial
    w_values, costs, n_iter = levenberg_marquadt(
        w0, residuals_fn, cost_fn, jacobian_fn,
        alpha=1e-3, alpha_variability=10, max_iter=100
    )

    print("Pesos verdadeiros:", true_w)
    print("Pesos encontrados:", w_values[-1])
    print("Custo final:", costs[-1])
    print("Iterações:", n_iter)
