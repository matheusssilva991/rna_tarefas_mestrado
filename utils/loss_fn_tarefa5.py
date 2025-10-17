import numpy as np
import sys
import os

sys.path.append(os.path.abspath("./libs"))
from network import hidden_forward


def tanh(x): return np.tanh(x)
def dtanh(x): return 1 - np.tanh(x)**2

def make_mse_loss_for_network(X1, X2, Y, n_neurons=2, activation_fn=np.tanh):
    def loss_fn(weights_flat):
        # Reconstruir os pesos para cada neurônio
        neurons_weights = []
        idx = 0

        # Reconstruir pesos da camada oculta
        for i in range(n_neurons):
            w_hidden = weights_flat[idx:idx+3]  # 3 pesos para cada neurônio oculto (x1, x2, bias)
            neurons_weights.append(w_hidden)
            idx += 3

        # Reconstruir pesos da camada de saída
        w_output = weights_flat[idx:]  # Resto dos pesos para a saída
        neurons_weights.append(w_output)

        # Calcular a saída da rede
        y_hat = hidden_forward(X1, X2, neurons_weights=neurons_weights, activation_fn=activation_fn)

        # Calcular MSE
        mse = np.mean((Y - y_hat) ** 2)
        return mse

    def grad_fn(weights_flat):
        # Gradiente numérico (simplificado)
        epsilon = 1e-8
        grad = np.zeros_like(weights_flat)

        for i in range(len(weights_flat)):
            weights_plus = weights_flat.copy()
            weights_plus[i] += epsilon

            weights_minus = weights_flat.copy()
            weights_minus[i] -= epsilon

            grad[i] = (loss_fn(weights_plus) - loss_fn(weights_minus)) / (2 * epsilon)

        return grad

    return loss_fn, grad_fn
