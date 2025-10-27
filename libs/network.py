import os
import sys
import numpy as np

sys.path.append(os.path.abspath("./libs"))
sys.path.append(os.path.abspath("./utils"))

from activations_fn import tanh_derivative
from levenberg_marquadt import levenberg_marquadt


def neuron(*features, weights=None, activation_fn=np.tanh):
    """
    Neurônio simples com múltiplas entradas e um termo de bias.
    Parâmetros
    ----------
    *features : arrays
        Features de entrada (x1, x2, ..., xn)
    weights : np.ndarray, optional
        Pesos do neurônio (incluindo o peso do bias como o último elemento).
        Se None, os pesos serão inicializados aleatoriamente.
    activation_fn : callable
        Função de ativação (ex: np.tanh)
    Retorna
    -------
    y : np.ndarray
        Saída do neurônio.
    """

    # Converter todas as features para arrays 1D
    features_arrays = [np.atleast_1d(f) for f in features]

    # Verificar que todas as features têm o mesmo número de amostras
    if not all(len(f) == len(features_arrays[0]) for f in features_arrays):
        raise ValueError("Todas as features devem ter o mesmo número de exemplos.")

    # Empilhar features em uma matriz 2D
    X = np.column_stack(features_arrays)

    # Criar a matriz X' com o termo de bias
    X_bias = np.column_stack([X, np.ones(X.shape[0])])

    # Inicializar pesos se não fornecidos
    if weights is None:
        weights = np.random.randn(X_bias.shape[1])

    # Verificar que o número de pesos é compatível
    if len(weights) != X_bias.shape[1]:
        raise ValueError(f"Número de pesos ({len(weights)}) deve ser igual ao número de features + 1 para o bias ({X_bias.shape[1]}).")

    # Calcular potencial de ativação
    activation_potentials = X_bias @ weights

    # Aplicar função de ativação
    y = activation_fn(activation_potentials)

    # Retornar escalar se a entrada era um único exemplo
    if X.shape[0] == 1:
        return y.item()
    return y


def hidden_forward(*features, neurons_weights, activation_fn=np.tanh):
    """
    Rede neural simples com uma camada oculta de N neurônios e uma camada de saída.

    Parâmetros
    ----------
    *features : arrays
        Features de entrada (x1, x2, ..., xn)
    neurons_weights : list of np.ndarray
        Lista com pesos dos neurônios:
            [w1, w2, ..., w_hiddenN, w_output]
    activation_fn : callable
        Função de ativação (ex: np.tanh)

    Retorna
    -------
    y_hat : np.ndarray
        Saída final da rede.
    """

    hidden_outputs = [
        neuron(*features, weights=weights, activation_fn=activation_fn)
        for weights in neurons_weights[:-1]
    ]
    # Passa as saídas ocultas como features da camada de saída
    y_hat = neuron(*hidden_outputs, weights=neurons_weights[-1], activation_fn=activation_fn)

    return y_hat


def make_residuals_fn(X1: np.ndarray, X2: np.ndarray, Y: np.ndarray, n_neurons=2, activation_fn=np.tanh):
    def residuals_fn(weights_flat) -> np.ndarray:
        # Reconstruir os pesos para cada neurônio
        neurons_weights = unflatten_weights(weights_flat, n_inputs=2, n_neurons=n_neurons)

        y_hat = hidden_forward(X1, X2, neurons_weights=neurons_weights, activation_fn=activation_fn)
        return Y - y_hat

    return residuals_fn


def make_mse_loss_for_network(X1, X2, Y, n_neurons=2, activation_fn=np.tanh):
    residuals_fn = make_residuals_fn(X1, X2, Y, n_neurons=n_neurons, activation_fn=activation_fn)

    def loss_fn(weights_flat):
        residuals = residuals_fn(weights_flat)
        return np.mean(residuals**2)

    return loss_fn


def make_jacobian_fn(X1, X2, n_neurons=2, activation_fn=np.tanh, activation_deriv=tanh_derivative):
    """
    Cria uma função que calcula a Jacobiana da rede neural.
    """

    def jacobian_fn(weights_flat):
        # Reconstruir os pesos para cada neurônio
        neurons_weights = unflatten_weights(weights_flat, n_inputs=2, n_neurons=n_neurons)

        # Preparar entradas
        X1_ = np.atleast_1d(X1)
        X2_ = np.atleast_1d(X2)
        X = np.stack([X1_, X2_, np.ones_like(X1_)], axis=1)  # (n_amostras, 3)
        n_samples = X.shape[0]

        num_neurons = len(neurons_weights) - 1  # todos menos o da saída
        w_out = neurons_weights[-1]            # pesos da camada de saída
        hidden_z = []                          # potenciais da camada oculta
        hidden_h = []                          # saídas da camada oculta

        # Calcular saídas da camada oculta
        for j in range(num_neurons):
            z_j = X @ neurons_weights[j]       # Potencial do neurônio j
            h_j = activation_fn(z_j)           # Saída do neurônio j
            hidden_z.append(z_j)
            hidden_h.append(h_j)

        # Empilhar saídas ocultas e calcular saída final
        hidden_h = np.stack(hidden_h, axis=1)
        z_out = np.column_stack([hidden_h, np.ones(n_samples)]) @ w_out

        # Inicializar Jacobiana
        total_params = num_neurons * X.shape[1] + len(w_out)
        J = np.zeros((n_samples, total_params))

        # Calcular derivadas parciais
        dy_dz_out = activation_deriv(z_out)
        idx = 0

        # Derivadas em relação aos pesos da camada oculta
        for j in range(num_neurons):
            output_weight_for_neuron_j = w_out[j]      # peso da saída para neurônio j (w_out_j)
            dz_hidden = activation_deriv(hidden_z[j])  # derivada da ativação do oculto (dz_j/dh_j)

            for i in range(X.shape[1]):  # x1, x2, bias
                if i < 2:
                    J[:, idx] = -dy_dz_out * output_weight_for_neuron_j * dz_hidden * X[:, i]
                else:  # bias da camada oculta
                    J[:, idx] = -dy_dz_out * output_weight_for_neuron_j * dz_hidden
                idx += 1

        # Derivadas em relação aos pesos da camada de saída
        for j in range(num_neurons):
            J[:, idx] = -dy_dz_out * hidden_h[:, j]
            idx += 1

        # Bias da saída
        J[:, idx] = -dy_dz_out

        return J

    return jacobian_fn


def unflatten_weights(weights_flat, n_inputs, n_neurons):
    neurons_weights = [] # Lista para armazenar os pesos de cada neurônio
    idx = 0 # Índice para rastrear a posição atual em weights_flat

    # Reconstruir pesos da camada oculta
    for _ in range(n_neurons):
        # Cada neurônio tem (n_inputs + 1) pesos (incluindo bias)
        w_hidden = weights_flat[idx:idx + (n_inputs + 1)]

        # Adicionar pesos do neurônio à lista
        neurons_weights.append(w_hidden)

        # Atualizar índice
        idx += n_inputs + 1

    # Reconstruir pesos da camada de saída
    w_output = weights_flat[idx:]

    # Adicionar pesos da camada de saída à lista
    neurons_weights.append(w_output)

    return neurons_weights


def optimize_network_weights(X1, X2, y, **kwargs):
    n_neurons = kwargs.get("n_neurons", 10)
    activation_fn = kwargs.get("activation_fn", np.tanh)
    n_iterations = kwargs.get("n_iterations", 1000)
    tolerance = kwargs.get("tolerance", 1e-6)
    alpha = kwargs.get("alpha", 1e-3)
    initial_weights = kwargs.get(
        "initial_weights",
        np.random.uniform(-1, 1, size=(n_neurons * 3,)),
    )

    # Função de custo e gradiente
    loss_function = make_mse_loss_for_network(
        X1, X2, y, activation_fn=activation_fn, n_neurons=n_neurons
    )

    # Função de resíduos
    residuals_fn = make_residuals_fn(
        X1, X2, y, n_neurons=n_neurons, activation_fn=np.tanh
    )

    # Função da jacobiana
    jacobian_fn = make_jacobian_fn(
        X1,
        X2,
        n_neurons=n_neurons,
        activation_fn=np.tanh,
        activation_deriv=tanh_derivative,
    )

    # Treinar com Levenberg-Marquardt
    neurons_weights, losses, n_iters = levenberg_marquadt(
        initial_weights,
        residuals_fn,
        loss_function,
        jacobian_fn,
        alpha=alpha,
        alpha_variability=10,
        max_iter=n_iterations,
        tolerance=tolerance,
        stopping_criteria=[1, 3],
    )

    return neurons_weights, losses, n_iters


def train_network(X1, X2, y, **kwargs):
    n_neurons = kwargs.get("n_neurons", 10)
    activation_fn = kwargs.get("activation_fn", np.tanh)
    n_epochs = kwargs.get("n_epochs", 100)
    n_iterations_per_epoch = kwargs.get("n_iterations_per_epoch", 1000)
    tolerance = kwargs.get("tolerance", 1e-6)
    alpha = kwargs.get("alpha", 1e-3)

    # Inicializar pesos
    # Para n_neurons na camada oculta
    hidden_weights = [np.random.uniform(-1, 1, size=3) for _ in range(n_neurons)]
    # Para o neurônio de saída
    output_weights = np.random.uniform(-1, 1, size=n_neurons + 1)
    # Concatenar todos os pesos
    initial_weights = np.concatenate([w.flatten() for w in hidden_weights] + [output_weights.flatten()])

    all_losses = []
    total_iters = 0

    for _ in range(n_epochs):
        # Treinar a rede neural
        initial_weights, losses, n_iters = optimize_network_weights(
            X1,
            X2,
            y,
            n_neurons=n_neurons,
            activation_fn=activation_fn,
            n_iterations=n_iterations_per_epoch,
            tolerance=tolerance,
            alpha=alpha,
            initial_weights=initial_weights,
        )
        initial_weights = initial_weights[-1]

        all_losses.extend(losses)
        total_iters += n_iters

        """ if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {losses[-1]:.6f}") """

    return initial_weights, all_losses, total_iters
