import numpy as np
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath("./libs"))
sys.path.append(os.path.abspath("./utils"))

from levenberg_marquadt import levenberg_marquadt
from normalize import  MinMaxNormalizer
from loss_fn_tarefa5 import make_mse_loss_for_network
from network import network_forward
from activations_fn import tanh_derivative


def make_residuals_fn(X1: np.ndarray, X2: np.ndarray, Y: np.ndarray, n_neurons=2, activation_fn=np.tanh):
    def residuals_fn(weights_flat) -> np.ndarray:
        # Reconstruir os pesos para cada neurônio
        neurons_weights = []
        idx = 0

        # Reconstruir pesos da camada oculta
        for i in range(n_neurons):
            w_hidden = weights_flat[idx:idx+3]  # 3 pesos para cada neurônio oculto
            neurons_weights.append(w_hidden)
            idx += 3

        # Reconstruir pesos da camada de saída
        w_output = weights_flat[idx:]
        neurons_weights.append(w_output)

        y_hat = network_forward(X1, X2, neurons_weights=neurons_weights, activation_fn=activation_fn)
        return Y - y_hat

    return residuals_fn


def make_jacobian_fn(X1, X2, n_neurons=2, activation_fn=np.tanh, activation_deriv=tanh_derivative):
    """
    Cria uma função que calcula a Jacobiana da rede neural.
    """

    def jacobian_fn(weights_flat):
        # Reconstruir os pesos para cada neurônio
        neurons_weights = []
        idx = 0

        # Reconstruir pesos da camada oculta
        for i in range(n_neurons):
            w_hidden = weights_flat[idx:idx+3]  # 3 pesos para cada neurônio oculto
            neurons_weights.append(w_hidden)
            idx += 3

        # Reconstruir pesos da camada de saída
        w_output = weights_flat[idx:]
        neurons_weights.append(w_output)

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
            v_j = w_out[j]                     # peso da saída para neurônio j
            dz_hidden = activation_deriv(hidden_z[j])  # derivada da ativação do oculto

            for i in range(X.shape[1]):        # x1, x2, bias
                if i < 2:
                    J[:, idx] = -dy_dz_out * v_j * dz_hidden * X[:, i]
                else:  # bias da camada oculta
                    J[:, idx] = -dy_dz_out * v_j * dz_hidden
                idx += 1

        # Derivadas em relação aos pesos da camada de saída
        for j in range(num_neurons):
            J[:, idx] = -dy_dz_out * hidden_h[:, j]
            idx += 1

        # Bias da saída
        J[:, idx] = -dy_dz_out

        return J

    return jacobian_fn


def unflatten_weights(weights_flat, n_inputs, n_hidden):
    neurons_weights = []
    idx = 0
    for _ in range(n_hidden):
        w_hidden = weights_flat[idx:idx + (n_inputs + 1)]
        neurons_weights.append(w_hidden)
        idx += n_inputs + 1
    w_output = weights_flat[idx:]
    neurons_weights.append(w_output)
    return neurons_weights



def main():
    # Carregar dados
    df = pd.read_excel('./data/Trabalho4dados.xlsx')

    # Separar features e target
    features = df[['x1', 'x2']]
    y = df['y']

    # Criar os objetos para Padronização
    scaler = MinMaxNormalizer(-1, 1)
    scaler_y = MinMaxNormalizer(-1, 1)

    # Ajusta os padronizadores aos dados
    scaler.fit(features)
    scaler_y.fit(y.to_frame())

    # Padroniza os dados
    scaled_features = scaler.normalize(features)
    scaled_y = scaler_y.normalize(y.to_frame()).squeeze()

    # Definir parâmetros da rede neural
    n_neurons = 2
    n_iterations = 10000
    tolerance = 1e-6
    alpha = 1e-3

    # Inicializar listas para armazenar os pesos
    neurons_weights = []
    neurons_raw_weights = []

    # Inicializar pesos para cada neurônio (2 ocultos + 1 de saída)
    # Cada neurônio oculto tem 3 pesos (x1, x2, bias)
    hidden_weights = [np.random.randn(3) for _ in range(n_neurons)]
    # Neurônio de saída tem n_neurons + 1 pesos (saídas dos neurônios ocultos + bias)
    output_weights = np.random.randn(n_neurons + 1)

    # Combinar todos os pesos em uma lista
    initial_weights_list = hidden_weights + [output_weights]

    # Preparar dados
    x1_scaled = scaled_features['x1'].values
    x2_scaled = scaled_features['x2'].values
    y_scaled = scaled_y.values

    # Função de custo e gradiente
    loss_function, _ = make_mse_loss_for_network(x1_scaled, x2_scaled, y_scaled, activation_fn=np.tanh)

    # Função de resíduos e jacobiana
    residuals_fn = make_residuals_fn(
        x1_scaled,
        x2_scaled,
        y_scaled,
        n_neurons=n_neurons,
        activation_fn=np.tanh
    )
    jacobian_fn = make_jacobian_fn(
        x1_scaled,
        x2_scaled,
        n_neurons=n_neurons,
        activation_fn=np.tanh,
        activation_deriv=tanh_derivative
    )

    # Concatenar todos os pesos em um único vetor para o Levenberg-Marquardt
    initial_weights_flat = np.concatenate([w.flatten() for w in initial_weights_list])

    print(f"Treinando rede com {n_neurons} neurônios na camada oculta...")

    # Treinar com Levenberg-Marquardt
    weights_flat, losses, n_iters = levenberg_marquadt(
        initial_weights_flat, residuals_fn, loss_function, jacobian_fn,
        alpha=alpha, alpha_variability=10, max_iter=n_iterations,
        tolerance=tolerance, stopping_criteria=[1, 3]
    )

    # Usar os pesos finais
    final_weights_flat = weights_flat[-1]

    # Reconstruir os pesos para cada neurônio
    idx = 0
    for i in range(n_neurons):
        w_hidden = final_weights_flat[idx:idx+3]
        neurons_raw_weights.append(w_hidden)
        idx += 3

    w_output = final_weights_flat[idx:]
    neurons_raw_weights.append(w_output)

    # Armazenar os pesos normalizados para uso na predição
    neurons_weights = neurons_raw_weights.copy()

    # Fazer previsão usando a rede treinada
    y_hat_scaled = network_forward(
        x1_scaled,
        x2_scaled,
        neurons_weights=neurons_weights,
        activation_fn=np.tanh
    )

    # Desnormalizar as previsões
    # Convertendo para DataFrame com o mesmo índice usado no treinamento
    y_hat_df = pd.DataFrame(y_hat_scaled.reshape(-1, 1), index=y.index)
    y_hat = scaler_y.denormalize(y_hat_df).values.flatten()

    # Calcular métricas
    mse_final = np.mean((y - y_hat) ** 2)
    rmse_final = np.sqrt(mse_final)
    mae_final = np.mean(np.abs(y - y_hat))

    # Preparar resultados para exibição
    dict_results = {
        'Feature_Set': "MinMax(-1,1)",
        'Loss_Function': "MSE",
        'Initial_Weights': str([f"N{i+1}:{[f'{w:.3f}' for w in weights]}" for i, weights in enumerate([*hidden_weights, output_weights])]),
        'Final_Weights': str([f"N{i+1}:{[f'{w:.3f}' for w in weights]}" for i, weights in enumerate(neurons_weights)]),
        'Final_Loss': losses[-1],
        'MSE_Final': mse_final,
        'RMSE_Final': rmse_final,
        'MAE_Final': mae_final,
        'Iterations': n_iters
    }

    # Exibir resultados
    print("Feature Set:", dict_results['Feature_Set'])
    print("Loss Function:", dict_results['Loss_Function'])
    print("Initial Weights:", dict_results['Initial_Weights'])
    print("Final Weights:", dict_results['Final_Weights'])
    print(f"Final Loss: {dict_results['Final_Loss']:.6f}")
    print(f"MSE Final: {dict_results['MSE_Final']:.6f}")
    print(f"RMSE Final: {dict_results['RMSE_Final']:.6f}")
    print(f"MAE Final: {dict_results['MAE_Final']:.6f}")
    print("Iterations:", dict_results['Iterations'])


if __name__ == "__main__":
    main()
