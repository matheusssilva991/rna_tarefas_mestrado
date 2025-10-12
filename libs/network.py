import numpy as np

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


def network_forward(*features, neurons_weights, activation_fn=np.tanh):
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
