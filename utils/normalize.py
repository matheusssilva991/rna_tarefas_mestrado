import numpy as np

def min_max_normalize(x: np.ndarray, n_min: float, n_max: float) -> np.ndarray:
    """
    Normaliza os dados usando Min-Max Scaling para o intervalo [0, 1].

    Parameters:
        x: np.ndarray
            Dados a serem normalizados.
    Returns:
        np.ndarray
            Dados normalizados.
    """
    x_min = np.min(x)
    x_max = np.max(x)
    return n_min + ((x - x_min) * (n_max - n_min)) / (x_max - x_min)


def padronize(x: np.ndarray) -> np.ndarray:
    """
    Padroniza os dados para que tenham média 0 e desvio padrão 1 (Z-score normalization).

    Parameters:
        x: np.ndarray
            Dados a serem padronizados.
    Returns:
        np.ndarray
            Dados padronizados.
    """
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std
