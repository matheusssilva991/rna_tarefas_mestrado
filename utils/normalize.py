import numpy as np

class MinMaxNormalizer:
    def __init__(self, n_min: float = 0.0, n_max: float = 1.0):
        self.n_min = n_min
        self.n_max = n_max
        self.x_min = None
        self.x_max = None

    def fit(self, X: np.ndarray):
        """
        Armazena min e max por feature.
        X: matriz (n_amostras, n_features)
        """
        self.x_min = np.min(X, axis=0)
        self.x_max = np.max(X, axis=0)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        if self.x_min is None or self.x_max is None:
            raise ValueError("Você deve chamar 'fit' antes de normalizar.")
        return self.n_min + ((X - self.x_min) * (self.n_max - self.n_min)) / (self.x_max - self.x_min)

    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        if self.x_min is None or self.x_max is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar.")
        return self.x_min + ((X_norm - self.n_min) * (self.x_max - self.x_min)) / (self.n_max - self.n_min)

    def desnormalize_weights(self, w: np.ndarray) -> np.ndarray:
        """
        Converte pesos de um modelo linear treinado em dados Min-Max normalizados
        para a escala original.
        w = [w1, w2, ..., bias]
        """
        if self.x_min is None or self.x_max is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar pesos.")

        w_no_bias = w[:-1]
        b = w[-1]

        # escala por feature
        scale = (self.x_max - self.x_min) / (self.n_max - self.n_min)

        w_orig = w_no_bias / scale
        b_orig = b - np.sum(w_no_bias * self.x_min / scale)

        return np.concatenate([w_orig, [b_orig]])

    def desnormalize_polynomial_weights(self, w: np.ndarray, powers: list) -> np.ndarray:
        """
        Desnormaliza pesos de um modelo polinomial treinado em dados Min-Max normalizados.

        Parâmetros:
        w: np.ndarray - Pesos [w₁, w₂, ..., bias]
        powers: list - Lista com as potências correspondentes a cada peso [p₁, p₂, ...]

        Exemplo para um modelo w[0]*x^3 + w[1]*y^2 + w[2]:
        powers = [3, 2]

        Retorna:
        np.ndarray - Pesos desnormalizados
        """
        if self.x_min is None or self.x_max is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar pesos.")

        if len(powers) != len(w) - 1:
            raise ValueError(f"Número de potências ({len(powers)}) deve ser igual ao número de pesos - 1 ({len(w)-1})")

        w_new = np.zeros_like(w)

        # Processar cada peso (exceto o bias)
        for i in range(len(w) - 1):
            # Escala para a feature
            scale = (self.x_max[i] - self.x_min[i]) / (self.n_max - self.n_min)
            # Aplicar escala elevada à potência
            w_new[i] = w[i] / (scale ** powers[i])

        # O bias é mais complexo, esta é uma aproximação
        # Para modelo polinomial, o ajuste exato do bias é mais complicado
        w_new[-1] = w[-1]

        return w_new


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        """
        Armazena média e desvio padrão por feature.
        X: matriz (n_amostras, n_features)
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Você deve chamar 'fit' antes de normalizar.")
        return (X - self.mean) / self.std

    def denormalize(self, X_std: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar.")
        return X_std * self.std + self.mean

    def desnormalize_weights(self, w: np.ndarray) -> np.ndarray:
        """
        Converte pesos de um modelo linear treinado em dados padronizados
        para a escala original.
        w = [w1, w2, ..., bias]
        """
        if self.mean is None or self.std is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar pesos.")

        w_no_bias = w[:-1]
        b = w[-1]

        w_orig = w_no_bias / self.std
        b_orig = b - np.sum((w_no_bias * self.mean) / self.std)

        return np.concatenate([w_orig, [b_orig]])

    def desnormalize_polynomial_weights(self, w: np.ndarray, powers: list) -> np.ndarray:
        """
        Desnormaliza pesos de um modelo polinomial treinado em dados padronizados.

        Parâmetros:
        w: np.ndarray - Pesos [w₁, w₂, ..., bias]
        powers: list - Lista com as potências correspondentes a cada peso [p₁, p₂, ...]

        Exemplo para um modelo w[0]*x^3 + w[1]*y^2 + w[2]:
        powers = [3, 2]

        Retorna:
        np.ndarray - Pesos desnormalizados
        """
        if self.mean is None or self.std is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar pesos.")

        if len(powers) != len(w) - 1:
            raise ValueError(f"Número de potências ({len(powers)}) deve ser igual ao número de pesos - 1 ({len(w)-1})")

        w_new = np.zeros_like(w)

        # Processar cada peso (exceto o bias)
        for i in range(len(w) - 1):
            # Aplicar escala elevada à potência
            w_new[i] = w[i] / (self.std[i] ** powers[i])

        # O bias é mais complexo, esta é uma aproximação para polinômios
        w_new[-1] = w[-1]

        return w_new
