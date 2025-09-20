import numpy as np

class MinMaxNormalizer:
    def __init__(self, n_min: float = 0.0, n_max: float = 1.0):
        self.n_min = n_min
        self.n_max = n_max
        self.x_min = None
        self.x_max = None

    def fit(self, x: np.ndarray):
        """Armazena min e max dos dados."""
        self.x_min = np.min(x)
        self.x_max = np.max(x)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.x_min is None or self.x_max is None:
            raise ValueError("Você deve chamar 'fit' antes de normalizar.")
        return self.n_min + ((x - self.x_min) * (self.n_max - self.n_min)) / (self.x_max - self.x_min)

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        if self.x_min is None or self.x_max is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar.")
        return self.x_min + ((x_norm - self.n_min) * (self.x_max - self.x_min)) / (self.n_max - self.n_min)

    def desnormalize_weights(self, w: np.ndarray) -> np.ndarray:
        """
        w = [a, b, c] -> c é o bias
        """
        w_no_bias = w[:-1]
        b = w[-1]
        scale = (self.x_max - self.x_min) / (self.n_max - self.n_min)
        w_orig = w_no_bias * scale
        b_orig = b - np.sum(w_orig * (self.x_min - self.n_min))
        return np.concatenate([w_orig, [b_orig]])



class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = np.mean(x)
        self.std = np.std(x)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Você deve chamar 'fit' antes de normalizar.")
        return (x - self.mean) / self.std

    def denormalize(self, x_std: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Você deve chamar 'fit' antes de desnormalizar.")
        return x_std * self.std + self.mean

    def desnormalize_weights(self, w: np.ndarray) -> np.ndarray:
        """
        w = [a, b, c] -> c é o bias
        """
        w_no_bias = w[:-1]
        b = w[-1]
        w_orig = w_no_bias / self.std
        b_orig = b - np.sum(w_no_bias * self.mean / self.std)
        return np.concatenate([w_orig, [b_orig]])
