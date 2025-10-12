import numpy as np


def tanh(x):
    """Função de ativação tanh"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivada de tanh(x) = 1 - tanh(x)^2"""
    t = np.tanh(x)
    return 1 - t**2
