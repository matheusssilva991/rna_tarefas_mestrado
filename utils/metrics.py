import numpy as np

def R2_score(y_true, y_pred):
    """
    Calcula o coeficiente de determinação R² entre os valores verdadeiros e previstos.
    Parâmetros
    ----------
    y_true : np.ndarray
        Valores verdadeiros.
    y_pred : np.ndarray
        Valores previstos.
    Retorna
    -------
    r2 : float
        Coeficiente de determinação R².
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def MSE(y_true, y_pred):
    """
    Calcula o erro quadrático médio (MSE) entre os valores verdadeiros e previstos.
    Parâmetros
    ----------
    y_true : np.ndarray
        Valores verdadeiros.
    y_pred : np.ndarray
        Valores previstos.
    Retorna
    -------
    mse : float
        Erro quadrático médio.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def MAE(y_true, y_pred):
    """
    Calcula o erro absoluto médio (MAE) entre os valores verdadeiros e previstos.
    Parâmetros
    ----------
    y_true : np.ndarray
        Valores verdadeiros.
    y_pred : np.ndarray
        Valores previstos.
    Retorna
    -------
    mae : float
        Erro absoluto médio.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def RMSE(y_true, y_pred):
    """
    Calcula a raiz do erro quadrático médio (RMSE) entre os valores verdadeiros e previstos.
    Parâmetros
    ----------
    y_true : np.ndarray
        Valores verdadeiros.
    y_pred : np.ndarray
        Valores previstos.
    Retorna
    -------
    rmse : float
        Raiz do erro quadrático médio.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse
