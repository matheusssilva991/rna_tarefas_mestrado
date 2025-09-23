import numpy as np
import plotly.graph_objects as go

def plot_best_by_loss(
    df_result,
    loss_name: str,
    df,
    features_normalized,
    features_standardized,
    title_prefix: str | None = None,
    show_original_points: bool = False
):
    """
    Filtra df_result pelo loss_name (MSE | RMSE | MAE), seleciona o melhor experimento,
    gera predições e plota a superfície original com os pontos preditos.

    Adaptado para a tarefa 3 com o algoritmo Levenberg-Marquardt.
    """
    # Normalizar o nome do loss
    loss_key = loss_name.strip().upper()
    valid = {"MSE", "MAE", "RMSE"}
    if loss_key not in valid:
        raise ValueError(f"Loss inválido: {loss_name}. Use um de {sorted(valid)}")

    # Filtrar e pegar menor loss
    metric_col = f"{loss_key}_Final"
    best_df = df_result.copy()
    if best_df.empty:
        raise ValueError(f"Nenhum resultado encontrado para {loss_key}")

    best_row = best_df.loc[best_df[metric_col].idxmin()]
    best_w = best_row["Final_Weights"]
    feature_set_used = best_row["Feature_Set"]

    print(f"Melhor resultado {loss_key}:")
    print(f"Feature Set: {feature_set_used}")
    print(f"Initial Weights: {best_row['Initial_Weights']}")
    print(f"Final Loss: {best_row['Final_Loss']}")
    print(f"Pesos: {best_w}")

    # Função de predição com os melhores pesos
    def make_function(w):
        def f(x, y):
            if np.isscalar(x) and np.isscalar(y):
                return w @ np.array([x, y, 1])
            else:
                return np.array([w @ np.array([xi, yi, 1]) for xi, yi in zip(x, y)])
        return f

    f_pred = make_function(best_w)

    # Selecionar os dados corretos para predição
    x_data_orig = df["x"].values
    y_data_orig = df["y"].values

    # Predições (aplicando potências para manter consistência com o código existente)
    predictions = f_pred(x_data_orig ** 3, y_data_orig ** 2)

    # Superfície original
    df_pivot = df.pivot(index="y", columns="x", values="z")
    x_axis = df_pivot.columns.values
    y_axis = df_pivot.index.values
    z_grid = df_pivot.values

    fig = go.Figure()

    # Superfície original
    fig.add_trace(go.Surface(
        x=x_axis,
        y=y_axis,
        z=z_grid,
        colorscale="Viridis",
        opacity=0.7,
        name="Dados Originais"
    ))

    # Pontos originais
    if show_original_points:
        fig.add_trace(go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=dict(size=4, color="red", symbol="circle"),
            name="Pontos Originais"
        ))

    # Pontos preditos
    fig.add_trace(go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=predictions,
        mode="markers",
        marker=dict(size=5, color="blue", symbol="circle"),
        name="Predições"
    ))

    # Métricas
    z_true = df["z"].values
    mse = float(np.mean((z_true - predictions) ** 2))
    mae = float(np.mean(np.abs(z_true - predictions)))
    rmse = float(np.sqrt(mse))

    title_main = title_prefix or "Ajuste de Curva"
    fig.update_layout(
        title=f"{title_main} - {loss_key} | Feature: {feature_set_used} | Loss: {best_row['Final_Loss']:.6f}",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(x=0.02, y=0.98)
    )
    fig.show()

    return {
        "loss": loss_key,
        "best_row": best_row,
        "weights": best_w,
        "feature_set_used": feature_set_used,
        "predictions": np.array(predictions),
        "metrics": {"MSE": mse, "MAE": mae, "RMSE": rmse},
        "fig": fig
    }
