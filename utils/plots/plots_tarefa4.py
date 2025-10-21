import plotly.graph_objects as go
import numpy as np


def plot_original_surface(x1_data, x2_data, y_data):
    fig_surface = go.Figure()

    # Superfície original (Mesh3d)
    fig_surface.add_trace(go.Mesh3d(
        x=x1_data,
        y=x2_data,
        z=y_data,
        intensity=y_data,  # Cor baseada nos valores de y
        colorscale='Viridis',
        opacity=0.8,
        name='Superfície Original',
        showscale=True
    ))

    # Pontos originais
    fig_surface.add_trace(go.Scatter3d(
        x=x1_data,
        y=x2_data,
        z=y_data,
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            symbol='circle',
            line=dict(width=1, color='darkred')
        ),
        name='Dados Originais'
    ))

    # Layout
    fig_surface.update_layout(
        title='Superfície dos Dados Originais',
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='y',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2),
            camera=dict(
                eye=dict(x=2.0, y=2.0, z=1.5),
                up=dict(x=0, y=0, z=1)
            ),
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            zaxis=dict(gridcolor='lightgray')
        ),
        width=900,
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
    )

    fig_surface.show()
    return fig_surface


def plot_predicted_surface_comparison(
    fig_surface,
    x1_data, x2_data, y_data, y_hat,
    final_denorm_weights,
    neuron_fn,
    mse_final=None, rmse_final=None, mae_final=None,
    grid_size=30,
    title='Comparação: Superfície Original vs. Prevista'
):

    # Criar grade para superfície prevista
    x1_range = np.linspace(min(x1_data), max(x1_data), grid_size)
    x2_range = np.linspace(min(x2_data), max(x2_data), grid_size)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    y_grid = np.zeros_like(x1_grid)

    # Calcular valores preditos para toda a superfície
    for i in range(x1_grid.shape[0]):
        for j in range(x1_grid.shape[1]):
            y_grid[i, j] = neuron_fn(x1_grid[i, j], x2_grid[i, j], weights=final_denorm_weights, activation_fn=np.tanh)

    # Adicionar superfície prevista à figura original
    fig_surface.add_trace(go.Surface(
        x=x1_grid, y=x2_grid, z=y_grid,
        colorscale='Reds',
        opacity=0.6,
        showscale=True,
        name='Superfície Prevista'
    ))

    # Adicionar pontos previstos
    fig_surface.add_trace(go.Scatter3d(
        x=x1_data,
        y=x2_data,
        z=y_hat,
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            symbol='diamond',
            line=dict(width=1, color='darkblue')
        ),
        name='Valores Previstos'
    ))

    # Adicionar linhas verticais para mostrar os erros
    for i in range(len(x1_data)):
        fig_surface.add_trace(go.Scatter3d(
            x=[x1_data[i], x1_data[i]],
            y=[x2_data[i], x2_data[i]],
            z=[y_data[i], y_hat[i]],
            mode='lines',
            line=dict(color='rgba(0,100,0,0.5)', width=2),
            showlegend=False
        ))

    # Atualizar título e layout
    fig_surface.update_layout(
        title=title,
        scene=dict(
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
            )
        )
    )

    # Adicionar informações sobre o modelo
    if final_denorm_weights is not None and mse_final is not None:
        pesos_info = (
            f"<b>Neurônio com tanh:</b> w=[{', '.join([f'{w:.4f}' for w in final_denorm_weights])}]<br>"
            f"MSE: {mse_final:.5f}, RMSE: {rmse_final:.5f}, MAE: {mae_final:.5f}"
        )
        fig_surface.add_annotation(
            x=0.5, y=0.02,
            xref="paper", yref="paper",
            text=pesos_info,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            borderpad=5
        )

    fig_surface.show()
    return fig_surface
