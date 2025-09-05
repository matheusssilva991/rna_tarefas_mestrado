import numpy as np
import plotly.graph_objects as go
from typing import Callable, List

def plot_function(f, x_values, costs, title="Superfície da Função de Custo + Pontos Encontrados"):
    x_values = np.array(x_values)
    costs = np.array(costs)

    x1_data = np.linspace(min(x_values[:,0])-3, max(x_values[:,0])+3, 100)
    x2_data = np.linspace(min(x_values[:,1])-3, max(x_values[:,1])+3, 100)
    X1, X2 = np.meshgrid(x1_data, x2_data)

    Y = np.array([[f(np.array([x1, x2])) for x1 in x1_data] for x2 in x2_data])

    fig = go.Figure()

    # Superfície
    fig.add_trace(go.Surface(
        x=X1,
        y=X2,
        z=Y,
        colorscale="Viridis",
        opacity=0.8
    ))

    # Pontos encontrados
    fig.add_trace(go.Scatter3d(
        x=x_values[:, 0],
        y=x_values[:, 1],
        z=costs,
        mode="markers+text",
        text=[f"P{i}" for i in range(len(x_values))],
        textposition="top center",
        marker=dict(size=6, color="red", symbol="circle")
    ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="J(x)",
        ),
        title=title,
        autosize=True,
    )

    fig.show()

def plot_six_hump_camel(
    f: Callable[[np.ndarray], float],
    x_values: List[np.ndarray],
    title: str = "Six-Hump Camel Function Optimization Path"
):
    # Preparação dos Dados
    x_values_arr = np.array(x_values)
    costs = np.array([f(x) for x in x_values])

    # Malha para a superfície
    x1_data = np.linspace(-2.5, 2.5, 100)
    x2_data = np.linspace(-1.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1_data, x2_data)
    Y = np.array([[f(np.array([x1, x2])) for x1 in x1_data] for x2 in x2_data])

    fig = go.Figure()

    # Superfície com Contornos
    fig.add_trace(go.Surface(
        x=X1, y=X2, z=Y,
        colorscale="viridis",
        opacity=0.8,
        contours_z=dict(
            show=True, usecolormap=True,
            highlightcolor="limegreen", project_z=True
        ),
        name='Superfície',
        showlegend=True
    ))

    # Caminho da Otimização
    fig.add_trace(go.Scatter3d(
        x=x_values_arr[:, 0],
        y=x_values_arr[:, 1],
        z=costs,
        mode='lines',
        line=dict(color='magenta', width=4),
        name='Caminho da Otimização'
    ))

    # Ponto Inicial
    fig.add_trace(go.Scatter3d(
        x=[x_values_arr[0, 0]],
        y=[x_values_arr[0, 1]],
        z=[costs[0]],
        mode='markers',
        marker=dict(size=8, color='cyan', symbol='diamond'),
        name='Início (P0)'
    ))
    # Ponto Final
    fig.add_trace(go.Scatter3d(
        x=[x_values_arr[-1, 0]],
        y=[x_values_arr[-1, 1]],
        z=[costs[-1]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name=f'Fim (P{len(x_values)-1})'
    ))

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        scene=dict(
            xaxis=dict(title='x₁', backgroundcolor="rgba(0, 0, 0,0)"),
            yaxis=dict(title='x₂', backgroundcolor="rgba(0, 0, 0,0)"),
            zaxis=dict(title='Custo f(x)', backgroundcolor="rgba(0, 0, 0,0)"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99)
    )

    fig.show()
