import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata


def plot_interpolated_surface_with_original_data(x1_data, x2_data, y_data,
                                      grid_size=100,
                                      colorscale='Viridis',
                                      opacity=0.85,
                                      marker_size=5):
    """
    Plota a superfície interpolada dos dados originais e os pontos em 3D.
    """

    # Criar grade regular
    x1_grid = np.linspace(min(x1_data), max(x1_data), grid_size)
    x2_grid = np.linspace(min(x2_data), max(x2_data), grid_size)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)

    # Interpolação
    points = np.column_stack((x1_data, x2_data))
    try:
        y_grid = griddata(points, y_data, (x1_mesh, x2_mesh), method="cubic")
    except Exception:
        y_grid = griddata(points, y_data, (x1_mesh, x2_mesh), method="linear")

    # Criar figura
    fig = go.Figure()

    # Superfície
    fig.add_trace(go.Surface(
        x=x1_grid,
        y=x2_grid,
        z=y_grid,
        colorscale=colorscale,
        showscale=True,
        opacity=opacity,
        contours={
            "z": {
                "show": True,
                "start": min(y_data),
                "end": max(y_data),
                "size": (max(y_data) - min(y_data)) / 15,
                "color": "white",
                "width": 1.5,
            }
        },
        name="Superfície Original",
        hoverinfo="z",
    ))

    # Pontos originais
    fig.add_trace(go.Scatter3d(
        x=x1_data,
        y=x2_data,
        z=y_data,
        mode="markers",
        marker=dict(
            size=marker_size,
            color="red",
            symbol="circle",
            line=dict(width=1, color="darkred")
        ),
        name="Dados Originais",
    ))

    # Layout
    fig.update_layout(
        title="Superfície dos Dados Originais",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"
        ),
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="y",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.5),
            zaxis=dict(
                range=[min(y_data) * 0.95, max(y_data) * 1.05]
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.0),
                center=dict(x=0, y=0, z=-0.2),
            ),
        ),
        width=900,
        height=700,
        margin=dict(l=10, r=10, t=60, b=10),
        template="plotly_white",
    )

    fig.show()
    return fig
