"""
plotting.py

Visualisation functions for:

- 3D formulation space
- BayBE recommendations
- Response surfaces
- Heatmaps
- Correlation analysis

Returns Plotly figures.
"""

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from scipy.interpolate import griddata


# ==========================================================
# 3D EXPERIMENTAL SPACE
# ==========================================================

def plot_experimental_space(
    campaign_df,
    hv_col,
    mon_col,
    objective_cols,
):
    """
    Plot all experimental data.

    First objective -> z-axis

    Second objective (if available)
    -> colour scale
    """

    z_col = objective_cols[0]

    if len(objective_cols) > 1:
        colour_col = objective_cols[1]
    else:
        colour_col = z_col

    fig = px.scatter_3d(
        campaign_df,
        x=hv_col,
        y=mon_col,
        z=z_col,
        color=colour_col,
        hover_data=campaign_df.columns
    )

    fig.update_layout(
        title="Experimental Space"
    )

    return fig


# ==========================================================
# OVERLAY RECOMMENDATIONS
# ==========================================================

def add_recommendations_to_3d_plot(
    fig,
    recommendations,
    hv_col,
    mon_col,
    z_value,
):
    """
    Add recommended experiments
    to existing figure.
    """

    if recommendations is None:
        return fig

    fig.add_trace(

        go.Scatter3d(

            x=recommendations[hv_col],

            y=recommendations[mon_col],

            z=np.repeat(
                z_value,
                len(recommendations)
            ),

            mode="markers",

            marker=dict(
                size=8,
                color="red",
                symbol="diamond"
            ),

            name="Recommendations"

        )

    )

    return fig


# ==========================================================
# RESPONSE SURFACE
# ==========================================================

def create_response_surface(
    campaign_df,
    hv_col,
    mon_col,
    objective_col,
    grid_size=50,
):
    """
    Create interpolated
    response surface.
    """

    x = campaign_df[hv_col]
    y = campaign_df[mon_col]
    z = campaign_df[objective_col]

    grid_x, grid_y = np.mgrid[
        x.min():x.max():complex(grid_size),
        y.min():y.max():complex(grid_size)
    ]

    grid_z = griddata(
        (x, y),
        z,
        (grid_x, grid_y),
        method="cubic"
    )

    fig = go.Figure(

        data=[

            go.Surface(
                x=grid_x,
                y=grid_y,
                z=grid_z,
                colorscale="Viridis"
            )

        ]

    )

    fig.update_layout(

        title=f"{objective_col} Surface",

        scene=dict(
            xaxis_title=hv_col,
            yaxis_title=mon_col,
            zaxis_title=objective_col
        ),

        height=700

    )

    return fig


# ==========================================================
# RESPONSE SURFACE + DATA POINTS
# ==========================================================

def create_surface_with_points(
    campaign_df,
    hv_col,
    mon_col,
    objective_col,
    grid_size=50,
):
    """
    Surface with experimental data
    overlaid.
    """

    x = campaign_df[hv_col]
    y = campaign_df[mon_col]
    z = campaign_df[objective_col]

    grid_x, grid_y = np.mgrid[
        x.min():x.max():complex(grid_size),
        y.min():y.max():complex(grid_size)
    ]

    grid_z = griddata(
        (x, y),
        z,
        (grid_x, grid_y),
        method="cubic"
    )

    fig = go.Figure()

    fig.add_trace(

        go.Surface(
            x=grid_x,
            y=grid_y,
            z=grid_z,
            opacity=0.8
        )

    )

    fig.add_trace(

        go.Scatter3d(

            x=x,
            y=y,
            z=z,

            mode="markers",

            marker=dict(
                size=6,
                color="black"
            ),

            name="Experiments"

        )

    )

    fig.update_layout(

        title=f"{objective_col} Surface",

        height=700

    )

    return fig


# ==========================================================
# HEATMAP
# ==========================================================

def create_heatmap(
    campaign_df,
    hv_col,
    mon_col,
    objective_col,
):
    """
    Heatmap representation.
    """

    pivot = campaign_df.pivot_table(

        values=objective_col,

        index=mon_col,

        columns=hv_col,

        aggfunc="mean"

    )

    fig = px.imshow(

        pivot,

        origin="lower",

        aspect="auto",

        labels=dict(
            x=hv_col,
            y=mon_col,
            color=objective_col
        )

    )

    fig.update_layout(

        title=f"{objective_col} Heatmap"

    )

    return fig


# ==========================================================
# CORRELATION MATRIX
# ==========================================================

def create_correlation_plot(
    campaign_df,
):
    """
    Correlation matrix
    for numerical variables.
    """

    numeric = campaign_df.select_dtypes(
        include=np.number
    )

    corr = numeric.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        title="Correlation Matrix"
    )

    return fig


# ==========================================================
# OBJECTIVE VS HV
# ==========================================================

def plot_objective_vs_hv(
    campaign_df,
    hv_col,
    objective_col,
):
    fig = px.scatter(

        campaign_df,

        x=hv_col,

        y=objective_col,

        trendline="ols"

    )

    fig.update_layout(
        title=f"{objective_col} vs {hv_col}"
    )

    return fig


# ==========================================================
# OBJECTIVE VS MONOLEIN
# ==========================================================

def plot_objective_vs_monolein(
    campaign_df,
    mon_col,
    objective_col,
):
    fig = px.scatter(

        campaign_df,

        x=mon_col,

        y=objective_col,

        trendline="ols"

    )

    fig.update_layout(
        title=f"{objective_col} vs {mon_col}"
    )

    return fig


# ==========================================================
# RECOMMENDATION MAP
# ==========================================================

def create_recommendation_map(
    campaign_df,
    recommendations,
    hv_col,
    mon_col,
):
    """
    2D formulation map showing
    explored vs recommended space.
    """

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=campaign_df[hv_col],

            y=campaign_df[mon_col],

            mode="markers",

            marker=dict(
                color="blue",
                size=8
            ),

            name="Measured"

        )

    )

    if recommendations is not None:

        fig.add_trace(

            go.Scatter(

                x=recommendations[hv_col],

                y=recommendations[mon_col],

                mode="markers",

                marker=dict(
                    color="red",
                    size=12,
                    symbol="diamond"
                ),

                name="Recommended"

            )

        )

    fig.update_layout(

        title="Formulation Map",

        xaxis_title=hv_col,

        yaxis_title=mon_col

    )

    return fig