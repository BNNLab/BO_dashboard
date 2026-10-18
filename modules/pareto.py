"""
pareto.py

Functions for Pareto-front analysis
and visualisation.

Assumes all objectives are to be maximised.
"""

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# PARETO CALCULATION
# ==========================================================

def find_pareto_front(
    df,
    objective_cols,
):
    """
    Identify non-dominated points.

    Parameters
    ----------
    df : pandas.DataFrame

    objective_cols : list[str]

    Returns
    -------
    pareto_df : pandas.DataFrame
    """

    values = df[objective_cols].values

    pareto_mask = np.ones(
        len(values),
        dtype=bool
    )

    for i, point in enumerate(values):

        if not pareto_mask[i]:
            continue

        for j, other in enumerate(values):

            if i == j:
                continue

            dominates = (
                np.all(other >= point)
                and
                np.any(other > point)
            )

            if dominates:

                pareto_mask[i] = False
                break

    pareto_df = (
        df[pareto_mask]
        .copy()
        .reset_index(drop=True)
    )

    return pareto_df


# ==========================================================
# PARETO RANK
# ==========================================================

def assign_pareto_rank(
    df,
    objective_cols,
):
    """
    Assign Pareto ranking.

    Pareto rank 1 -> front
    Pareto rank 2 -> second front
    etc.

    Useful for colouring scatter plots.
    """

    remaining = df.copy()

    ranks = pd.Series(
        index=df.index,
        dtype=int
    )

    rank = 1

    while len(remaining) > 0:

        front = find_pareto_front(
            remaining,
            objective_cols
        )

        ranks.loc[
            front.index
        ] = rank

        remaining = remaining.drop(
            front.index
        )

        rank += 1

    output = df.copy()

    output["ParetoRank"] = ranks

    return output


# ==========================================================
# 2D PARETO PLOT
# ==========================================================

def plot_pareto_2d(
    df,
    objective_1,
    objective_2,
):
    """
    Generate 2D Pareto plot.
    """

    pareto_df = find_pareto_front(
        df,
        [
            objective_1,
            objective_2
        ]
    )

    fig = px.scatter(
        df,
        x=objective_1,
        y=objective_2,
        opacity=0.7
    )

    fig.add_trace(

        go.Scatter(
            x=pareto_df[objective_1],
            y=pareto_df[objective_2],

            mode="markers",

            marker=dict(
                size=14,
                color="red",
                line=dict(
                    color="black",
                    width=1
                )
            ),

            name="Pareto Front"
        )

    )

    fig.update_layout(
        title="Pareto Front Analysis"
    )

    return fig


# ==========================================================
# PARETO RANK PLOT
# ==========================================================

def plot_pareto_ranked(
    df,
    objective_1,
    objective_2,
):
    """
    Scatter plot coloured by Pareto rank.
    """

    ranked_df = assign_pareto_rank(
        df,
        [
            objective_1,
            objective_2
        ]
    )

    fig = px.scatter(
        ranked_df,
        x=objective_1,
        y=objective_2,
        color="ParetoRank",
        hover_data=ranked_df.columns
    )

    fig.update_layout(
        title="Pareto Ranking"
    )

    return fig


# ==========================================================
# 3D PARETO PLOT
# ==========================================================

def plot_pareto_3d(
    df,
    objective_1,
    objective_2,
    objective_3,
):
    """
    3-objective Pareto analysis.
    """

    pareto_df = find_pareto_front(
        df,
        [
            objective_1,
            objective_2,
            objective_3
        ]
    )

    fig = px.scatter_3d(
        df,
        x=objective_1,
        y=objective_2,
        z=objective_3,
        opacity=0.4
    )

    fig.add_trace(

        go.Scatter3d(

            x=pareto_df[objective_1],
            y=pareto_df[objective_2],
            z=pareto_df[objective_3],

            mode="markers",

            marker=dict(
                size=8,
                color="red"
            ),

            name="Pareto Front"

        )

    )

    fig.update_layout(
        title="3D Pareto Front"
    )

    return fig


# ==========================================================
# PARETO SUMMARY
# ==========================================================

def pareto_summary(
    df,
    objective_cols,
):
    """
    Return simple summary statistics.
    """

    pareto_df = find_pareto_front(
        df,
        objective_cols
    )

    summary = {

        "total_points":
            len(df),

        "pareto_points":
            len(pareto_df),

        "pareto_fraction":
            len(pareto_df) / len(df)
            if len(df) > 0
            else 0

    }

    return summary