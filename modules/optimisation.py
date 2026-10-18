"""
optimisation.py

Core BayBE optimisation functions.

Handles:

- Data preparation
- Objective normalisation
- Search-space construction
- BayBE campaign creation
- Recommendation generation
- Removal of previously tested formulations
"""

import numpy as np
import pandas as pd

from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.objectives import DesirabilityObjective
from baybe.campaign import Campaign


# ==========================================================
# DATASET CREATION
# ==========================================================

def build_campaign_dataframe(
    df,
    hv_col,
    mon_col,
    objective_cols,
    day_col=None,
    day_value=0,
):
    """
    Construct optimisation dataframe.

    Parameters
    ----------
    df : DataFrame

    hv_col : str
    mon_col : str

    objective_cols : list

    day_col : str or None

    day_value : numeric

    Returns
    -------
    campaign_df
    """

    if day_col is not None:

        campaign_df = (
            df[df[day_col] == day_value]
            [
                [hv_col, mon_col]
                + objective_cols
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    else:

        campaign_df = (
            df[
                [hv_col, mon_col]
                + objective_cols
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    return campaign_df


# ==========================================================
# NORMALISATION
# ==========================================================

def normalise_objectives(
    campaign_df,
    objective_cols,
    method="Max",
):

    campaign_df = campaign_df.copy()

    score_cols = []

    for obj in objective_cols:

        score_col = f"{obj}_Score"

        values = campaign_df[obj]

        if method == "Max":

            maximum = values.max()

            if maximum == 0:

                campaign_df[score_col] = 0.0

            else:

                campaign_df[score_col] = (
                    values / maximum
                )

        elif method == "Min-Max":

            minimum = values.min()
            maximum = values.max()

            if maximum == minimum:

                campaign_df[score_col] = 0.0

            else:

                campaign_df[score_col] = (
                    values - minimum
                ) / (
                    maximum - minimum
                )

        elif method == "Robust":

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)

            if q3 == q1:

                campaign_df[score_col] = 0.0

            else:

                campaign_df[score_col] = (
                    values - q1
                ) / (
                    q3 - q1
                )

                campaign_df[score_col] = (
                    campaign_df[score_col]
                    .clip(0, 1)
                )

        else:

            raise ValueError(
                f"Unknown normalisation method: {method}"
            )

        score_cols.append(score_col)

    return campaign_df, score_cols


# ==========================================================
# SEARCH SPACE
# ==========================================================

def build_searchspace(
    hv_col,
    mon_col,
    hv_min,
    hv_max,
    hv_step,
    mon_min,
    mon_max,
    mon_step,
):

    hv_values = np.arange(
        hv_min,
        hv_max + hv_step,
        hv_step,
    )

    mon_values = np.arange(
        mon_min,
        mon_max + mon_step,
        mon_step,
    )

    parameters = [

        NumericalDiscreteParameter(
            name=hv_col,
            values=hv_values.tolist(),
        ),

        NumericalDiscreteParameter(
            name=mon_col,
            values=mon_values.tolist(),
        ),
    ]

    searchspace = SearchSpace.from_product(
        parameters
    )

    return searchspace


# ==========================================================
# BUILD TARGETS
# ==========================================================

def build_targets(score_cols):

    targets = []

    for score_col in score_cols:

        targets.append(

            NumericalTarget(
                name=score_col,
                mode="MAX",
                bounds=(0, 1),
            )

        )

    return targets


# ==========================================================
# BUILD OBJECTIVE
# ==========================================================

def build_objective(
    score_cols,
    weights,
):
    """
    Automatically switch between
    single-objective and multi-objective
    BayBE workflows.
    """

    targets = build_targets(
        score_cols
    )

    if len(targets) == 1:

        return targets[0]

    return DesirabilityObjective(
        targets=targets,
        weights=weights,
    )

    targets = build_targets(
        score_cols
    )

    objective = DesirabilityObjective(
        targets=targets,
        weights=weights,
    )

    return objective


# ==========================================================
# CREATE CAMPAIGN
# ==========================================================

def create_campaign(
    searchspace,
    objective,
):

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )

    return campaign


# ==========================================================
# RECOMMENDATIONS
# ==========================================================

def run_baybe_optimisation(
    campaign_df,
    score_cols,
    hv_col,
    mon_col,
    searchspace,
    weights,
    batch_size,
):

    objective = build_objective(
        score_cols,
        weights,
    )

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )

    measurement_df = campaign_df[
        [
            hv_col,
            mon_col,
        ]
        + score_cols
    ]

    campaign.add_measurements(
        measurement_df
    )

    recommendations = campaign.recommend(
        batch_size=batch_size
    )

    recommendations = remove_tested_points(
        recommendations,
        campaign_df,
        hv_col,
        mon_col,
    )

    return recommendations


# ==========================================================
# REMOVE TESTED POINTS
# ==========================================================

def remove_tested_points(
    recommendations,
    campaign_df,
    hv_col,
    mon_col,
):

    tested = set(
        zip(
            campaign_df[hv_col],
            campaign_df[mon_col],
        )
    )

    recommendations = recommendations[
        ~recommendations.apply(
            lambda row:
            (
                row[hv_col],
                row[mon_col],
            )
            in tested,
            axis=1,
        )
    ]

    recommendations = (
        recommendations
        .drop_duplicates(
            subset=[
                hv_col,
                mon_col,
            ]
        )
        .reset_index(drop=True)
    )

    return recommendations


# ==========================================================
# COMPLETE WORKFLOW
# ==========================================================

def optimise_from_dataframe(
    df,
    hv_col,
    mon_col,
    objective_cols,
    weights,
    hv_min,
    hv_max,
    hv_step,
    mon_min,
    mon_max,
    mon_step,
    batch_size,
    normalisation_method="Robust",
    day_col=None,
    day_value=0,
):

    campaign_df = build_campaign_dataframe(
        df=df,
        hv_col=hv_col,
        mon_col=mon_col,
        objective_cols=objective_cols,
        day_col=day_col,
        day_value=day_value,
    )

    campaign_df, score_cols = (
        normalise_objectives(
            campaign_df,
            objective_cols,
            normalisation_method,
        )
    )

    searchspace = build_searchspace(
        hv_col=hv_col,
        mon_col=mon_col,
        hv_min=hv_min,
        hv_max=hv_max,
        hv_step=hv_step,
        mon_min=mon_min,
        mon_max=mon_max,
        mon_step=mon_step,
    )

    recommendations = (
        run_baybe_optimisation(
            campaign_df=campaign_df,
            score_cols=score_cols,
            hv_col=hv_col,
            mon_col=mon_col,
            searchspace=searchspace,
            weights=weights,
            batch_size=batch_size,
        )
    )

    return (
        campaign_df,
        score_cols,
        recommendations,
        searchspace,
    )