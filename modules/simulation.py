"""
simulation.py

Virtual Bayesian optimisation campaigns.

Workflow

1. Train surrogate on BayBE score columns
2. Treat surrogate as virtual ground truth
3. Run repeated BayBE campaigns
4. Estimate convergence speed
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from baybe.campaign import Campaign

from modules.optimisation import (
    build_objective,
)


# ==========================================================
# SURROGATE MODEL
# ==========================================================

def train_surrogate_model(
    campaign_df,
    hv_col,
    mon_col,
    score_cols,
):
    """
    Train surrogate directly on
    normalised BayBE score columns.
    """

    X = campaign_df[
        [hv_col, mon_col]
    ]

    y = campaign_df[
        score_cols
    ]

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
    )

    model.fit(X, y)

    return model


# ==========================================================
# LANDSCAPE GENERATION
# ==========================================================

def generate_landscape(
    model,
    hv_values,
    mon_values,
    hv_col,
    mon_col,
    score_cols,
):
    """
    Predict full search space.
    """

    grid = pd.DataFrame(
        [
            [hv, mon]
            for hv in hv_values
            for mon in mon_values
        ],
        columns=[
            hv_col,
            mon_col,
        ],
    )

    predictions = model.predict(
        grid
    )

    for i, col in enumerate(score_cols):

        grid[col] = predictions[:, i]

    return grid


# ==========================================================
# SCORE CALCULATION
# ==========================================================

def calculate_weighted_score(
    df,
    score_cols,
    weights,
):
    """
    Weighted desirability score.
    """

    score = np.zeros(len(df))

    for i, col in enumerate(score_cols):

        score += (
            weights[i]
            * df[col]
        )

    return score


# ==========================================================
# FIND OPTIMUM
# ==========================================================

def find_virtual_optimum(
    landscape_df,
    score_cols,
    weights,
):
    """
    Locate optimum score.
    """

    scores = calculate_weighted_score(
        landscape_df,
        score_cols,
        weights,
    )

    best_idx = np.argmax(scores)

    optimum_score = scores[best_idx]

    optimum_row = (
        landscape_df
        .iloc[best_idx]
        .copy()
    )

    return optimum_score, optimum_row


# ==========================================================
# VIRTUAL EXPERIMENT
# ==========================================================

def evaluate_virtual_experiment(
    model,
    recommendations,
    hv_col,
    mon_col,
    score_cols,
):
    """
    Predict virtual experimental outcome.
    """

    X = recommendations[
        [hv_col, mon_col]
    ]

    predictions = model.predict(X)

    results = recommendations.copy()

    for i, score_col in enumerate(score_cols):

        results[score_col] = (
            predictions[:, i]
        )

    return results


# ==========================================================
# SINGLE VIRTUAL CAMPAIGN
# ==========================================================

def run_virtual_campaign(
    initial_df,
    score_cols,
    searchspace,
    hv_col,
    mon_col,
    weights,
    batch_size,
    surrogate_model,
    optimum_score,
    convergence_fraction=0.95,
    max_cycles=30,
):
    """
    Simulate one BayBE campaign.
    """

    current_df = (
        initial_df.copy()
    )

    history = []

    objective = build_objective(
        score_cols,
        weights,
    )

    for cycle in range(max_cycles):

        campaign = Campaign(
            searchspace=searchspace,
            objective=objective,
        )

        campaign.add_measurements(
            current_df[
                [hv_col, mon_col]
                + score_cols
            ]
        )

        recommendations = campaign.recommend(
            batch_size=batch_size
        )

        tested = set(
            zip(
                current_df[hv_col],
                current_df[mon_col]
            )
        )

        recommendations = recommendations[
            ~recommendations.apply(
                lambda row:
                (
                    row[hv_col],
                    row[mon_col]
                ) in tested,
                axis=1,
            )
        ]

        if len(recommendations) == 0:

            return {
                "converged": False,
                "cycles": cycle + 1,
                "best_score": history[-1]
                if history else 0,
                "history": history,
            }

        virtual_results = (
            evaluate_virtual_experiment(
                surrogate_model,
                recommendations,
                hv_col,
                mon_col,
                score_cols,
            )
        )

        current_df = pd.concat(
            [
                current_df,
                virtual_results,
            ],
            ignore_index=True,
        )

        scores = calculate_weighted_score(
            current_df,
            score_cols,
            weights,
        )

        best_score = scores.max()

        history.append(
            best_score
        )

        if (
            best_score
            >=
            convergence_fraction
            * optimum_score
        ):

            return {
                "converged": True,
                "cycles": cycle + 1,
                "best_score": best_score,
                "history": history,
            }

    return {
        "converged": False,
        "cycles": max_cycles,
        "best_score": best_score,
        "history": history,
    }


# ==========================================================
# MONTE CARLO SIMULATION
# ==========================================================

def monte_carlo_simulation(
    campaign_df,
    score_cols,
    searchspace,
    hv_col,
    mon_col,
    weights,
    hv_values,
    mon_values,
    batch_size=3,
    n_runs=100,
    seed_size=10,
    convergence_fraction=0.95,
    max_cycles=30,
):
    """
    Estimate BayBE convergence
    through repeated virtual campaigns.
    """

    surrogate_model = (
        train_surrogate_model(
            campaign_df,
            hv_col,
            mon_col,
            score_cols,
        )
    )

    landscape = generate_landscape(
        surrogate_model,
        hv_values,
        mon_values,
        hv_col,
        mon_col,
        score_cols,
    )

    optimum_score, optimum_row = (
        find_virtual_optimum(
            landscape,
            score_cols,
            weights,
        )
    )

    cycle_results = []
    histories = []

    for run in range(n_runs):

        seed_df = (
            campaign_df.sample(
                min(
                    seed_size,
                    len(campaign_df)
                ),
                random_state=run,
            )
            .copy()
        )

        result = run_virtual_campaign(
            initial_df=seed_df,
            score_cols=score_cols,
            searchspace=searchspace,
            hv_col=hv_col,
            mon_col=mon_col,
            weights=weights,
            batch_size=batch_size,
            surrogate_model=surrogate_model,
            optimum_score=optimum_score,
            convergence_fraction=convergence_fraction,
            max_cycles=max_cycles,
        )

        cycle_results.append(
            result["cycles"]
        )

        histories.append(
            result["history"]
        )

    return {

        "mean_cycles":
            float(np.mean(cycle_results)),

        "median_cycles":
            float(np.median(cycle_results)),

        "std_cycles":
            float(np.std(cycle_results)),

        "success_rate":
            float(
                np.mean(
                    np.array(cycle_results)
                    < max_cycles
                )
            ),

        "optimum_score":
            float(optimum_score),

        "optimum_row":
            optimum_row,

        "cycle_results":
            cycle_results,

        "histories":
            histories,
    }