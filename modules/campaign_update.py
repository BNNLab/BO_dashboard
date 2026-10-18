"""
campaign_update.py

Functions for managing iterative
Bayesian optimisation campaigns.

Supports:

- Loading campaign history
- Appending new experiments
- Removing duplicates
- Saving updated campaigns
- Checking completion of recommendations
"""

from pathlib import Path

import pandas as pd


# ==========================================================
# LOAD CAMPAIGN HISTORY
# ==========================================================

def load_campaign_history(
    filepath,
):
    """
    Load previously saved campaign.
    """

    filepath = Path(filepath)

    if not filepath.exists():

        raise FileNotFoundError(
            f"Campaign history file not found: "
            f"{filepath}"
        )

    return pd.read_csv(filepath)


# ==========================================================
# SAVE CAMPAIGN HISTORY
# ==========================================================

def save_campaign_history(
    df,
    filepath,
):
    """
    Save campaign history.
    """

    Path(filepath).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        filepath,
        index=False
    )


# ==========================================================
# APPEND NEW RESULTS
# ==========================================================

def append_experimental_results(
    campaign_df,
    new_results_df,
):
    """
    Append newly measured data.

    Returns updated dataframe.
    """

    updated_df = pd.concat(
        [
            campaign_df,
            new_results_df
        ],
        ignore_index=True
    )

    return updated_df


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicate_formulations(
    df,
    variable_cols,
):
    """
    Remove repeated formulations.

    Keeps last occurrence.
    """

    updated_df = df.drop_duplicates(
        subset=variable_cols,
        keep="last"
    )

    updated_df = (
        updated_df
        .reset_index(drop=True)
    )

    return updated_df


# ==========================================================
# MERGE CAMPAIGNS
# ==========================================================

def merge_campaign_data(
    existing_df,
    new_df,
    variable_cols,
):
    """
    Merge newly measured data.

    New data overwrites existing
    measurements if duplicated.
    """

    merged = pd.concat(
        [
            existing_df,
            new_df
        ],
        ignore_index=True
    )

    merged = remove_duplicate_formulations(
        merged,
        variable_cols
    )

    return merged


# ==========================================================
# FIND UNTESTED RECOMMENDATIONS
# ==========================================================

def identify_remaining_recommendations(
    recommendations_df,
    completed_df,
    variable_cols,
):
    """
    Determine which recommendations
    have not yet been carried out.
    """

    tested = set(

        tuple(row)

        for row in completed_df[
            variable_cols
        ].values

    )

    remaining = recommendations_df[
        ~recommendations_df[
            variable_cols
        ]
        .apply(
            tuple,
            axis=1
        )
        .isin(tested)
    ]

    return (
        remaining
        .reset_index(drop=True)
    )


# ==========================================================
# VALIDATE UPDATE FILE
# ==========================================================

def validate_update_data(
    update_df,
    required_columns,
):
    """
    Ensure uploaded data contain
    all required columns.
    """

    missing = [

        col

        for col in required_columns

        if col not in update_df.columns

    ]

    if len(missing) > 0:

        raise ValueError(

            "Missing required columns: "
            + ", ".join(missing)

        )

    return True


# ==========================================================
# CAMPAIGN STATUS
# ==========================================================

def campaign_summary(
    campaign_df,
    hv_col,
    mon_col,
):
    """
    Simple campaign statistics.
    """

    unique_formulations = (
        campaign_df[
            [hv_col, mon_col]
        ]
        .drop_duplicates()
    )

    return {

        "total_records":
            len(campaign_df),

        "unique_formulations":
            len(unique_formulations),

        "hv_min":
            campaign_df[hv_col].min(),

        "hv_max":
            campaign_df[hv_col].max(),

        "monolein_min":
            campaign_df[mon_col].min(),

        "monolein_max":
            campaign_df[mon_col].max(),

    }


# ==========================================================
# FULL UPDATE WORKFLOW
# ==========================================================

def update_campaign(
    existing_df,
    new_results_df,
    variable_cols,
):
    """
    Complete update workflow.
    """

    updated_df = merge_campaign_data(
        existing_df,
        new_results_df,
        variable_cols
    )

    return updated_df