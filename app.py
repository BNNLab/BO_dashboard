import streamlit as st
import pandas as pd
import numpy as np

from modules.optimisation import (
    optimise_from_dataframe,
)

from modules.plotting import (
    plot_experimental_space,
    add_recommendations_to_3d_plot,
    create_surface_with_points,
    create_heatmap,
    create_correlation_plot,
    create_recommendation_map,
)

from modules.pareto import (
    plot_pareto_2d,
    plot_pareto_3d,
    pareto_summary,
)

from modules.campaign_update import (
    update_campaign,
    campaign_summary,
)

from modules.simulation import (
    monte_carlo_simulation,
)

# ==========================================================
# PAGE SETUP
# ==========================================================

st.set_page_config(
    page_title="Bayesian Optimisation Dashboard",
    layout="wide",
)

st.title("🧭 Bayesian Optimisation Dashboard")

st.markdown(
    """
    Upload initial experimental data for Bayesian Optimisation. 
    Once uploaded, model parameters are available on the sidebar.
    """
)

st.sidebar.image("images/BNNLab_v3.png")

# ==========================================================
# DATA UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

st.success(f"Loaded {len(df)} rows")

# ==========================================================
# OPTIMISATION MODE
# ==========================================================

st.sidebar.header("Optimisation Mode")

optimisation_mode = st.sidebar.radio(
    "Select optimisation type",
    [
        "Single Objective",
        "Multi Objective",
    ]
)

# ==========================================================
# COLUMN SELECTION
# ==========================================================

st.sidebar.header("Column Selection")

columns = list(df.columns)

hv_col = st.sidebar.selectbox(
    "Variable X",
    columns,
    index=columns.index("HV")
    if "HV" in columns
    else 0,
)

mon_col = st.sidebar.selectbox(
    "Variable Y",
    columns,
    index=columns.index("Monolein")
    if "Monolein" in columns
    else min(1, len(columns) - 1),
)

day_col = st.sidebar.selectbox(
    "Day Column",
    columns,
    index=columns.index("Day")
    if "Day" in columns
    else 0,
)

# ==========================================================
# DAY SELECTION
# ==========================================================

# Get the Day values actually present in the uploaded dataset
day_values = (
    pd.to_numeric(df[day_col], errors="coerce")
    .dropna()
    .unique()
    .tolist()
)

day_values = sorted(day_values)

if not day_values:
    st.error(
        f"No valid Day values were found in the '{day_col}' column."
    )
    st.stop()

# Display integers cleanly where appropriate, e.g. 0, 14, 30
day_options = [
    int(value) if float(value).is_integer() else float(value)
    for value in day_values
]

day_value = st.sidebar.selectbox(
    "Day to Optimise",
    options=day_options,
    index=0,
    help=(
        "Select the Day value whose experimental responses "
        "will be used as the optimisation targets."
    ),
)

available_objectives = [
    c for c in columns
    if c not in [hv_col, mon_col, day_col]
]

if optimisation_mode == "Single Objective":

    selected_objective = st.sidebar.selectbox(
        "Optimisation Objective",
        available_objectives,
    )

    objective_cols = [selected_objective]

else:

    objective_cols = st.sidebar.multiselect(
        "Optimisation Objectives",
        available_objectives,
        default=[
            c
            for c in [
                "5/Transfection",
                "Encapsulation",
            ]
            if c in available_objectives
        ],
    )

    if len(objective_cols) < 2:

        st.warning(
            "Please select at least two objectives for multi-objective optimisation."
        )

        st.stop()

# ==========================================================
# SEARCH SPACE
# ==========================================================

st.sidebar.header("Search Space")

hv_min = st.sidebar.number_input(
    "HV Minimum",
    value=float(df[hv_col].min()),
)

hv_max = st.sidebar.number_input(
    "HV Maximum",
    value=float(df[hv_col].max()),
)

hv_step = st.sidebar.number_input(
    "HV Step",
    value=0.5,
)



mon_min = st.sidebar.number_input(
    "Monolein Minimum",
    value=float(df[mon_col].min()),
)

mon_max = st.sidebar.number_input(
    "Monolein Maximum",
    value=float(df[mon_col].max()),
)

mon_step = st.sidebar.number_input(
    "Monolein Step",
    value=5.0,
)

batch_size = st.sidebar.slider(
    "Batch Size",
    1,
    20,
    3,
)


# ==========================================================
# NORMALISATION
# ==========================================================

normalisation_method = st.sidebar.selectbox(
    "Normalisation",
    [
        "Max",
        "Min-Max",
        "Robust",
    ],
)



# ==========================================================
# OBJECTIVE WEIGHTS
# ==========================================================

if optimisation_mode == "Multi Objective":

    st.sidebar.header(
        "Objective Weights"
    )

    weights = []

    for obj in objective_cols:

        weight = st.sidebar.slider(
            obj,
            0.0,
            1.0,
            1.0 / len(objective_cols),
            0.05,
        )

        weights.append(weight)

    weights = np.array(weights)

    if weights.sum() == 0:

        st.error(
            "At least one objective weight must be non-zero."
        )

        st.stop()

    weights = weights / weights.sum()

else:

    weights = np.array([1.0])

# ==========================================================
# RUN OPTIMISATION
# ==========================================================

try:

    (
        campaign_df,
        score_cols,
        recommendations,
        searchspace,
    ) = optimise_from_dataframe(
        df=df,
        hv_col=hv_col,
        mon_col=mon_col,
        objective_cols=objective_cols,
        weights=weights.tolist(),
        hv_min=hv_min,
        hv_max=hv_max,
        hv_step=hv_step,
        mon_min=mon_min,
        mon_max=mon_max,
        mon_step=mon_step,
        batch_size=batch_size,
        normalisation_method=normalisation_method,
        day_col=day_col,
        day_value=day_value,
    )

except Exception as e:

    st.error(f"Optimisation failed:\n\n{e}")
    st.stop()

# ==========================================================
# TABS
# ==========================================================

tabs = st.tabs(
    [
        "Recommendations",
        "3D Visualisation",
        "Pareto",
        "Response Surfaces",
        "Campaign Update",
        "Simulation",
    ]
)

# ==========================================================
# TAB 1
# ==========================================================

with tabs[0]:

    st.write(
        f"**Optimisation mode:** {optimisation_mode}"
    )
    
    st.subheader("Training Dataset")

    st.dataframe(campaign_df)

    st.subheader(
        "Recommended Experiments"
    )

    st.dataframe(
        recommendations
    )

    csv = recommendations.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Recommendations",
        csv,
        "BayBE_next_batch.csv",
        "text/csv",
    )

# ==========================================================
# TAB 2
# ==========================================================

with tabs[1]:
    st.subheader(
        "3D Formulation Space"
    )

    fig = plot_experimental_space(
        campaign_df,
        hv_col,
        mon_col,
        objective_cols,
    )

    fig = add_recommendations_to_3d_plot(
        fig,
        recommendations,
        hv_col,
        mon_col,
        campaign_df[
            objective_cols[0]
        ].max(),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "Recommendation Map"
    )

    map_fig = create_recommendation_map(
        campaign_df,
        recommendations,
        hv_col,
        mon_col,
    )

    st.plotly_chart(
        map_fig,
        use_container_width=True,
    )

# ==========================================================
# TAB 3
# ==========================================================

with tabs[2]:

    st.subheader(
        "Pareto Analysis"
    )

    if optimisation_mode == "Single Objective":

        st.info(
            "Pareto analysis requires multiple objectives."
        )

    elif len(objective_cols) == 2:

        fig = plot_pareto_2d(
            campaign_df,
            objective_cols[0],
            objective_cols[1],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        fig = plot_pareto_3d(
            campaign_df,
            objective_cols[0],
            objective_cols[1],
            objective_cols[2],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    if optimisation_mode == "Multi Objective":

        summary = pareto_summary(
            campaign_df,
            objective_cols,
        )

        st.json(summary)

# ==========================================================
# TAB 4
# ==========================================================

with tabs[3]:
    st.subheader(
        "Response Surface Explorer"
    )

    surface_objective = st.selectbox(
        "Objective",
        objective_cols,
    )

    fig = create_surface_with_points(
        campaign_df,
        hv_col,
        mon_col,
        surface_objective,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    heatmap = create_heatmap(
        campaign_df,
        hv_col,
        mon_col,
        surface_objective,
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )

    corr = create_correlation_plot(
        campaign_df
    )

    st.plotly_chart(
        corr,
        use_container_width=True,
    )

# ==========================================================
# TAB 5
# ==========================================================

with tabs[4]:
    st.subheader(
        "Campaign Update"
    )

    update_file = st.file_uploader(
        "Upload completed experiments",
        type=["csv"],
        key="update_file",
    )

    if update_file is not None:

        update_df = pd.read_csv(
            update_file
        )

        updated_campaign = update_campaign(
            existing_df=campaign_df,
            new_results_df=update_df,
            variable_cols=[
                hv_col,
                mon_col,
            ],
        )

        st.success(
            "Campaign updated."
        )

        st.dataframe(
            updated_campaign
        )

        summary = campaign_summary(
            updated_campaign,
            hv_col,
            mon_col,
        )

        st.json(summary)

        updated_csv = (
            updated_campaign
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Updated Campaign",
            updated_csv,
            "campaign_history.csv",
            "text/csv",
        )

# ==========================================================
# TAB 6
# ==========================================================

with tabs[5]:
    st.subheader(
        "Virtual Optimisation Simulation"
    )

    n_runs = st.slider(
        "Simulation Runs",
        10,
        500,
        100,
    )

    max_cycles = st.slider(
        "Maximum Cycles",
        5,
        50,
        30,
    )

    if st.button(
        "Run Simulation"
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

        result = monte_carlo_simulation(
            campaign_df=campaign_df,
            score_cols=score_cols,
            searchspace=searchspace,
            hv_col=hv_col,
            mon_col=mon_col,
            weights=weights.tolist(),
            hv_values=hv_values,
            mon_values=mon_values,
            batch_size=batch_size,
            n_runs=n_runs,
            max_cycles=max_cycles,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Mean Cycles",
            f"{result['mean_cycles']:.1f}",
        )

        c2.metric(
            "Median Cycles",
            f"{result['median_cycles']:.1f}",
        )

        c3.metric(
            "Std Dev",
            f"{result['std_cycles']:.1f}",
        )

        c4.metric(
            "Success Rate",
            f"{100 * result['success_rate']:.1f}%",
        )

        st.subheader(
            "Estimated Optimum"
        )

        st.dataframe(
            pd.DataFrame(
                [result["optimum_row"]]
            )
        )

        hist_df = pd.DataFrame(
            {
                "Cycles":
                result["cycle_results"]
            }
        )

        st.bar_chart(
            hist_df["Cycles"]
            .value_counts()
            .sort_index()
        )

# The rest of the sidebar

st.sidebar.header("Acknowledgements")
st.sidebar.write("This web tool was built to help users interact with BayBE package for Bayesian Optimisation. The tool include ability to generate recommended experiments, 3D visualisation of those experiments, updating model with new data and simulation based on experimental data.")
st.sidebar.header("Disclaimer")
st.sidebar.write("This software was developed by BNNLab, with all rights reserved. It is offered 'as is', without warranty of any kind, express or implied. The user assumes all risk for any malfunctions, errors, or damages resulting from the use of this software. The creator is not responsible for any direct or indirect loss arising from its use.")
