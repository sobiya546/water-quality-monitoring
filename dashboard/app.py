import sys
import os

import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.preprocessing import (
    detect_mapping,
    apply_mapping,
    clean_data
)

from src.analytics import (
    get_summary,
    source_quality,
    seasonal_analysis,
    region_analysis
)

from src.model import train_models

from src.prediction import (
    predict_water_quality
)

from src.recommendations import (
    generate_recommendations
)


st.set_page_config(
    page_title="Water Quality Monitoring",
    page_icon="💧",
    layout="wide"
)


st.title("💧 Water Quality Monitoring & Contamination Prediction")
st.write(
    "AI-based monitoring and prediction system for water quality."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)


@st.cache_data
def load_demo_data():

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "raw",
        "demo_water_quality.csv"
    )

    return pd.read_csv(path)


if uploaded_file:

    raw_df = pd.read_csv(uploaded_file)

    st.sidebar.success(
        "Uploaded dataset loaded."
    )

else:

    raw_df = load_demo_data()

    st.sidebar.info(
        "Using synthetic demo dataset."
    )


# --------------------------------------------------
# COLUMN MAPPING
# --------------------------------------------------

mapping = detect_mapping(
    raw_df.columns
)

if mapping:

    df = apply_mapping(
        raw_df,
        mapping
    )

else:

    df = raw_df.copy()


df = clean_data(df)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

model_bundle = None
model_error = None

if "Potability" in df.columns:

    try:

        model_bundle = train_models(df)

    except Exception as error:

        model_error = str(error)


# --------------------------------------------------
# TABS
# --------------------------------------------------

overview_tab, source_tab, predict_tab, recommendation_tab = st.tabs(
    [
        "📊 Overview",
        "🌦️ Source & Season",
        "🤖 Predict",
        "💡 Recommendations"
    ]
)


# ==================================================
# OVERVIEW
# ==================================================

with overview_tab:

    st.header("Water Quality Overview")

    summary = get_summary(df)

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Total Samples",
        summary["total"]
    )

    c2.metric(
        "Potable / Safe",
        summary["potable"]
    )

    c3.metric(
        "Non-Potable / Unsafe",
        summary["non_potable"]
    )

    c4.metric(
        "Sources",
        summary["sources"]
    )

    c5.metric(
        "Regions",
        summary["regions"]
    )

    st.divider()

    if "Potability" in df.columns:

        counts = (
            df["Potability"]
            .map({
                0: "Non-Potable",
                1: "Potable"
            })
            .value_counts()
            .reset_index()
        )

        counts.columns = [
            "Status",
            "Count"
        ]

        fig = px.pie(
            counts,
            names="Status",
            values="Count",
            title="Potability Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Parameter Distributions"
    )

    parameters = [
        "pH",
        "Turbidity",
        "Dissolved_Oxygen",
        "Hardness",
        "Bacterial_Count",
        "TDS",
        "Conductivity",
        "Temperature"
    ]

    available_parameters = [
        p for p in parameters
        if p in df.columns
    ]

    if available_parameters:

        selected = st.selectbox(
            "Select parameter",
            available_parameters
        )

        fig = px.histogram(
            df,
            x=selected,
            title=f"{selected} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Correlation Heatmap"
    )

    numeric_df = df.select_dtypes(
        include="number"
    )

    if len(numeric_df.columns) >= 2:

        fig = px.imshow(
            numeric_df.corr(),
            text_auto=True,
            title="Parameter Correlation"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==================================================
# SOURCE & SEASON
# ==================================================

with source_tab:

    st.header("Source & Seasonal Analysis")

    if "Source_Type" in df.columns:

        st.subheader(
            "Source-wise Sample Count"
        )

        source_count = (
            df["Source_Type"]
            .value_counts()
            .reset_index()
        )

        source_count.columns = [
            "Source",
            "Count"
        ]

        fig = px.bar(
            source_count,
            x="Source",
            y="Count"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if "Potability" in df.columns:

            quality = source_quality(df)

            if not quality.empty:

                st.subheader(
                    "Potable / Non-Potable by Source"
                )

                st.bar_chart(quality)

    if "Region" in df.columns:

        st.subheader(
            "Region Quality Analysis"
        )

        region_data = region_analysis(df)

        if not region_data.empty:

            region_data[
                "Potability"
            ] *= 100

            region_data = region_data.rename(
                columns={
                    "Potability":
                    "Potable Percentage"
                }
            )

            st.dataframe(
                region_data,
                use_container_width=True
            )

    if "Season" in df.columns:

        st.subheader(
            "Seasonal Parameter Trends"
        )

        numeric_columns = [
            c for c in [
                "pH",
                "Turbidity",
                "Dissolved_Oxygen",
                "Hardness",
                "Bacterial_Count",
                "TDS",
                "Conductivity",
                "Temperature"
            ]
            if c in df.columns
        ]

        if numeric_columns:

            selected_parameter = st.selectbox(
                "Parameter",
                numeric_columns,
                key="season_parameter"
            )

            seasonal = (
                df.groupby("Season")[
                    selected_parameter
                ]
                .mean()
                .reset_index()
            )

            fig = px.line(
                seasonal,
                x="Season",
                y=selected_parameter,
                markers=True,
                title=f"{selected_parameter} by Season"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ==================================================
# PREDICT
# ==================================================

with predict_tab:

    st.header(
        "🤖 Water Quality Prediction"
    )

    if model_bundle is None:

        st.error(
            "Model could not be trained."
        )

        if model_error:
            st.write(model_error)

    else:

        values = {}

        col1, col2 = st.columns(2)

        with col1:

            values["pH"] = st.number_input(
                "pH",
                min_value=0.0,
                max_value=14.0,
                value=7.0
            )

            values["Turbidity"] = st.number_input(
                "Turbidity (NTU)",
                min_value=0.0,
                value=2.0
            )

            values["Dissolved_Oxygen"] = st.number_input(
                "Dissolved Oxygen (mg/L)",
                min_value=0.0,
                value=7.0
            )

            values["Hardness"] = st.number_input(
                "Hardness (mg/L)",
                min_value=0.0,
                value=150.0
            )

        with col2:

            values["Bacterial_Count"] = st.number_input(
                "Bacterial Count",
                min_value=0.0,
                value=10.0
            )

            values["TDS"] = st.number_input(
                "TDS (mg/L)",
                min_value=0.0,
                value=250.0
            )

            values["Conductivity"] = st.number_input(
                "Conductivity",
                min_value=0.0,
                value=400.0
            )

            values["Temperature"] = st.number_input(
                "Temperature (°C)",
                min_value=0.0,
                value=25.0
            )

        if st.button(
            "Predict Water Quality",
            type="primary"
        ):

            result = predict_water_quality(
                model_bundle,
                values
            )

            prediction = result[
                "prediction"
            ]

            probability = result[
                "probability"
            ]

            if prediction == 1:

                st.success(
                    "✅ POTABLE / SAFE WATER"
                )

            else:

                st.error(
                    "⚠️ NON-POTABLE / UNSAFE WATER"
                )

            st.metric(
                "Prediction Probability",
                f"{probability * 100:.2f}%"
            )

            st.subheader(
                "Input Values"
            )

            st.dataframe(
                pd.DataFrame([values]),
                use_container_width=True
            )

        st.divider()

        st.subheader(
            "Internal Model Evaluation"
        )

    

# ==================================================
# RECOMMENDATIONS
# ==================================================

with recommendation_tab:

    st.header(
        "💡 Water Quality Recommendations"
    )

    st.write(
        "Recommendations are generated using "
        "transparent water-quality rules."
    )

    recommendation_values = {}

    for parameter in [
        "pH",
        "Turbidity",
        "Dissolved_Oxygen",
        "Hardness",
        "Bacterial_Count",
        "TDS",
        "Conductivity",
        "Temperature"
    ]:

        if parameter in df.columns:

            recommendation_values[
                parameter
            ] = float(
                df[parameter].mean()
            )

    recommendations = generate_recommendations(
        recommendation_values
    )

    for recommendation in recommendations:

        st.info(
            "💧 " + recommendation
        )