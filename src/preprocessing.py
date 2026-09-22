import pandas as pd
import numpy as np


COLUMN_ALIASES = {
    "Sample_ID": ["sample_id", "sampleid", "sample"],
    "Source_ID": ["source_id", "sourceid", "source"],
    "Source_Type": ["source_type", "sourcetype", "water_source"],
    "Location": ["location", "place", "area"],
    "Region": ["region", "district", "zone"],
    "Date": ["date", "sample_date", "collection_date"],
    "pH": ["ph", "p_h"],
    "Turbidity": ["turbidity"],
    "Dissolved_Oxygen": [
        "dissolved_oxygen",
        "dissolved oxygen",
        "do"
    ],
    "Hardness": ["hardness"],
    "Bacterial_Count": [
        "bacterial_count",
        "bacterial count",
        "bacteria",
        "bacterial"
    ],
    "TDS": ["tds", "total_dissolved_solids"],
    "Conductivity": ["conductivity"],
    "Temperature": ["temperature", "temp"],
    "Potability": [
        "potability",
        "potable",
        "safe",
        "water_quality"
    ]
}


NUMERIC_COLUMNS = [
    "pH",
    "Turbidity",
    "Dissolved_Oxygen",
    "Hardness",
    "Bacterial_Count",
    "TDS",
    "Conductivity",
    "Temperature"
]


def normalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def detect_mapping(columns):
    normalized = {
        normalize_name(col): col
        for col in columns
    }

    mapping = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_normalized = normalize_name(alias)

            if alias_normalized in normalized:
                mapping[standard_name] = normalized[alias_normalized]
                break

    return mapping


def apply_mapping(df, mapping):
    reverse_mapping = {
        original: standard
        for standard, original in mapping.items()
    }

    return df.rename(columns=reverse_mapping)


def clean_data(df):
    df = df.copy()

    df = df.drop_duplicates()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Season"] = df["Date"].dt.month.map(
            get_season
        )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = df[column].fillna(
                df[column].median()
            )

    if "Potability" in df.columns:
        df["Potability"] = normalize_target(
            df["Potability"]
        )

    return df


def normalize_target(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.apply(
            lambda x: 1 if x >= 1 else 0
        )

    values = series.astype(str).str.lower().str.strip()

    safe_values = [
        "1",
        "yes",
        "true",
        "potable",
        "safe",
        "good"
    ]

    return values.isin(safe_values).astype(int)


def get_season(month):
    if pd.isna(month):
        return "Unknown"

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Summer"

    if month in [6, 7, 8, 9]:
        return "Monsoon"

    return "Post-Monsoon"


def detect_outliers(df):
    result = {}

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = (
            (df[column] < lower) |
            (df[column] > upper)
        ).sum()

        result[column] = int(count)

    return result