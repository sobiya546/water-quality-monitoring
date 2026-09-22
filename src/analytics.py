import pandas as pd


def get_summary(df):

    total = len(df)

    potable = 0
    non_potable = 0

    if "Potability" in df.columns:
        potable = int(
            (df["Potability"] == 1).sum()
        )

        non_potable = int(
            (df["Potability"] == 0).sum()
        )

    sources = (
        df["Source_ID"].nunique()
        if "Source_ID" in df.columns
        else 0
    )

    regions = (
        df["Region"].nunique()
        if "Region" in df.columns
        else 0
    )

    return {
        "total": total,
        "potable": potable,
        "non_potable": non_potable,
        "sources": sources,
        "regions": regions
    }


def source_counts(df):

    if "Source_Type" not in df.columns:
        return pd.DataFrame()

    return (
        df["Source_Type"]
        .value_counts()
        .reset_index()
    )


def source_quality(df):

    if (
        "Source_Type" not in df.columns
        or "Potability" not in df.columns
    ):
        return pd.DataFrame()

    return pd.crosstab(
        df["Source_Type"],
        df["Potability"]
    ).rename(
        columns={
            0: "Non-Potable",
            1: "Potable"
        }
    )


def seasonal_analysis(df):

    if (
        "Season" not in df.columns
        or "Potability" not in df.columns
    ):
        return pd.DataFrame()

    return df.groupby(
        "Season"
    )["Potability"].mean().reset_index()


def region_analysis(df):

    if (
        "Region" not in df.columns
        or "Potability" not in df.columns
    ):
        return pd.DataFrame()

    return df.groupby(
        "Region"
    )["Potability"].mean().reset_index()