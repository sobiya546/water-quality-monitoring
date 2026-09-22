import pandas as pd


def predict_water_quality(
    model_bundle,
    values
):
    features = model_bundle["features"]

    row = {}

    for feature in features:
        row[feature] = values.get(
            feature,
            0
        )

    data = pd.DataFrame([row])

    model = model_bundle["models"][
        "Random Forest"
    ]

    prediction = model.predict(data)[0]

    probability = model.predict_proba(
        data
    )[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }