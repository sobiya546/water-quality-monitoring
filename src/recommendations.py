def generate_recommendations(values):

    recommendations = []

    turbidity = values.get(
        "Turbidity",
        0
    )

    bacteria = values.get(
        "Bacterial_Count",
        0
    )

    ph = values.get(
        "pH",
        7
    )

    tds = values.get(
        "TDS",
        0
    )

    oxygen = values.get(
        "Dissolved_Oxygen",
        0
    )

    if turbidity > 5:
        recommendations.append(
            "High turbidity detected: "
            "consider filtration and sedimentation."
        )

    if bacteria > 100:
        recommendations.append(
            "High bacterial count detected: "
            "perform microbiological testing "
            "and appropriate disinfection."
        )

    if ph < 6.5 or ph > 8.5:
        recommendations.append(
            "Abnormal pH detected: investigate "
            "the water source and treatment process."
        )

    if tds > 500:
        recommendations.append(
            "High TDS detected: investigate "
            "dissolved solids and consider "
            "appropriate treatment."
        )

    if oxygen < 5:
        recommendations.append(
            "Low dissolved oxygen detected: "
            "investigate possible organic pollution "
            "and consider aeration."
        )

    if not recommendations:
        recommendations.append(
            "No major rule-based issues detected. "
            "Continue regular monitoring."
        )

    return recommendations