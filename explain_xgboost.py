
import joblib
import shap
import pandas as pd
import numpy as np

model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")

features = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Smokes",
    "Smokes (years)",
    "Hormonal Contraceptives",
    "IUD",
    "STDs",
    "STDs (number)",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV"
]

explainer = shap.TreeExplainer(model)


def explain_clinical_input(values):

    user_data = pd.DataFrame(
        [values],
        columns=features
    )

    user_data_imputed = imputer.transform(
        user_data
    )

    probability = model.predict_proba(
        user_data_imputed
    )[0][1]

    prediction = int(probability >= 0.5)

    shap_values = explainer.shap_values(
        user_data_imputed
    )

    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            shap_row = np.array(shap_values[1])[0]
        else:
            shap_row = np.array(shap_values[0])[0]

    else:

        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            shap_row = shap_array[0, :, 1]
        else:
            shap_row = shap_array[0]

    absolute_values = np.abs(shap_row)

    total_influence = absolute_values.sum()

    if total_influence > 0:

        influence_percentages = (
            absolute_values / total_influence
        ) * 100

    else:

        influence_percentages = np.zeros(
            len(features)
        )

    explanations = []

    for i, feature in enumerate(features):

        shap_value = float(shap_row[i])

        influence = float(
            influence_percentages[i]
        )

        value = values[i]

        if shap_value > 0:
            direction = "higher"

        elif shap_value < 0:
            direction = "lower"

        else:
            direction = "little"

        explanations.append({
            "feature": feature,
            "value": value,
            "influence": round(influence, 1),
            "direction": direction
        })

    explanations.sort(
        key=lambda x: x["influence"],
        reverse=True
    )

    top_explanations = explanations[:5]

    return {
        "prediction": prediction,
        "probability": round(
            probability * 100,
            2
        ),
        "explanations": top_explanations
    }


def format_feature_value(feature, value):

    if feature == "Age":
        return f"{value:g} years"

    if feature == "Number of sexual partners":
        return f"{value:g}"

    if feature == "First sexual intercourse":
        return f"{value:g} years"

    if feature == "Smokes":
        return "Yes" if value == 1 else "No"

    if feature == "Smokes (years)":
        return f"{value:g}"

    if feature == "Hormonal Contraceptives":
        return "Yes" if value == 1 else "No"

    if feature == "IUD":
        return "Yes" if value == 1 else "No"

    if feature == "STDs":
        return "Yes" if value == 1 else "No"

    if feature == "STDs (number)":
        return f"{value:g}"

    if feature == "Dx:Cancer":
        return "Yes" if value == 1 else "No"

    if feature == "Dx:CIN":
        return "Yes" if value == 1 else "No"

    if feature == "Dx:HPV":
        return "Yes" if value == 1 else "No"

    return str(value)


def create_clinical_explanation(result):

    probability = result["probability"]

    lines = []

    lines.append("Clinical AI Explanation")
    lines.append("")

    lines.append(
        f"Predicted probability: {probability:.2f}%"
    )

    lines.append("")

    lines.append(
        f"The model estimated a {probability:.2f}% "
        "probability for the positive prediction "
        "based on the clinical information provided."
    )

    lines.append("")

    lines.append(
        "Factors that influenced this prediction:"
    )

    lines.append("")

    for item in result["explanations"]:

        feature = item["feature"]
        value = item["value"]
        influence = item["influence"]
        direction = item["direction"]

        readable_value = format_feature_value(
            feature,
            value
        )

        if direction == "higher":

            sentence = (
                "pushed the model's predicted "
                "probability higher."
            )

        elif direction == "lower":

            sentence = (
                "pushed the model's predicted "
                "probability lower."
            )

        else:

            sentence = (
                "had little influence on the "
                "model's predicted probability."
            )

        lines.append(
            f"• {feature}: {readable_value}"
        )

        lines.append(
            f"  This factor contributed {influence:.1f}% "
            "of the displayed SHAP influence and "
            f"{sentence}"
        )

        lines.append("")

    lines.append(
        "These percentages represent the relative "
        "SHAP influence of the displayed factors on "
        "this model prediction. They do not represent "
        "individual medical risk percentages or causation."
    )

    lines.append("")

    lines.append(
        "This is an AI-based screening-support result "
        "and not a medical diagnosis."
    )

    return "\n".join(lines)

