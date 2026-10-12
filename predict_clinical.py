import pandas as pd
import joblib
import shap

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

model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")

print("Enter patient information:\n")

patient = {}

for feature in features:
    patient[feature] = float(input(f"{feature}: "))

X = pd.DataFrame([patient], columns=features)

X_imputed = imputer.transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=features)

prediction = model.predict(X_imputed)[0]
probability = model.predict_proba(X_imputed)[0][1]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_imputed)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

values = shap_values[0]

explanation = pd.DataFrame({
    "Feature": features,
    "Value": [patient[f] for f in features],
    "SHAP Value": values
})

explanation["Absolute Impact"] = explanation["SHAP Value"].abs()

total_impact = explanation["Absolute Impact"].sum()

explanation["Influence"] = (
    explanation["Absolute Impact"] / total_impact
) * 100

explanation = explanation.sort_values(
    "Absolute Impact",
    ascending=False
)

print("\n" + "=" * 60)
print("CLINICAL SCREENING RESULT")
print("=" * 60)

print(f"\nModel prediction: {prediction}")
print(f"Model probability: {probability * 100:.2f}%")

print("\nWhy did the model give this result?\n")

for _, row in explanation.head(5).iterrows():

    if row["SHAP Value"] > 0:
        direction = "higher model output"
    else:
        direction = "lower model output"

    print(
        f"• {row['Feature']} = {row['Value']} → "
        f"This factor has contributed toward a {direction} | "
        f"Influence: {row['Influence']:.1f}%"
    )

print(
    "\nThe influence percentages show the relative contribution "
    "of the displayed factors to this model prediction."
)

print(
    "This is an AI-based screening-support result, "
    "not a medical diagnosis."
)