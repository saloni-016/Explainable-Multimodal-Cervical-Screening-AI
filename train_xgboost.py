import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")

df = df.replace("?", np.nan)

df = df.apply(
    pd.to_numeric,
    errors="coerce"
)


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


X = df[features]

y = df["Biopsy"]


imputer = SimpleImputer(
    strategy="mean"
)

X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("Accuracy:", accuracy)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


joblib.dump(
    model,
    "model.pkl"
)


joblib.dump(
    imputer,
    "imputer.pkl"
)


print("\nModel saved as model.pkl")

print("Imputer saved as imputer.pkl")