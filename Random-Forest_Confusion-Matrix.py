import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# --- Load ---
df = pd.read_csv("./heart.csv")

# --- Preprocessing ---
# Drop columns with >50% missing
df.drop(columns=df.columns[df.isna().mean() > 0.5], inplace=True)

# Separate target before outlier handling
y = df["output"]
x = df.drop("output", axis=1)

# Fill missing (mode for binary/categorical, mean for continuous)
for col in x.columns:
    if x[col].nunique() <= 5:
        x[col].fillna(x[col].mode()[0], inplace=True)
    else:
        x[col].fillna(x[col].mean(), inplace=True)

# Outlier removal (features only)
numeric = x.select_dtypes(include=["float64", "int64"])
z = np.abs((numeric - numeric.mean()) / numeric.std())
mask = (z <= 3).all(axis=1)
x, y = x[mask], y[mask]

# --- Train / Test ---
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = RFC(n_estimators=500, max_features="sqrt", max_depth=8, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# --- Evaluation ---
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# --- Feature Importance ---
importances = pd.Series(model.feature_importances_, index=x.columns)
importances.sort_values(ascending=True).plot(kind="barh")
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

# --- Partial Dependence Plots (replaces manual simulation) ---
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(14, 8))
PartialDependenceDisplay.from_estimator(model, x_test, features=range(x.shape[1]), ax=ax)
plt.tight_layout()
plt.show()
