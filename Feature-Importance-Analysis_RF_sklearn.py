# -*- coding: utf-8 -*-
"""Feature importance analysis using Random Forest on heart disease data."""

import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
df = pd.read_csv(os.path.join("data", "heart.csv"))
y = df["output"]
x = df.drop("output", axis=1)

# Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train
model_rf = RFC(n_estimators=500, max_features="sqrt", max_depth=8, random_state=42)
model_rf.fit(x_train, y_train)

# Evaluate
y_pred = model_rf.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(model_rf.feature_importances_, index=x.columns)
importances.sort_values(ascending=True).plot(kind="barh", figsize=(8, 6))
plt.xlabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()
