
import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from feature_engineering import load_data

# Load and balance data exactly as in training
df = load_data()
n = df["label"].value_counts().min()

df = (
    df.groupby("label", group_keys=False)
      .sample(n=n, random_state=42)
      .reset_index(drop=True)
)

X = df["response"]
y = df["label"]
groups = df["prompt"]

# Reproduce the prompt-grouped split
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)
train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

model = joblib.load("models/best_model.joblib")
predictions = model.predict(X_test)

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Rejected", "Chosen"],
    output_dict=True,
    zero_division=0
)

print("Classification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Rejected", "Chosen"],
    zero_division=0
))

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

cm = confusion_matrix(y_test, predictions, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Rejected", "Chosen"]
)
disp.plot(cmap="Blues")
plt.title("Naive Bayes - Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
plt.show()

print("\nSaved classification report and confusion matrix in models/.")
