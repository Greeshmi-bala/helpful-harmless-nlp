
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from feature_engineering import load_data

# Load data
df = load_data()

# Balance labels while retaining prompt groups
counts = df["label"].value_counts()
n = min(counts)
df = (
    df.groupby("label", group_keys=False)
      .sample(n=n, random_state=42)
      .reset_index(drop=True)
)

X = df["response"]
y = df["label"]
groups = df["prompt"]

# Split by prompt so the same prompt isn't in train and test
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = []
best_f1 = -1
best_model = None
best_name = ""

for name, classifier in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("classifier", classifier)
    ])

    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        )
    }
    results.append(metrics)
    print(metrics)

    if metrics["f1_score"] > best_f1:
        best_f1 = metrics["f1_score"]
        best_model = pipeline
        best_name = name

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

joblib.dump(best_model, output_dir / "best_model.joblib")
pd.DataFrame(results).to_csv(
    output_dir / "model_comparison.csv", index=False
)

with open(output_dir / "best_model_info.json", "w") as file:
    json.dump(
        {"best_model": best_name, "f1_score": best_f1},
        file, indent=4
    )

print(f"\nBest model: {best_name} (F1: {best_f1:.4f})")
print("Saved model and comparison results in models/.")
