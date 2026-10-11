
import joblib
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score

# Load the original paired dataset
df = pd.read_csv("data/hh_rlhf_cleaned.csv")
df = df.dropna(
    subset=["prompt", "chosen_response", "rejected_response"]
)

# Reproduce the prompt-grouped test split
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, groups=df["prompt"])
)

test_df = df.iloc[test_idx].copy()

# Load the trained response classifier
model = joblib.load("models/best_model.joblib")

# Predict whether each response is chosen (1) or rejected (0)
chosen_pred = model.predict(test_df["chosen_response"])
rejected_pred = model.predict(test_df["rejected_response"])

# Compare the classifier's confidence scores when available
if hasattr(model, "predict_proba"):
    chosen_score = model.predict_proba(
        test_df["chosen_response"]
    )[:, 1]
    rejected_score = model.predict_proba(
        test_df["rejected_response"]
    )[:, 1]
elif hasattr(model, "decision_function"):
    chosen_score = model.decision_function(
        test_df["chosen_response"]
    )
    rejected_score = model.decision_function(
        test_df["rejected_response"]
    )
else:
    raise ValueError("Model does not provide ranking scores.")

# Pairwise accuracy: how often chosen scores higher than rejected
pairwise_accuracy = (chosen_score > rejected_score).mean()

# Also measure how often the classifier labels each response correctly
individual_accuracy = (
    (chosen_pred == 1).mean() + (rejected_pred == 0).mean()
) / 2

print(f"Test pairs: {len(test_df)}")
print(f"Pairwise preference accuracy: {pairwise_accuracy:.4f}")
print(f"Individual balanced accuracy: {individual_accuracy:.4f}")

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

pd.DataFrame([{
    "test_pairs": len(test_df),
    "pairwise_accuracy": pairwise_accuracy,
    "individual_balanced_accuracy": individual_accuracy
}]).to_csv(
    output_dir / "pairwise_results.csv", index=False
)

print("Saved results to models/pairwise_results.csv")
