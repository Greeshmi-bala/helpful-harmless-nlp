"""
Prediction module for the HH-RLHF NLP project.

Loads the saved machine learning model and compares two responses.
"""

from pathlib import Path
import json
import joblib


# Project paths
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_model.joblib"

INFO_PATH = ROOT / "models" / "best_model_info.json"


def load_model():
    """Load the saved machine learning model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_model_info():
    """Load the saved model information."""

    if INFO_PATH.exists():
        with open(INFO_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "best_model": "Unknown",
        "f1_score": None
    }


def predict_response_pair(
    prompt,
    response_a,
    response_b,
    model=None
):
    """
    Compare two candidate responses.

    The current baseline classifies each response separately.
    It is not a dedicated pairwise ranking model.
    """

    prompt = (prompt or "").strip()
    response_a = (response_a or "").strip()
    response_b = (response_b or "").strip()

    if not prompt or not response_a or not response_b:
        raise ValueError(
            "Please enter the prompt, Response A, and Response B."
        )

    if model is None:
        model = load_model()

    def classify_response(response):
        """
        Try raw-text input formats.

        If the model requires engineered numerical features,
        this function must be adapted to Member 2's feature pipeline.
        """

        candidates = [
            f"{prompt}\n\nResponse: {response}",
            response
        ]

        errors = []

        for text in candidates:
            try:
                prediction = model.predict([text])

                if len(prediction) == 1:
                    return prediction[0]

            except Exception as error:
                errors.append(str(error))

        raise TypeError(
            "The saved model does not accept raw text directly. "
            "Connect the feature extraction logic from "
            "src/feature_engineering.py and src/train_models.py. "
            + " | ".join(errors[-2:])
        )

    label_a = classify_response(response_a)
    label_b = classify_response(response_b)

    def is_chosen(label):
        return str(label).strip().lower() in {
            "1",
            "chosen",
            "preferred",
            "positive",
            "true"
        }

    a_chosen = is_chosen(label_a)
    b_chosen = is_chosen(label_b)

    if a_chosen and not b_chosen:
        winner = "Response A"

    elif b_chosen and not a_chosen:
        winner = "Response B"

    else:
        winner = "Tie / inconclusive"

    return {
        "winner": winner,
        "response_a_label": str(label_a),
        "response_b_label": str(label_b),
        "explanation": (
            "The model classifies each response individually. "
            "This is a baseline comparison, not a dedicated "
            "pairwise ranking prediction."
        )
    }