"""
Advanced NLP analysis for the HH-RLHF project.

Includes:
1. Response length analysis
2. Model comparison visualization
3. Confusion matrix export
4. Semantic similarity using Sentence Transformers
"""

from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Project directories
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "hh_rlhf_cleaned.csv"

MODELS_DIR = ROOT / "models"

OUTPUT_DIR = ROOT / "outputs" / "member3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the cleaned HH-RLHF dataset."""

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "prompt",
        "chosen_response",
        "rejected_response"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    return df.dropna(
        subset=list(required_columns)
    ).copy()


def make_response_length_chart(df):
    """Compare the lengths of chosen and rejected responses."""

    chosen = (
        df["chosen_response"]
        .astype(str)
        .str.split()
        .str.len()
    )

    rejected = (
        df["rejected_response"]
        .astype(str)
        .str.split()
        .str.len()
    )

    summary = pd.DataFrame({
        "response_type": [
            "Chosen",
            "Rejected"
        ],

        "mean_words": [
            chosen.mean(),
            rejected.mean()
        ],

        "median_words": [
            chosen.median(),
            rejected.median()
        ],

        "mean_characters": [
            df["chosen_response"].astype(str).str.len().mean(),
            df["rejected_response"].astype(str).str.len().mean()
        ]
    })

    summary.to_csv(
        OUTPUT_DIR / "response_length_summary.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        chosen.clip(upper=chosen.quantile(0.99)),
        bins=40,
        alpha=0.6,
        label="Chosen"
    )

    ax.hist(
        rejected.clip(upper=rejected.quantile(0.99)),
        bins=40,
        alpha=0.6,
        label="Rejected"
    )

    ax.set_title("Response Length Distribution")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Number of Responses")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "response_length_distribution.png",
        dpi=160
    )

    plt.close(fig)

    return summary


def make_model_comparison_chart():
    """Visualize the existing model evaluation results."""

    source = MODELS_DIR / "model_comparison.csv"

    if not source.exists():
        print("Model comparison CSV not found.")
        return

    df = pd.read_csv(source)

    metrics = [
        column
        for column in [
            "accuracy",
            "precision",
            "recall",
            "f1_score"
        ]
        if column in df.columns
    ]

    if "model" not in df.columns or not metrics:
        print("Unexpected model comparison file format.")
        return

    ax = df.set_index("model")[metrics].plot(
        kind="bar",
        figsize=(10, 5)
    )

    ax.set_title("Machine Learning Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.figure.tight_layout()

    ax.figure.savefig(
        OUTPUT_DIR / "model_comparison_metrics.png",
        dpi=160
    )

    plt.close(ax.figure)


def copy_confusion_matrix():
    """Copy the existing confusion matrix into the output folder."""

    source = MODELS_DIR / "confusion_matrix.png"

    if source.exists():
        shutil.copy2(
            source,
            OUTPUT_DIR / "confusion_matrix.png"
        )


def run_basic_analysis():
    """Generate the basic Member 3 visualizations."""

    df = load_data()

    summary = make_response_length_chart(df)

    make_model_comparison_chart()

    copy_confusion_matrix()

    print(f"Dataset rows analyzed: {len(df):,}")

    print("\nResponse Length Summary:")

    print(summary.to_string(index=False))

    print(f"\nOutput directory: {OUTPUT_DIR}")


def run_semantic_similarity(
    sample_size=300,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Compare prompt-to-chosen and prompt-to-rejected semantic similarity.
    """

    try:
        from sentence_transformers import SentenceTransformer

    except ImportError as error:
        raise ImportError(
            "Install Sentence Transformers using: "
            "python -m pip install sentence-transformers"
        ) from error

    df = load_data()

    sample = df.sample(
        n=min(sample_size, len(df)),
        random_state=42
    ).reset_index(drop=True)

    encoder = SentenceTransformer(model_name)

    prompts = sample["prompt"].astype(str).tolist()

    chosen = sample["chosen_response"].astype(str).tolist()

    rejected = sample["rejected_response"].astype(str).tolist()

    prompt_embeddings = encoder.encode(
        prompts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    chosen_embeddings = encoder.encode(
        chosen,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    rejected_embeddings = encoder.encode(
        rejected,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    def cosine_similarity_rows(a, b):

        numerator = np.sum(a * b, axis=1)

        denominator = (
            np.linalg.norm(a, axis=1)
            * np.linalg.norm(b, axis=1)
            + 1e-12
        )

        return numerator / denominator

    chosen_similarity = cosine_similarity_rows(
        prompt_embeddings,
        chosen_embeddings
    )

    rejected_similarity = cosine_similarity_rows(
        prompt_embeddings,
        rejected_embeddings
    )

    results = pd.DataFrame({
        "prompt": prompts,

        "chosen_response": chosen,

        "rejected_response": rejected,

        "prompt_chosen_similarity": chosen_similarity,

        "prompt_rejected_similarity": rejected_similarity
    })

    results.to_csv(
        OUTPUT_DIR / "semantic_similarity_results.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        [
            chosen_similarity,
            rejected_similarity
        ],

        labels=[
            "Prompt - Chosen",
            "Prompt - Rejected"
        ]
    )

    ax.set_title("Prompt-Response Semantic Similarity")

    ax.set_ylabel("Cosine Similarity")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "semantic_similarity_comparison.png",
        dpi=160
    )

    plt.close(fig)

    summary = {
        "mean_prompt_chosen_similarity": float(
            np.mean(chosen_similarity)
        ),

        "mean_prompt_rejected_similarity": float(
            np.mean(rejected_similarity)
        ),

        "sample_size": len(results)
    }

    with open(
        OUTPUT_DIR / "semantic_similarity_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    print("\nSemantic Similarity Results:")

    print(json.dumps(summary, indent=2))

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":

    run_basic_analysis()