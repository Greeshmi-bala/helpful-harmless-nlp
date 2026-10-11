
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion


def load_data():
    df = pd.read_csv("data/hh_rlhf_cleaned.csv")

    chosen = pd.DataFrame({
        "prompt": df["prompt"],
        "response": df["chosen_response"],
        "label": 1
    })

    rejected = pd.DataFrame({
        "prompt": df["prompt"],
        "response": df["rejected_response"],
        "label": 0
    })

    data = pd.concat([chosen, rejected], ignore_index=True)
    data = data.dropna(subset=["prompt", "response"])
    data = data.drop_duplicates(subset=["prompt", "response", "label"])

    data["response_length"] = data["response"].str.len()
    data["word_count"] = data["response"].str.split().str.len()
    data["sentence_count"] = data["response"].str.count(r"[.!?]+")

    data["vocabulary_diversity"] = data["response"].apply(
        lambda text: len(set(text.lower().split())) / max(len(text.split()), 1)
    )

    return data


def build_features():
    word_tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=5000
    )

    text_features = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf)
    ])

    return text_features


if __name__ == "__main__":
    data = load_data()
    print("Processed shape:", data.shape)
    print(data["label"].value_counts())
    print("\nFeature engineering setup is ready.")
