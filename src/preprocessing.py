import re
import pandas as pd


def extract_conversation_parts(conversation):
    """
    Extract the final user prompt and final assistant response
    from an HH-RLHF conversation.
    """

    turns = re.split(r"\n\n(?=(?:Human|Assistant):)", conversation)

    human_messages = []
    assistant_messages = []

    for turn in turns:
        turn = turn.strip()

        if turn.startswith("Human:"):
            human_messages.append(turn[len("Human:"):].strip())

        elif turn.startswith("Assistant:"):
            assistant_messages.append(turn[len("Assistant:"):].strip())

    if not human_messages or not assistant_messages:
        return None, None

    prompt = human_messages[-1]
    response = assistant_messages[-1]

    return prompt, response


def clean_text(text):
    """
    Basic text cleaning.
    """

    if text is None:
        return ""

    # Replace multiple spaces/newlines with one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def create_dataframe(dataset, max_samples=10000):
    """
    Convert HH-RLHF dataset into a Pandas DataFrame.
    """

    records = []

    for i, sample in enumerate(dataset):

        if i >= max_samples:
            break

        chosen_prompt, chosen_response = extract_conversation_parts(
            sample["chosen"]
        )

        rejected_prompt, rejected_response = extract_conversation_parts(
            sample["rejected"]
        )

        # Skip malformed examples
        if not chosen_prompt or not chosen_response:
            continue

        if not rejected_response:
            continue

        # Clean text
        chosen_prompt = clean_text(chosen_prompt)
        chosen_response = clean_text(chosen_response)
        rejected_response = clean_text(rejected_response)

        records.append({
            "prompt": chosen_prompt,
            "chosen_response": chosen_response,
            "rejected_response": rejected_response
        })

    return pd.DataFrame(records)


def validate_dataframe(df):
    """
    Check the dataset for missing values,
    duplicates and identical responses.
    """

    print("\n===== DATASET VALIDATION =====")

    print("\nShape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    identical = (
        df["chosen_response"] == df["rejected_response"]
    )

    print("\nIdentical chosen/rejected responses:")
    print(identical.sum())

    print(
        "\nPercentage of identical responses:",
        round(identical.mean() * 100, 2),
        "%"
    )

    return identical


def remove_invalid_rows(df):
    """
    Remove invalid or unusable preference pairs.
    """

    before = len(df)

    # Remove missing values
    df = df.dropna(
        subset=[
            "prompt",
            "chosen_response",
            "rejected_response"
        ]
    )

    # Remove empty strings
    df = df[
        (df["prompt"].str.strip() != "") &
        (df["chosen_response"].str.strip() != "") &
        (df["rejected_response"].str.strip() != "")
    ]

    # Remove identical chosen/rejected responses
    df = df[
        df["chosen_response"] != df["rejected_response"]
    ]

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    after = len(df)

    print(f"\nRows before cleaning: {before}")
    print(f"Rows after cleaning: {after}")
    print(f"Rows removed: {before - after}")

    return df.reset_index(drop=True)


if __name__ == "__main__":

    from data_loader import load_hh_rlhf

    # Load dataset
    ds = load_hh_rlhf()

    # Create DataFrame from first 10,000 training examples
    df = create_dataframe(
        ds["train"],
        max_samples=10000
    )

    # Validate
    validate_dataframe(df)

    # Clean
    df = remove_invalid_rows(df)

    # Display sample
    print("\n===== FINAL DATASET =====")

    print(df.head())

    print("\nFinal shape:")
    print(df.shape)
    # Save cleaned dataset
    output_path = "../data/hh_rlhf_cleaned.csv"

    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to: {output_path}")