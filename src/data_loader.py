from datasets import load_dataset


def load_hh_rlhf():
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="helpful-base",
        streaming=True
    )

    return dataset


if __name__ == "__main__":
    ds = load_hh_rlhf()

    print(ds)

    sample = next(iter(ds["train"]))

    print("\n--- CHOSEN ---")
    print(sample["chosen"])

    print("\n--- REJECTED ---")
    print(sample["rejected"])