
import pandas as pd

df = pd.read_csv("data/hh_rlhf_cleaned.csv")

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3).to_string())
print("\nMissing values:")
print(df.isnull().sum())
