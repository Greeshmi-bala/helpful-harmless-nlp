
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("models/model_comparison.csv")

print(results.to_string(index=False))

results.plot(
    x="model",
    y=["accuracy", "precision", "recall", "f1_score"],
    kind="bar",
    figsize=(10, 6)
)

plt.title("ML Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("models/model_comparison.png", dpi=150)
plt.show()
