import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from ddos.config import PROCESSED_DIR

df = pd.read_csv(PROCESSED_DIR / "final_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]


rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

df_balanced = pd.DataFrame(X_res, columns=X.columns)
df_balanced["label"] = y_res

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df_balanced.to_csv(PROCESSED_DIR / "final_balanced_dataset.csv", index=False)

print(df_balanced["label"].value_counts())
print(f"\nTotal rows: {len(df_balanced)}")
