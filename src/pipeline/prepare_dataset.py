import pandas as pd

df = pd.read_csv(
    r"D:\coperate-misconduct-warning\data\raw\finnlp_dataset\aaer_mark5.csv",
    sep=";",
    header=None,
    skiprows=2,
    encoding="utf-8",
    on_bad_lines="skip",
    quoting=1
)

print(f"Shape: {df.shape}")
print(f"\nColumn count: {len(df.columns)}")
print(f"\nCIK column (9): {df[9].head(5).tolist()}")
print(f"fraud_start (7): {df[7].head(5).tolist()}")
print(f"fraud_end (8):   {df[8].head(5).tolist()}")
print(f"\nMissing CIKs: {df[9].isna().sum()}")