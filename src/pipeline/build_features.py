import configs.settings as settings
import pandas as pd

df = pd.read_parquet(settings.FEATURES_DIR / "trainval_features.parquet")

print(df.columns.tolist())
print(df.shape)
print(df.dtypes)
