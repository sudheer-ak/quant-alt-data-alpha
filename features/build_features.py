import pandas as pd
import numpy as np

print("Loading price labels...")

df = pd.read_parquet("data/processed/price_labels.parquet")

df = df.sort_values(["ticker", "Date"])

# ==========================
# Feature Engineering
# ==========================

def add_features(group):
    group = group.copy()
    
    # 5-day momentum
    group["mom_5"] = group["adj_close"].pct_change(5)
    
    # 20-day momentum
    group["mom_20"] = group["adj_close"].pct_change(20)
    
    # 20-day rolling volatility
    group["vol_20"] = group["adj_close"].pct_change().rolling(20).std()
    
    # Deviation from 20-day mean
    group["mean_20"] = group["adj_close"].rolling(20).mean()
    group["price_vs_mean_20"] = (
        group["adj_close"] - group["mean_20"]
    ) / group["mean_20"]
    
    return group

df = (
    df.groupby("ticker", group_keys=False)
      .apply(add_features)
      .reset_index(drop=True)
)

# Drop early rows with NaN from rolling windows
df = df.dropna()

output_path = "data/features/price_features.parquet"
df.to_parquet(output_path, index=False)

print(f"Saved features to {output_path}")
print(f"Total rows after feature engineering: {len(df)}")
