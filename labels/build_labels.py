import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==========================
# Load config + tickers
# ==========================
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("ingest/tickers.yaml", "r") as f:
    tickers_config = yaml.safe_load(f)

tickers = [t["ticker"] for t in tickers_config["tickers"]]

start_date = config["project"]["start_date"]
end_date = config["project"]["end_date"]
horizon = config["labels"]["horizon_days"]

# ==========================
# Download price data
# ==========================
all_data = []

print("Downloading price data...")

for ticker in tickers:
    print(f"Fetching {ticker}")
    
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )
    
    if df.empty:
        continue

    df = df.reset_index()
    df["ticker"] = ticker
    
    # Handle possible multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Use Adjusted Close if available, otherwise Close
    if "Adj Close" in df.columns:
        df["adj_close"] = df["Adj Close"]
    elif "Close" in df.columns:
        df["adj_close"] = df["Close"]
    else:
        raise ValueError("No Close or Adj Close column found.")

    
    # Forward return
    df["forward_return"] = (
        df["adj_close"].shift(-horizon) - df["adj_close"]
    ) / df["adj_close"]
    
    # Classification label
    df["label"] = (df["forward_return"] > 0).astype(int)
    
    # Drop rows where forward return not available
    df = df.dropna(subset=["forward_return"])
    
    all_data.append(
        df[["Date", "ticker", "adj_close", "forward_return", "label"]]
    )

# ==========================
# Combine and save
# ==========================
final_df = pd.concat(all_data)

output_path = "data/processed/price_labels.parquet"
final_df.to_parquet(output_path, index=False)

print(f"Saved labels to {output_path}")
print(f"Total rows: {len(final_df)}")
