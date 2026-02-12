import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Loading features...")

df = pd.read_parquet("data/features/price_features.parquet")

# Select feature columns
feature_cols = [
    "mom_5",
    "mom_20",
    "vol_20",
    "price_vs_mean_20"
]

X = df[feature_cols]
y = df["label"]

# Simple time split (NOT random)
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

# Predict
preds = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, preds)

print(f"Baseline Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds))
