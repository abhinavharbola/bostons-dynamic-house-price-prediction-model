import pandas as pd
import json
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Starting model training process...")

# ----------------------------
# 1. Load and Clean Data
# ----------------------------
print("Loading and cleaning data...")
data = fetch_openml(name="boston", version=1, as_frame=True, parser="pandas")
X = data.data
y = data.target.astype(float)

# Handle numeric conversions and NaNs
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.dropna()
# Align y with X after dropping NaNs
y = y[X.index]

# --- CRITICAL ETHICAL FIX ---
# Drop the 'B' feature as discussed in the review.
if 'B' in X.columns:
    X = X.drop('B', axis=1)
    print("Dropped ethically problematic 'B' feature.")
# ----------------------------

# ----------------------------
# 2. Split Data (Validation)
# ----------------------------
print("Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 3. Train Model
# ----------------------------
print("Training RandomForestRegressor...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 4. Evaluate Model
# ----------------------------
print("Evaluating model performance on test set...")
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("-" * 30)
print(f"Model Evaluation (Test Set):")
print(f"  Mean Absolute Error (MAE): {mae:.3f}")
print(f"  R-squared (R2):          {r2:.3f}")
print("-" * 30)

# ----------------------------
# 5. Save Artifacts
# ----------------------------
print("Saving model and artifacts...")

# Save the trained model
joblib.dump(model, 'model.joblib')
print("Saved model to model.joblib")

# Save the training data (for SHAP background)
# Using parquet for efficiency
X_train.to_parquet('X_train.parquet')
print("Saved SHAP background data to X_train.parquet")

# ---
# Save artifacts for the Streamlit app UI
# This prevents data leakage by only using training set stats
# ---
numeric_features = {}
categorical_features = {}
column_order = list(X.columns) # Save column order

for col in column_order:
    if pd.api.types.is_numeric_dtype(X_train[col]) and X_train[col].nunique() > 2:
        # It's a continuous numeric feature
        numeric_features[col] = {
            'min': float(X_train[col].min()),
            'max': float(X_train[col].max()),
            'mean': float(X_train[col].mean())
        }
    else:
        # It's categorical or binary (like CHAS)
        categorical_features[col] = {
            'options': [float(v) for v in X_train[col].unique()]
        }

# Combine all artifacts into one JSON
model_artifacts = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "column_order": column_order,
    "evaluation_metrics": {
        "mae": mae,
        "r2": r2
    }
}

with open('model_artifacts.json', 'w') as f:
    json.dump(model_artifacts, f, indent=4)
print("Saved app UI artifacts to model_artifacts.json")

print("\nTraining process complete. Artifacts are ready.")