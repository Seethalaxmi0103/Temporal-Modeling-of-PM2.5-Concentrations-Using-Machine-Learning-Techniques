# ================================================================
# Project: Predictive Analytics for Air Quality Monitoring
# Dataset: air_quality.csv
# Author: Seethalaxmi V
# ================================================================

# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- Step 2: Load Dataset ---
print("Loading dataset...")
df = pd.read_csv("air-quality-india.csv")   # Ensure your CSV file is in the same folder
print("Dataset Loaded Successfully ✅")
print(df.head())

# --- Step 3: Check and Rename Columns (if needed) ---
print("\nColumn Names:", df.columns.tolist())
# Rename columns to consistent format
df.columns = [col.strip().lower().replace('.', '_') for col in df.columns]

# Example: if column names differ, adjust here
# df.rename(columns={'PM2.5': 'pm2_5'}, inplace=True)

# --- Step 4: Handle Missing Values ---
df['pm2_5'].fillna(df['pm2_5'].mean(), inplace=True)

# --- Step 5: Convert Timestamp to Datetime ---
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
else:
    # Create timestamp column if missing
    df['timestamp'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=df['day'], hour=df['hour']))

# --- Step 6: Feature Engineering ---
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# --- Step 7: Define Features and Target ---
X = df[['year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend']]
y = df['pm2_5']

# --- Step 8: Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# --- Step 9: Train Model ---
print("\nTraining Random Forest Model...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Model Trained Successfully ✅")

# --- Step 10: Evaluate Model ---
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# --- Step 11: Visualize Actual vs Predicted ---
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual PM2.5', color='blue')
plt.plot(y_pred[:100], label='Predicted PM2.5', color='red', linestyle='dashed')
plt.title("PM2.5 Prediction vs Actual (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("PM2.5 Concentration")
plt.legend()
plt.show()

# --- Step 12: Save Trained Model ---
joblib.dump(model, "air_quality_model.pkl")
print("\nModel saved as 'air_quality_model.pkl' ✅")

# --- Step 13: Optional - Predict Future PM2.5 ---
print("\nPredicting Future PM2.5 Levels (Next 24 hours)...")

# Example: next 24 hours of 13 Nov 2025
future_dates = pd.DataFrame({
    'year': [2025] * 24,
    'month': [11] * 24,
    'day': [13] * 24,
    'hour': list(range(24))
})

# Generate time features
future_dates['day_of_week'] = 3  # Example: Wednesday
future_dates['is_weekend'] = 0

# Predict
future_pred = model.predict(future_dates)
future_dates['Predicted_PM2.5'] = future_pred

print(future_dates.head())

# Plot future predictions
plt.figure(figsize=(10, 5))
plt.plot(future_dates['hour'], future_dates['Predicted_PM2.5'], marker='o')
plt.title("Predicted PM2.5 for Next 24 Hours")
plt.xlabel("Hour of the Day")
plt.ylabel("PM2.5 Concentration")
plt.grid(True)
plt.show()

print("\n✅ Air Quality Prediction Completed Successfully!")
