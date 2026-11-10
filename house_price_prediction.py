"""
HOUSE PRICE PREDICTION | REGRESSION - CAPSTONE PROJECT
Author: Mthunzi Ntombela
GitHub: https://github.com/Mthunzy
Dataset: California Housing - 20,640 houses
Goal: Predict house price based on location, rooms, income, age
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------
print("Loading data...")
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print(f"Total houses: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(df.head())
print(df.info())

# ------------------------------------------------
# 2. EXPLORE DATA
# ------------------------------------------------
print("\n--- Summary Statistics ---")
print(df.describe())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# ------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------
print(f"\nBefore cleaning: {df.shape}")
df_clean = df.dropna()
print(f"After cleaning: {df_clean.shape}")

# ------------------------------------------------
# 4. PREPARE FEATURES (X) AND TARGET (y)
# ------------------------------------------------
# Target: median_house_value = Price we want to predict
# Features: Everything else except price and ocean_proximity text
X = df_clean.drop(['median_house_value', 'ocean_proximity'], axis=1)
y = df_clean['median_house_value']

print(f"\nFeatures used: {list(X.columns)}")
print(f"Target: median_house_value")
print(f"Average house price: R{y.mean():,.0f}")
print(f"Most expensive house: R{y.max():,.0f}")
print(f"Cheapest house: R{y.min():,.0f}")

# ------------------------------------------------
# 5. SPLIT DATA 80% TRAIN / 20% TEST
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)} houses")
print(f"Test size: {len(X_test)} houses")

# ------------------------------------------------
# 6. MODEL 1 - LINEAR REGRESSION (Baseline)
# ------------------------------------------------
print("\n--- Training Model 1: Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print(f"Linear Regression Results:")
print(f"MAE: {lr_mae:,.0f} - Average error per house")
print(f"RMSE: {lr_rmse:,.0f}")
print(f"R2 Score: {lr_r2:.2%} - Explains {lr_r2:.0%} of price variation")

# ------------------------------------------------
# 7. MODEL 2 - RANDOM FOREST REGRESSION (Better Model)
# ------------------------------------------------
print("\n--- Training Model 2: Random Forest ---")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest Results:")
print(f"MAE: {rf_mae:,.0f} - Average error per house")
print(f"RMSE: {rf_rmse:,.0f}")
print(f"R2 Score: {rf_r2:.2%} - Explains {rf_r2:.0%} of price variation")

# ------------------------------------------------
# 8. COMPARE MODELS
# ------------------------------------------------
print("\n--- MODEL COMPARISON ---")
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [lr_mae, rf_mae],
    'RMSE': [lr_rmse, rf_rmse],
    'R2 Score': [lr_r2, rf_r2]
})
print(comparison)

if rf_r2 > lr_r2:
    print("\nWINNER: Random Forest is better!")
    print(f"Improvement: {(rf_r2 - lr_r2)*100:.1f}% better than Linear Regression")
else:
    print("\nWINNER: Linear Regression is better!")

# ------------------------------------------------
# 9. FEATURE IMPORTANCE - What drives house price?
# ------------------------------------------------
importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Predictors of House Price:")
print(importances)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
importances.plot(kind='bar')
plt.title('What Drives House Price?')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance chart saved as feature_importance.png")

# ------------------------------------------------
# 10. BUSINESS CONCLUSION
# ------------------------------------------------
print("\n--- BUSINESS INSIGHT ---")
print("1. Location (latitude/longitude) is the #1 driver of house price.")
print("2. Median income in the area is the #2 driver - richer area = higher price.")
print("3. Random Forest beats Linear Regression for house prices.")
print("4. Estate agents can use this to price houses fairly and quickly.")
print("5. Buyers can use this to check if a house is overpriced.")
