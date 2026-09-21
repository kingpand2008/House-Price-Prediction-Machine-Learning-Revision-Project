import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("housing_price_dataset.csv")

print("\n========== DATASET ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

columns_to_remove = [
    "price_in_lakhs",
    "property_id",
    "price_category"
]

X = df.drop(columns=columns_to_remove)
y = df["price_in_lakhs"]
#----------------------------------------------
numerical_f=X.select_dtypes(include=["int64", "float64"]).columns
categorical_f=X.select_dtypes(include=["str"]).columns

# ============================================================
# 4. NUMERICAL PIPELINE
# ============================================================

numerical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scalar",StandardScaler())
])

# ============================================================
# 5. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

# ============================================================
# 6. COMBINE NUMERICAL + CATEGORICAL PROCESSING
# ============================================================

preprocessor = ColumnTransformer([

    ("num", numerical_pipeline, numerical_f),

    ("cat", categorical_pipeline, categorical_f)
])

# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN / TEST ==========")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ============================================================
# 8. LINEAR REGRESSION
# ============================================================

linear_model = Pipeline([

    ("preprocessor", preprocessor),

    ("model", LinearRegression())
])

print("\nTraining Linear Regression...")

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

# ============================================================
# 9. LINEAR REGRESSION EVALUATION
# ============================================================

linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_mse = mean_squared_error(
    y_test,
    linear_pred
)

linear_rmse = linear_mse ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_pred
)

print("\n========== LINEAR REGRESSION ==========")

print("MAE :", linear_mae)
print("MSE :", linear_mse)
print("RMSE:", linear_rmse)
print("R²  :", linear_r2)

# ============================================================
# 10. RIDGE REGRESSION
# ============================================================

ridge_model = Pipeline([

    ("preprocessor", preprocessor),

    ("model", Ridge(alpha=1.0))
])

print("\nTraining Ridge Regression...")

ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

ridge_mae = mean_absolute_error(
    y_test,
    ridge_pred
)

ridge_mse = mean_squared_error(
    y_test,
    ridge_pred
)

ridge_rmse = ridge_mse ** 0.5

ridge_r2 = r2_score(
    y_test,
    ridge_pred
)

print("\n========== RIDGE REGRESSION ==========")

print("MAE :", ridge_mae)
print("MSE :", ridge_mse)
print("RMSE:", ridge_rmse)
print("R²  :", ridge_r2)

# ============================================================
# 11. LASSO REGRESSION
# ============================================================

lasso_model = Pipeline([

    ("preprocessor", preprocessor),

    ("model", Lasso(
        alpha=0.1,
        max_iter=5000
    ))
])

print("\nTraining Lasso Regression...")

lasso_model.fit(X_train, y_train)

lasso_pred = lasso_model.predict(X_test)

lasso_mae = mean_absolute_error(
    y_test,
    lasso_pred
)

lasso_mse = mean_squared_error(
    y_test,
    lasso_pred
)

lasso_rmse = lasso_mse ** 0.5

lasso_r2 = r2_score(
    y_test,
    lasso_pred
)

print("\n========== LASSO REGRESSION ==========")

print("MAE :", lasso_mae)
print("MSE :", lasso_mse)
print("RMSE:", lasso_rmse)
print("R²  :", lasso_r2)

# ============================================================
# 12. RANDOM FOREST
# ============================================================

forest_model = Pipeline([

    ("preprocessor", preprocessor),

    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")

forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)

forest_mae = mean_absolute_error(
    y_test,
    forest_pred
)

forest_mse = mean_squared_error(
    y_test,
    forest_pred
)

forest_rmse = forest_mse ** 0.5

forest_r2 = r2_score(
    y_test,
    forest_pred
)

print("\n========== RANDOM FOREST ==========")

print("MAE :", forest_mae)
print("MSE :", forest_mse)
print("RMSE:", forest_rmse)
print("R²  :", forest_r2)

# ============================================================
# 13. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        ridge_mae,
        lasso_mae,
        forest_mae
    ],

    "RMSE": [
        linear_rmse,
        ridge_rmse,
        lasso_rmse,
        forest_rmse
    ],

    "R2": [
        linear_r2,
        ridge_r2,
        lasso_r2,
        forest_r2
    ]
})

print("\n========== MODEL COMPARISON ==========")

print(results.to_string(index=False))

# ============================================================
# 14. PREDICT ONE HOUSE
# ============================================================

# IMPORTANT:
# These column names MUST exist in your dataset.
#
# Change the values to whatever house you want to test.

new_house = pd.DataFrame({

    "bhk": [3],

    "bathrooms": [2],

    "balconies": [2],

    "built_up_area": [1500],

    "carpet_area": [1200],

    "floor_number": [3],

    "total_floors": [10],

    "property_age": [5],

    "parking_spaces": [1],

    "security_score": [8],

    "gym_available": [1],

    "swimming_pool": [0],

    "power_backup": [1],

    "lift_available": [1],

    "maintenance_fee_monthly": [3000],

    "distance_to_city_center_km": [8],

    "distance_to_metro_km": [2],

    "nearby_schools": [5],

    "nearby_hospitals": [3],

    "city": ["Chennai"],

    "locality": ["Velachery"],

    "locality_tier": ["Tier 1"],

    "property_type": ["Apartment"],

    "floor_category": ["Mid Floor"],

    "facing": ["East"],

    "furnishing_status": ["Semi-Furnished"],

    "transaction_type": ["Resale"]
})

# ============================================================
# 15. MAKE PREDICTION
# ============================================================

prediction = linear_model.predict(new_house)

print("\n========== HOUSE PRICE PREDICTION ==========")

print(f"Predicted house price: "f"{prediction[0]:.2f} lakhs")
