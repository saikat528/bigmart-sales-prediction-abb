# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:17:54 2026

@author: SAIKAT GHOSH
"""

# ============================================================
# BIGMART SALES PREDICTION
# Advanced Feature Engineering + Stage-wise Tuning
# ============================================================

import numpy as np
import pandas as pd
from itertools import product

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

test_keys = test_df[["Item_Identifier", "Outlet_Identifier"]].copy()

# ============================================================
# 2. BASIC CLEANING
# ============================================================
def basic_cleaning(df):
    df = df.copy()

    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({
        "low fat": "Low Fat",
        "LF": "Low Fat",
        "reg": "Regular"
    })

    df["Item_Group"] = df["Item_Identifier"].str[:2].map({
        "FD": "Food",
        "DR": "Drinks",
        "NC": "Non-Consumable"
    })

    df.loc[df["Item_Group"] == "Non-Consumable", "Item_Fat_Content"] = "NA"
    return df


train_df = basic_cleaning(train_df)
test_df = basic_cleaning(test_df)

# ============================================================
# 3. TRAIN-ONLY IMPUTATION
# ============================================================
weight_map = train_df.groupby("Item_Identifier")["Item_Weight"].median()
global_weight = train_df["Item_Weight"].median()

for df in [train_df, test_df]:
    df["Item_Weight"] = df["Item_Weight"].fillna(
        df["Item_Identifier"].map(weight_map)
    ).fillna(global_weight)

outlet_size_map = (
    train_df.groupby("Outlet_Type")["Outlet_Size"]
    .agg(lambda x: x.mode()[0])
)

for df in [train_df, test_df]:
    df["Outlet_Size"] = df["Outlet_Size"].fillna(
        df["Outlet_Type"].map(outlet_size_map)
    )

# ============================================================
# 4. FIX ZERO VISIBILITY
# ============================================================
visibility_map = train_df.groupby("Item_Type")["Item_Visibility"].median()

for df in [train_df, test_df]:
    zero_mask = df["Item_Visibility"] == 0
    df.loc[zero_mask, "Item_Visibility"] = (
        df.loc[zero_mask, "Item_Type"].map(visibility_map)
    )

# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    df = df.copy()

    df["Outlet_Age"] = 2013 - df["Outlet_Establishment_Year"]
    df["Outlet_Maturity_Score"] = np.log1p(df["Outlet_Age"])

    df["Visibility_Index"] = (
        df["Item_Visibility"] /
        df.groupby("Item_Type")["Item_Visibility"].transform("mean")
    )

    df["Outlet_Visibility_Ratio"] = (
        df["Item_Visibility"] /
        df.groupby("Outlet_Identifier")["Item_Visibility"].transform("mean")
    )

    df["Price_Weight_Ratio"] = df["Item_MRP"] / df["Item_Weight"]

    df["Price_Relative_Mean"] = (
        df["Item_MRP"] /
        df.groupby("Item_Type")["Item_MRP"].transform("mean")
    )

    df["Item_Type_Density"] = (
        df.groupby("Item_Type")["Item_Type"].transform("count") / df.shape[0]
    )

    df["Is_LowFat_Food"] = (
        (df["Item_Fat_Content"] == "Low Fat") &
        (df["Item_Group"] == "Food")
    ).astype(int)

    df["Visibility_Quality"] = np.where(
        df["Item_Visibility"] < 0.01, 0,
        np.where(df["Item_Visibility"] > 0.2, 0.5, 1)
    )

    df["Outlet_Item_Variety"] = (
        df.groupby("Outlet_Identifier")["Item_Type"].transform("nunique")
    )

    df["MRP_Band"] = pd.qcut(
        df["Item_MRP"], q=4,
        labels=["Low", "Medium", "High", "Premium"]
    ).astype(str)

    df["Item_Outlet_Exposure"] = (
        df.groupby(["Item_Type", "Outlet_Type"])["Item_Visibility"]
        .transform("mean")
    )

    return df


train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# ============================================================
# 6. DROP IDENTIFIERS
# ============================================================
drop_cols = [
    "Item_Identifier",
    "Outlet_Identifier",
    "Outlet_Establishment_Year"
]

train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# ============================================================
# 7. SPLIT X / y
# ============================================================
y = train_df["Item_Outlet_Sales"]
X = train_df.drop("Item_Outlet_Sales", axis=1)

# ============================================================
# 8. ENCODING
# ============================================================
cat_cols = X.select_dtypes(include="object").columns

try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

encoder.fit(X[cat_cols])

X_final = np.hstack([
    X.drop(columns=cat_cols).values,
    encoder.transform(X[cat_cols])
])

test_final = np.hstack([
    test_df.drop(columns=cat_cols).values,
    encoder.transform(test_df[cat_cols])
])

# ============================================================
# 9. CV RMSE FUNCTION
# ============================================================
def cv_rmse(model, X, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rmses = []

    for tr, val in kf.split(X):
        model.fit(X[tr], y.iloc[tr])
        preds = model.predict(X[val])
        rmses.append(
            np.sqrt(mean_squared_error(y.iloc[val], preds))
        )

    return np.mean(rmses)

# ============================================================
# 10. STAGE-WISE HYPERPARAMETER TUNING (HistGB)
# ============================================================
print("\nStage 1: Coarse Search")

stage1_grid = {
    "learning_rate": [0.02, 0.04, 0.06],
    "max_depth": [5, 7, 9],
    "min_samples_leaf": [20, 30, 40]
}

best_rmse = np.inf
best_params = None

for values in product(*stage1_grid.values()):
    params = dict(zip(stage1_grid.keys(), values))
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    rmse = cv_rmse(model, X_final, y)
    print(f"{params} -> RMSE: {rmse:.2f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params

print("\nStage 2: Refined Search")

stage2_grid = {
    "learning_rate": [
        best_params["learning_rate"] * 0.8,
        best_params["learning_rate"],
        best_params["learning_rate"] * 1.2
    ],
    "max_depth": [
        best_params["max_depth"] - 1,
        best_params["max_depth"],
        best_params["max_depth"] + 1
    ],
    "min_samples_leaf": [
        int(best_params["min_samples_leaf"] * 0.8),
        best_params["min_samples_leaf"],
        int(best_params["min_samples_leaf"] * 1.2)
    ]
}

for values in product(*stage2_grid.values()):
    params = dict(zip(stage2_grid.keys(), values))
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    rmse = cv_rmse(model, X_final, y)
    print(f"{params} -> RMSE: {rmse:.2f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params

print("\nStage 3: Regularization Search")

stage3_grid = {
    "l2_regularization": [0.0, 0.1, 0.3],
    "max_bins": [200,255]
}

final_params = best_params.copy()

for values in product(*stage3_grid.values()):
    params = final_params | dict(zip(stage3_grid.keys(), values))
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    rmse = cv_rmse(model, X_final, y)
    print(f"{params} -> RMSE: {rmse:.2f}")

    if rmse < best_rmse:
        best_rmse = rmse
        final_params = params



if __name__ == "__main__":
    # ============================================================
    # 11. TRAIN FINAL MODEL
    # ============================================================
    print("\n==============================")
    print("FINAL MODEL (OPTIMIZED)")
    print("==============================")
    print(f"Parameters: {final_params}")
    print(f"CV RMSE   : {best_rmse:.2f}")

    final_model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        **final_params
    )

    final_model.fit(X_final, y)

    # ============================================================
    # 12. PREDICT TEST
    # ============================================================
    test_preds = np.maximum(0, final_model.predict(test_final))

    submission = test_keys.copy()
    submission["Item_Outlet_Sales"] = test_preds
    submission.to_csv("data/submission.csv", index=False)

    print("\nsubmission.csv created successfully")