# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:24:24 2026

@author: firas, sueda, emir

#  House Prices Prediction
"""

# Import libraries

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

# %% Load the dataset

project_root = Path(__file__).resolve().parent

train_old_path = project_root / "data" / "train.csv"
train_path = project_root / "data" / "train_cleaned.csv"
test_path = project_root / "data" / "test.csv"


# %% Clean 

dataset_old_df = pd.read_csv(train_old_path)
dataset_df = pd.read_csv(train_path)

dataset_old_df = dataset_old_df.drop(columns=["Id"])
dataset_df = dataset_df.drop(columns=["Id"])


print(f"Deafult Dataset shape is {dataset_old_df.shape}")
print(f"Cleaned Dataset shape is {dataset_df.shape}")


# %% Numerical vs Categorical 

numeric_features = dataset_df.drop(columns=["SalePrice"]).select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = dataset_df.select_dtypes(include=["object", "string"]).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Numeric features: {len(categorical_features)}")

# %% Data Transformation

# Now we have to separate the ordinal from pure categorical
ordinal_features = [
      "Street",
      "Alley",
      "LotShape",
      "Utilities",
      "LandSlope",
      "ExterQual",
      "ExterCond",
      "BsmtQual",
      "BsmtCond",
      "BsmtExposure",
      "BsmtFinType1",
      "BsmtFinType2",
      "HeatingQC",
      "CentralAir",
      "Electrical",
      "KitchenQual",
      "Functional",
      "FireplaceQu",
      "GarageFinish",
      "GarageQual",
      "GarageCond",
      "PavedDrive",
      "PoolQC",
      "Fence",
]

# NOT ordinal -> one-hot (all the other categorical)  
onehot_features = [
      col for col in categorical_features
      if col not in ordinal_features
]


quality_order = ["Po", "Fa", "TA", "Gd", "Ex"]
quality_order_with_none = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

ordinal_categories = [
      ["Grvl", "Pave"],                          # Street
      ["None", "Grvl", "Pave"],                  # Alley
      ["IR3", "IR2", "IR1", "Reg"],              # LotShape
      ["ELO", "NoSeWa", "NoSewr", "AllPub"],     # Utilities
      ["Sev", "Mod", "Gtl"],                     # LandSlope
      quality_order,                             # ExterQual
      quality_order,                             # ExterCond
      quality_order_with_none,                   # BsmtQual
      quality_order_with_none,                   # BsmtCond
      ["None", "No", "Mn", "Av", "Gd"],          # BsmtExposure
      ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],  # BsmtFinType1
      ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],  # BsmtFinType2
      quality_order,                             # HeatingQC
      ["N", "Y"],                                # CentralAir
      ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"], # Electrical
      quality_order,                             # KitchenQual
      ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"], # Functional
      quality_order_with_none,                   # FireplaceQu
      ["None", "Unf", "RFn", "Fin"],             # GarageFinish
      quality_order_with_none,                   # GarageQual
      quality_order_with_none,                   # GarageCond
      ["N", "P", "Y"],                           # PavedDrive
      ["None", "Fa", "TA", "Gd", "Ex"],          # PoolQC
      ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"], # Fence
]

# Data transforming
    # 1. numeric_transformer:
    #    Missing values -> median
    # 2. ordinal_transformer:
    #    Missing values -> "None", categories -> ordered numbers
    #    Example: [None < Po < Fa < TA < Gd < Ex] -> [0,1,2,3,4,5]
    # 3. onehot_transformer:
    #    Missing values -> frequent value , Then 1Hot
    
    #    preprocessor:
    #    Combines all transformations in one object:
    #    - numeric_features use numeric_transformer
    #    - ordinal_features use ordinal_transformer
    #    - onehot_features use onehot_transformer

numeric_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="median"))
      ]
)

ordinal_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
          ("ordinal", OrdinalEncoder(
              categories=ordinal_categories,
              handle_unknown="use_encoded_value",
              unknown_value=-1
          ))
      ]
)

onehot_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="most_frequent")),
          ("onehot", OneHotEncoder(handle_unknown="ignore"))
      ]
)

# ColumnTransformer: to apply tranformations to every column differently
preprocessor = ColumnTransformer(
      transformers=[
          ("num", numeric_transformer, numeric_features),
          ("ord", ordinal_transformer, ordinal_features),
          ("cat", onehot_transformer, onehot_features),
      ],
      sparse_threshold=0
)

print(f"Ordinal categorical features: {len(ordinal_features)}")
print(f"One-hot categorical features: {len(onehot_features)}")

# %% Cleaned, transformed data

X0 = dataset_df.drop(columns="SalePrice")
y = dataset_df["SalePrice"]
X1 = preprocessor.fit_transform(X0)

print("Original cleaned data shape:", X0.shape)
print("Transformed data shape:", X1.shape)
print("Missing values before transformation:", X0.isnull().sum().sum())
print("Missing values after transformation:", np.isnan(X1).sum())


# %% Normalization of the transformed data

scaler = StandardScaler()
X = scaler.fit_transform(X1)

print("Normalized X shape:", X.shape)
print("Missing values after normalization:", np.isnan(X).sum())

# %%  Train Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

print(f"Training set: {len(X_train)} samples, {X_train.shape[1]} features")
print(f"Validation set: {len(X_test)} samples")

# %% Models A (basic)

model_a_candidates = {
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "AdaBoost": AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
        n_estimators=100,
        random_state=RANDOM_STATE,
    ),
    "Gradient_Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "XGBoost":xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    ),
}

model_a_results = []
model_a_trained = {}

for name, model in model_a_candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(preds)))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    model_a_results.append({
            "model": name,
            "RMSE": rmse,
            "RMSLE": rmsle,
            "MAE": mae,
            "R2": r2,
    })
    model_a_trained[name] = model

# Inspect models accuracy
model_a_df = pd.DataFrame(model_a_results).sort_values("RMSLE")

# Here we could include some of the most relevant grafics 


# %% Models B (GridCV)


# %% Models C (Mean of the best Models)


# %% Model D (100% Train , 0% Test)


# %% Submissions

test_data = pd.read_csv(test_path)
test_X = test_data.drop(columns="Id")
test_X["MSSubClass"] = test_X["MSSubClass"].astype(str)
test_X = preprocessor.transform(test_X)
test_X = scaler.transform(test_X)

for model_name, model in model_a_trained.items():
    submission = pd.DataFrame({
        "Id": test_data["Id"],
        "SalePrice": np.maximum(model.predict(test_X), 0),
    })

    submission_path = project_root / "submissions" / f"submission_{model_name.lower()}.csv"
    submission.to_csv(submission_path, index=False)

print("Saved all submissions")
