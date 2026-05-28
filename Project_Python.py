"""
Created on Mon May 25 19:24:24 2026

@author: firas, sueda, emir

#  House Prices Prediction
"""

# Import libraries
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

#Load the dataset
project_root = Path(__file__).resolve().parent
train_path = project_root / "data" / "train.csv"
test_path = project_root / "data" / "test.csv"

##########################################################
# %% Data Cleaning - (from the file data_analyze.ipynb)
##########################################################


dataset_df = pd.read_csv(train_path)
dataset_df = dataset_df.drop(columns=["Id"]) #Remove Id columns
dataset_original_df = dataset_df.copy()

none_cols = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType"
]

#instead of blindly filling all missing values we analyze and inspect first 
#Replace the missing values with "None" for the columns that have missing values representing the absence of a feature ex. features(e.g., no pool, no garage, etc.)
for col in none_cols:
    dataset_df[col] = dataset_df[col].fillna("None")

# Fill remaining numerical missing values with median (or 0 if it makes more sense as in the example of GarageYrBlt and MasVnrArea)
dataset_df["LotFrontage"] = dataset_df["LotFrontage"].fillna(dataset_df["LotFrontage"].median())
dataset_df["GarageYrBlt"] = dataset_df["GarageYrBlt"].fillna(0)
dataset_df["MasVnrArea"] = dataset_df["MasVnrArea"].fillna(0)

# Fill remaining categorical missing value
dataset_df["Electrical"] = dataset_df["Electrical"].fillna(dataset_df["Electrical"].mode()[0])

# Convert MSSubClass to categorical
dataset_df["MSSubClass"] = dataset_df["MSSubClass"].astype(str)

print("-------------- Cleaning data: -----------------")
print("Missing values before cleaning:", dataset_original_df.isnull().sum().sum())
print("Total missing values after cleaning:", dataset_df.isnull().sum().sum())
print(f"Cleaned Dataset shape is {dataset_df.shape}")


###########################################################
# %% Data Transformation
###########################################################
# Now we separate the ordinal features from categorical ones

#Numerical vs Categorical features seperation
numeric_features = dataset_df.drop(columns=["SalePrice"]).select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = dataset_df.select_dtypes(include=["object", "string"]).columns.tolist()

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

# Typical order
quality_order = ["Po", "Fa", "TA", "Gd", "Ex"]
quality_order_with_none = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

# We need to specify the order of categories for ordinal features, so that they can be transformed into ordered numbers
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
    #    Missing values -> "None"
    #    categories -> ordered numbers
    #    Example: [None < Po < Fa < TA < Gd < Ex] -> [0,1,2,3,4,5]
    # 3. onehot_transformer:
    #    Missing values -> frequent value , then 1Hot
    
    #    preprocessor:
    #    Combines all transformations in one object:
    #    - numeric_features use numeric_transformer
    #    - ordinal_features use ordinal_transformer
    #    - onehot_features use onehot_transformer


numeric_transformer = Pipeline(
      steps=[("imputer", SimpleImputer(strategy="median"))])

ordinal_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
          ("ordinal", OrdinalEncoder(
              categories=ordinal_categories,
              handle_unknown="use_encoded_value",
              unknown_value=-1))])

onehot_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="most_frequent")),
          ("onehot", OneHotEncoder(handle_unknown="ignore"))])

# ColumnTransformer: to apply tranformations to every column differently
preprocessor = ColumnTransformer(
      transformers=[
          ("num", numeric_transformer, numeric_features),
          ("ord", ordinal_transformer, ordinal_features),
          ("cat", onehot_transformer, onehot_features),],
      sparse_threshold=0)


# Cleaned, transformed data
X0 = dataset_df.drop(columns="SalePrice")   # X original
y = np.log1p(dataset_df["SalePrice"])       # y in log scale
X1 = preprocessor.fit_transform(X0)         # X cleaned

# Normalization
scaler = StandardScaler()
X = scaler.fit_transform(X1)                # X normalized

def to_price(log_values):
    return np.maximum(np.expm1(log_values), 0)

print("-------------- Transforming data: -----------------")
print("Original cleaned data shape:", X0.shape)
print("Transformed data shape:", X1.shape)

#############################################################
# %% Train Data
#############################################################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,)

print("-------------- Training data: -----------------")
print(f"Training set: {len(X_train)} samples, {X_train.shape[1]} features")
print(f"Validation set: {len(X_test)} samples")

############################################################
# %% MODEL A (basic) - DEFAULT MODELS
############################################################

model_a_candidates = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsRegressor(
        n_neighbors=10,
        weights="distance",
        algorithm="auto"
    ),
    "SVR": SVR(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "AdaBoost": AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE),
        n_estimators=200,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    ),
    "Gradient_Boosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE
    ),
    "XGBoost":xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        n_jobs=-1,
    ),
}

model_a_results = []
model_a_trained = {}

for name, model in model_a_candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_price = to_price(preds)
    y_test_price = to_price(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_price, preds_price))
    rmsle = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test_price, preds_price)
    r2 = r2_score(y_test_price, preds_price)

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


# Other models that could be intresting
# Ridge
# Lasso
# ElasticNet
# Voting Regressor

############################################################
# %% MODEL A Graphs
############################################################

# RMSLE comparison
plt.figure(figsize=(10, 5))
sns.barplot(
    data=model_a_df,
    x="RMSLE",
    y="model"
)
plt.title("Model A - Default Models Compared by RMSLE")
plt.xlabel("RMSLE lower is better")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# # RMSE comparison
# plt.figure(figsize=(10, 5))
# sns.barplot(
#     data=model_a_df.sort_values("RMSE"),
#     x="RMSE",
#     y="model"
# )
# plt.title("Model A - Default Models Compared by RMSE")
# plt.xlabel("RMSE lower is better")
# plt.ylabel("Model")
# plt.tight_layout()
# plt.show()

# # R2 comparison
# plt.figure(figsize=(10, 5))
# sns.barplot(
#     data=model_a_df.sort_values("R2", ascending=False),
#     x="R2",
#     y="model"
# )
# plt.title("Model A - Default Models Compared by R²")
# plt.xlabel("R² higher is better")
# plt.ylabel("Model")
# plt.tight_layout()
# plt.show()

#############################################################
# %% Models B (GridSearchCV) Playing with hyperparameters 
# we'll be using the best 3 models from model A to do hyperparameter tuning with GridSearchCV
# Random Forest, Gradient Boosting and XGBoost
#############################################################

def rmsle_metric(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# GridSearchCV maximizes the score.
# Because RMSLE is an error metric and lower is better,
# we use greater_is_better=False.

rmsle_scorer = make_scorer(
    rmsle_metric,
    greater_is_better=False
)

cv_strategy = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

###########################
# %% The best 3 models
###########################

#this part might take a while to run because of the large hyperparameter grid and the use of cross-validation !!!

model_b_candidates = {
    "XGBoost": {
        "model": xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": [200, 300],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    },

    "Random Forest": {
        "model": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": [200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 0.8]
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "subsample": [0.8, 1.0]
        }
    }
}

#############################################################
# %% MODEL B CACHE - Save GridSearchCV results
#############################################################

model_cache_dir = project_root / "saved_models"
model_cache_dir.mkdir(exist_ok=True)

model_b_cache_path = model_cache_dir / "model_b_gridsearch_results_log_target.pkl"

if model_b_cache_path.exists():
    try:
        print("\nLoading saved Model B results...")
        
        saved_model_b = joblib.load(model_b_cache_path)
        
        model_b_df = saved_model_b["model_b_df"]
        model_b_trained = saved_model_b["model_b_trained"]
        model_b_results = saved_model_b["model_b_results"]
    
        print("Model B loaded successfully. GridSearchCV was skipped.")
    except:
        print("Incopatible Cache, delete it.")
else:
    print("\nNo saved Model B found. Running GridSearchCV for the first time...")

    model_b_results = []
    model_b_trained = {}

    for name, item in model_b_candidates.items():
        print(f"\nTuning {name} with GridSearchCV...")

        grid_search = GridSearchCV(
            estimator=item["model"],
            param_grid=item["params"],
            scoring=rmsle_scorer,
            cv=cv_strategy,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        preds = best_model.predict(X_test)
        preds_price = to_price(preds)
        y_test_price = to_price(y_test)

        rmse = np.sqrt(mean_squared_error(y_test_price, preds_price))
        rmsle = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test_price, preds_price)
        r2 = r2_score(y_test_price, preds_price)

        model_b_results.append({
            "model": name,
            "best_params": grid_search.best_params_,
            "CV_RMSLE": -grid_search.best_score_,
            "Test_RMSE": rmse,
            "Test_RMSLE": rmsle,
            "Test_MAE": mae,
            "Test_R2": r2
        })

        model_b_trained[name] = best_model

        print(f"\nBest parameters for {name}:")
        print(grid_search.best_params_)

        print(f"\n{name} tuned results:")
        print(f"CV RMSLE: {-grid_search.best_score_:.4f}")
        print(f"Test RMSLE: {rmsle:.4f}")
        print(f"Test RMSE: {rmse:.2f}")
        print(f"Test MAE: {mae:.2f}")
        print(f"Test R2: {r2:.4f}")

    model_b_df = pd.DataFrame(model_b_results).sort_values("Test_RMSLE")

    saved_model_b = {
        "model_b_df": model_b_df,
        "model_b_trained": model_b_trained,
        "model_b_results": model_b_results
    }

    joblib.dump(saved_model_b, model_b_cache_path)

    print(f"\nModel B saved successfully to: {model_b_cache_path}")
print("\nMODEL B - GridSearchCV Results")
print(
    model_b_df[
        [
            "model",
            "CV_RMSLE",
            "Test_RMSLE",
            "Test_RMSE",
            "Test_MAE",
            "Test_R2"
        ]
    ].to_string(index=False)
)

print("\nMODEL A - Default Model Results (for comparation)")
print(model_a_df.to_string(index=False))

#####################################################
# %% Models C (Mean of the best Models)
#####################################################
# Get predictions
preds_c_xgb = model_b_trained["XGBoost"].predict(X_test)
preds_c_gb = model_b_trained["Gradient Boosting"].predict(X_test)
preds_c_rf = model_b_trained["Random Forest"].predict(X_test)

# Find best weights on validation set
best_rmsle = float('inf')
best_weights = None

for w_xgb in np.arange(0, 1.01, 0.01):
    for w_gb in np.arange(0, 1.01 - w_xgb, 0.01):
        w_rf = 1 - w_xgb - w_gb
        if w_rf < 0: continue
        weighted_pred = w_xgb * preds_c_xgb + w_gb * preds_c_gb + w_rf * preds_c_rf
        rmsle = np.sqrt(mean_squared_error(y_test, weighted_pred))
        if rmsle < best_rmsle:
            best_rmsle = rmsle
            best_weights = (w_xgb, w_gb, w_rf)

# Apply best weights
w_xgb, w_gb, w_rf = best_weights
model_c_preds = w_xgb * preds_c_xgb + w_gb * preds_c_gb + w_rf * preds_c_rf

print("-------------- Model C: -----------------")
print(f"\n BEST WEIGHTS: XGB={w_xgb:.2f}, GB={w_gb:.2f}, RF={w_rf:.2f}")
print(f"Best validation RMSLE: {best_rmsle:.4f}")
print(f"Model C - RMSE: {np.sqrt(mean_squared_error(to_price(y_test), to_price(model_c_preds))):.2f}")

########################################################
# %% Model D (100% Train , 0% Test)
########################################################

# Convert df to dictionary so we can extract bestparams easier
model_params = dict(zip(model_b_df["model"], model_b_df["best_params"]))

best_params_xgb = model_params["XGBoost"]
best_params_rf = model_params["Random Forest"]
best_params_gb = model_params["Gradient Boosting"]

# "XGBoost"
model_d_xgb = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    **best_params_xgb
)
# "Gradient_Boosting"
model_d_gb = GradientBoostingRegressor(
    random_state=RANDOM_STATE,
    **best_params_gb
)
# "Random Forest"
model_d_rf = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    **best_params_rf
)

model_d_xgb.fit(X, y)
model_d_gb.fit(X, y)
model_d_rf.fit(X, y)
# The model d is prepared to use already with this

######################################################
# %% Submissions
######################################################


test_data = pd.read_csv(test_path)

# We have to redo the same data corrections that we did on our train data.
test_X = test_data.drop(columns="Id")

for col in none_cols:
    test_X[col] = test_X[col].fillna("None")

test_X["LotFrontage"] = test_X["LotFrontage"].fillna(dataset_df["LotFrontage"].median())
test_X["GarageYrBlt"] = test_X["GarageYrBlt"].fillna(0)
test_X["MasVnrArea"] = test_X["MasVnrArea"].fillna(0)
test_X["Electrical"] = test_X["Electrical"].fillna(dataset_df["Electrical"].mode()[0])
test_X["MSSubClass"] = test_X["MSSubClass"].astype(str)

test_X_preprocessed = preprocessor.transform(test_X)
test_X_scaled = scaler.transform(test_X_preprocessed)

# Predict with Model D
preds_d_gb = model_d_gb.predict(test_X_scaled)
preds_d_xgb = model_d_xgb.predict(test_X_scaled)
preds_d_rf = model_d_rf.predict(test_X_scaled)

# Use the best weights found on model C
model_d_pred = (0.6 * preds_d_xgb) + (0.3 * preds_d_gb) + (0.1 * preds_d_rf)
model_d_pred = to_price(model_d_pred)

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": model_d_pred
})
submission.to_csv(project_root / "submissions" / "submission_model_d2_normal_dist.csv", index=False)

########################################################
# %% Graphs
########################################################

# Model A vs Model B: tuned models should improve or stay close to defaults
model_a_plot = model_a_df[["model", "RMSLE"]].rename(columns={"RMSLE": "RMSLE"})
model_a_plot["version"] = "Model A default"

model_b_plot = model_b_df[["model", "Test_RMSLE"]].rename(columns={"Test_RMSLE": "RMSLE"})
model_b_plot["version"] = "Model B tuned"

models_comparison_plot = pd.concat([model_a_plot, model_b_plot], ignore_index=True)

plt.figure(figsize=(10, 5))
sns.barplot(data=models_comparison_plot, x="RMSLE", y="model", hue="version")
plt.title("Default vs Tuned Models - RMSLE")
plt.xlabel("RMSLE lower is better")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# Random Forest: validation error by number of trees
rf_tree_results = []
for n_trees in [50, 100, 200, 300]:
    rf = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    rf_tree_results.append({
        "n_estimators": n_trees,
        "RMSLE": np.sqrt(mean_squared_error(y_test, preds))
    })

rf_tree_df = pd.DataFrame(rf_tree_results)

plt.figure(figsize=(7, 4))
sns.lineplot(data=rf_tree_df, x="n_estimators", y="RMSLE", marker="o")
plt.title("Random Forest - Trees vs Validation RMSLE")
plt.xlabel("Number of trees")
plt.ylabel("RMSLE")
plt.tight_layout()
plt.show()

# Gradient Boosting: validation error as trees are added
boosting_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE,
)
boosting_model.fit(X_train, y_train)

boosting_results = []
for i, preds in enumerate(boosting_model.staged_predict(X_test), start=1):
    if i % 10 == 0:
        boosting_results.append({
            "n_estimators": i,
            "RMSLE": np.sqrt(mean_squared_error(y_test, preds))
        })

boosting_df = pd.DataFrame(boosting_results)

plt.figure(figsize=(7, 4))
sns.lineplot(data=boosting_df, x="n_estimators", y="RMSLE")
plt.title("Gradient Boosting - Trees vs Validation RMSLE")
plt.xlabel("Number of boosting trees")
plt.ylabel("RMSLE")
plt.tight_layout()
plt.show()

# Use XGBoost if it is available; otherwise use the best available Model A model.
if "XGBoost" in model_b_trained:
    best_model_name = "XGBoost"
    best_model = model_b_trained[best_model_name]
else:
    best_model_name = model_a_df.iloc[0]["model"]
    best_model = model_a_trained[best_model_name]

best_preds = to_price(best_model.predict(X_test))
y_test_price = to_price(y_test)
residuals = y_test_price - best_preds
abs_errors = np.abs(residuals).to_numpy()

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test_price, y=best_preds, alpha=0.7)
plt.plot([y_test_price.min(), y_test_price.max()], [y_test_price.min(), y_test_price.max()], color="red", linestyle="--")
plt.title(f"{best_model_name} - Actual vs Predicted")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.scatterplot(x=best_preds, y=residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.title(f"{best_model_name} - Residuals vs Predicted")
plt.xlabel("Predicted SalePrice")
plt.ylabel("Residual: actual - predicted")
plt.tight_layout()
plt.show()

# Largest individual errors: useful to inspect difficult houses
largest_error_positions = np.argsort(abs_errors)[-10:]

plt.figure(figsize=(9, 4))
sns.barplot(x=abs_errors[largest_error_positions], y=[f"Case {i}" for i in largest_error_positions])
plt.title(f"{best_model_name} - Largest Validation Errors")
plt.xlabel("Absolute error")
plt.ylabel("Validation case")
plt.tight_layout()
plt.show()

# SalePrice before and after log transformation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(dataset_df["SalePrice"], kde=True, ax=axes[0])
axes[0].set_title("Original SalePrice")
axes[0].set_xlabel("SalePrice")

sns.histplot(np.log1p(dataset_df["SalePrice"]), kde=True, ax=axes[1])
axes[1].set_title("Log transformed SalePrice")
axes[1].set_xlabel("log1p(SalePrice)")

plt.tight_layout()
plt.show()

# Other intresting gaphs:
  # Actual vs Predicted
  # Residuals distribution
  # Residuals vs Predicted
  # Top largest errors
  # Feature importance
  # Permutation importance
  # Model leaderboard
  # Random Forest n_estimators curve
  # Random Forest n_estimators curve
