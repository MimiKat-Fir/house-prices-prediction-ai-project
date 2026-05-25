# %% [markdown]
# # House Prices Prediction

# %% [markdown]
# ## Import libraries
# 

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


# %% [markdown]
# ## Load the dataset
# 

# %%
PROJECT_ROOT = Path.cwd()
PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_OLD_PATH = DATA_DIR / "train.csv"
TRAIN_PATH = DATA_DIR / "train_cleaned.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


dataset_df = pd.read_csv(TRAIN_PATH)
print(f"Full train dataset shape is {dataset_df.shape}")

# %% [markdown]
# ### Cleaning the dataset (from Data_Analyze)
# 

# %% [markdown]
# We will drop the `Id` column as it is not necessary for model training.

# %%
dataset_df = dataset_df.drop(columns="Id")
dataset_df.head(3)


# %%
dataset_df.info()

# %% [markdown]
# ### Visualization of the numerical data
# 

# %%
df_num = dataset_df.select_dtypes(include = ['float64', 'int64'])
df_num.head()

# %% [markdown]
# Now let us plot the distribution for all the numerical features.

# %%
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

# %% [markdown]
# ### Categorical vs Numerical



# %% [markdown]
# ### Prepare the (train / test) data

# %%
X = dataset_df.drop(columns="SalePrice")
y = dataset_df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=RANDOM_STATE,
)

print(f"Training set: {len(X_train)} samples, {X_train.shape[1]} features")
print(f"Validation set: {len(X_test)} samples")


# %%
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Numeric features: {len(categorical_features)}")

# %% Data Clean

# Now we have to separate the ordinal from pure categorical
# These are categorical but have a logical order
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

  # Everything categorical but NOT ordinal -> one-hot
onehot_features = [
      col for col in categorical_features
      if col not in ordinal_features
]

# %% Preparing data for transformation

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

# %% Data transforming

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

preprocessor = ColumnTransformer(
      transformers=[
          ("num", numeric_transformer, numeric_features),
          ("ord", ordinal_transformer, ordinal_features),
          ("cat", onehot_transformer, onehot_features),
      ]
)

print(f"Ordinal categorical features: {len(ordinal_features)}")
print(f"One-hot categorical features: {len(onehot_features)}")


# %% Model F
# Simple regression tournament.

model_f_candidates = {
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
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "XGBoost":xgb.XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    ),
}

# %%
model_f_results = []
model_f_trained = {}

for name, model in model_f_candidates.items():
    candidate = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    candidate.fit(X_train, y_train)
    preds = candidate.predict(X_test)
    preds = np.maximum(preds, 0)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(preds)))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    model_f_results.append(
        {
            "model": name,
            "RMSE": rmse,
            "RMSLE": rmsle,
            "MAE": mae,
            "R2": r2,
        }
    )
    model_f_trained[name] = candidate

model_f_df = pd.DataFrame(model_f_results).sort_values("RMSLE")
model_f_df

# %% [markdown]
# ## Model S

# %% [markdown]
# ## Inspect models accuracy
# 

# %% [markdown]
# # Submission
# 
# Finally, predict on the competition test data using the local `data/test.csv` file and save a Kaggle-compatible submission file in the project root.
# 

# %%

# test_data = pd.read_csv(TEST_PATH)
# submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# ids = test_data.pop("Id")
# test_predictions = rf_model.predict(test_data)

# submission["Id"] = ids
# submission["SalePrice"] = test_predictions

# output_path = PROJECT_ROOT / "submission.csv"
# submission.to_csv(output_path, index=False)

# print(f"Saved submission to: {output_path}")
# submission.head()


# %%
# submission.describe()



