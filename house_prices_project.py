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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
except ImportError:
    xgb = None


RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


def find_project_root() -> Path:
    project_root = Path.cwd()

    while not (project_root / "data" / "train.csv").exists():
        if project_root.parent == project_root:
            raise FileNotFoundError("Could not find data/train.csv")
        project_root = project_root.parent

    return project_root


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = project_root / "data"
    train_cleaned_path = data_dir / "train_cleaned.csv"
    train_path = train_cleaned_path if train_cleaned_path.exists() else data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_submission_path = data_dir / "sample_submission.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_submission_path)

    print(f"Training file: {train_path}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df, sample_submission


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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
        "MasVnrType",
    ]

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    return df


def show_basic_visualizations(dataset_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.histplot(dataset_df["SalePrice"], kde=True)
    plt.title("Distribution of SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(8, 3))
    sns.boxplot(x=dataset_df["SalePrice"])
    plt.title("Boxplot of SalePrice")
    plt.show()

    df_num = dataset_df.select_dtypes(include=["float64", "int64"])
    df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    plt.show()

    corr = dataset_df.corr(numeric_only=True)
    price_corr = corr["SalePrice"]

    top_corr_table = pd.DataFrame(
        {
            "feature": price_corr.index,
            "correlation_with_saleprice": price_corr.values,
            "absolute_correlation": price_corr.abs().values,
        }
    )

    top_corr_table = top_corr_table.sort_values(
        by="absolute_correlation",
        ascending=False,
    )
    top_corr_table = top_corr_table[top_corr_table["feature"] != "SalePrice"]
    print(top_corr_table.head(15))

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=dataset_df, x="GrLivArea", y="SalePrice")
    plt.title("GrLivArea vs SalePrice")
    plt.show()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

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
    ordinal_features = [col for col in ordinal_features if col in categorical_features]

    onehot_features = [
        col for col in categorical_features
        if col not in ordinal_features
    ]

    quality_order = ["Po", "Fa", "TA", "Gd", "Ex"]
    quality_order_with_none = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

    ordinal_orders = {
        "Street": ["Grvl", "Pave"],
        "Alley": ["None", "Grvl", "Pave"],
        "LotShape": ["IR3", "IR2", "IR1", "Reg"],
        "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
        "LandSlope": ["Sev", "Mod", "Gtl"],
        "ExterQual": quality_order,
        "ExterCond": quality_order,
        "BsmtQual": quality_order_with_none,
        "BsmtCond": quality_order_with_none,
        "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
        "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "HeatingQC": quality_order,
        "CentralAir": ["N", "Y"],
        "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
        "KitchenQual": quality_order,
        "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
        "FireplaceQu": quality_order_with_none,
        "GarageFinish": ["None", "Unf", "RFn", "Fin"],
        "GarageQual": quality_order_with_none,
        "GarageCond": quality_order_with_none,
        "PavedDrive": ["N", "P", "Y"],
        "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
        "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    }
    ordinal_categories = [ordinal_orders[col] for col in ordinal_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    onehot_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Ordinal categorical features: {len(ordinal_features)}")
    print(f"One-hot categorical features: {len(onehot_features)}")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("cat", onehot_transformer, onehot_features),
        ]
    )


def get_candidate_models() -> dict:
    models = {
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
    }

    if xgb is not None:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return models


def evaluate_models(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    results = []
    trained_models = {}

    for name, model in get_candidate_models().items():
        candidate = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        candidate.fit(X_train, y_train)
        preds = np.maximum(candidate.predict(X_valid), 0)

        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_valid), np.log1p(preds)))
        mae = mean_absolute_error(y_valid, preds)
        r2 = r2_score(y_valid, preds)

        results.append(
            {
                "model": name,
                "RMSE": rmse,
                "RMSLE": rmsle,
                "MAE": mae,
                "R2": r2,
            }
        )
        trained_models[name] = candidate

    results_df = pd.DataFrame(results).sort_values("RMSLE")
    return results_df, trained_models


def main() -> None:
    project_root = find_project_root()
    dataset_df, test_data, submission = load_data(project_root)

    dataset_df = clean_dataset(dataset_df)
    test_data = clean_dataset(test_data)

    show_plots = False
    if show_plots:
        show_basic_visualizations(dataset_df)

    dataset_df["MSSubClass"] = dataset_df["MSSubClass"].astype(str)
    test_data["MSSubClass"] = test_data["MSSubClass"].astype(str)

    X = dataset_df.drop(columns=["Id", "SalePrice"])
    y = dataset_df["SalePrice"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_train)
    results_df, trained_models = evaluate_models(
        preprocessor,
        X_train,
        X_valid,
        y_train,
        y_valid,
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model by RMSLE: {best_model_name}")

    test_ids = test_data.pop("Id")
    test_predictions = np.maximum(best_model.predict(test_data), 0)

    submission["Id"] = test_ids
    submission["SalePrice"] = test_predictions

    output_path = project_root / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to: {output_path}")


if __name__ == "__main__":
    main()
