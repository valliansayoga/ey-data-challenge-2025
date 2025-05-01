import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None


to_drop = ["Latitude", "Longitude", "datetime"]
target = "UHI Index"
solar_stat = "Max"


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[float, float]:
    """
    Evaluate a machine learning model's performance.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Model to evaluate.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Test target values.

    Returns
    -------
    tuple[float, float]
        R2 scores for in-sample and out-of-sample predictions.
    """
    model.fit(X_train, y_train)
    insample = r2_score(y_train, model.predict(X_train))
    outsample = r2_score(y_test, model.predict(X_test))
    return insample, outsample


def load_data(typ: str = "Train", osm_to_keep: list[str] = None) -> pd.DataFrame:
    """
    Load and preprocess data from multiple CSV files.

    Parameters
    ----------
    typ : str, optional
        Type of data to load ("Train" or "Submission"), by default "Train"
    osm_to_keep : list[str], optional
        List of OSM features to keep, by default None

    Returns
    -------
    pd.DataFrame
        Combined and preprocessed dataframe.

    Notes
    -----
    The function:
    1. Loads data from multiple CSV files in Google Drive
    2. Combines them horizontally
    3. Drops specified columns
    4. Adds engineered features
    5. Handles missing and infinite values
    """
    global to_drop

    df = (
        pd.concat(
            [
                pd.read_csv(f"/content/drive/MyDrive/EY 2025/{typ}_Final.csv"),
                pd.read_csv(
                    f"/content/drive/MyDrive/EY 2025/{typ}_Building_Features_Revamped.csv"
                ).fillna(0),
                pd.read_csv(
                    f"/content/drive/MyDrive/EY 2025/{typ}_SolarData{solar_stat}SinceTrainTime.csv"
                ),
                pd.read_csv(
                    f"/content/drive/MyDrive/EY 2025/{typ}_OSM.csv", usecols=osm_to_keep
                ),
            ],
            axis=1,
        )
        .drop(to_drop, axis=1, errors="ignore")
        .fillna(0)
    )
    df.drop(target, axis=1).columns
    df = df.pipe(add_features).replace(np.inf, np.nan).fillna(0)
    return df


def prepare_data(df: pd.DataFrame, typ: str, train_size: float) -> tuple:
    """
    Prepare data for model training and evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target.
    typ : str
        Type of data ("Train").
    train_size : float
        Proportion of data to use for training (0-1).

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test split
    """
    X_train, X_test, y_train, y_test = create_train(df, train_size=train_size)
    return X_train, X_test, y_train, y_test


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional features.

    Notes
    -----
    Adds various spectral indices and combinations including:
    - EVI, NDBI, NDWI, NDVI combinations and ratios
    - Building density features
    - Various vegetation and water indices
    - Spectral combinations and transformations
    """
    stats = ["median", "mean", "min", "max", "var", "std"]
    epsilon = 1e-7

    count_cols = df.columns[df.columns.str.contains("nearby_building_count")]
    for col in count_cols:
        divider = int(col.split("_")[0].replace("m", "")[:-1])
        df[f"{col}_density_per_10m"] = df[col] / divider

    for stat in stats:
        df[f"{stat}_evi_x_lwir"] = df[f"evi_{stat}"] * df[f"lwir_{stat}"]
        df[f"{stat}_ndbi_x_lwir"] = df[f"ndbi_{stat}"] * df[f"lwir_{stat}"]
        df[f"{stat}_ndbi_/_ndwi"] = df[f"ndbi_{stat}"] / df[f"ndwi_{stat}"].add(epsilon)
        df[f"{stat}_ndbi_/_ndvi"] = df[f"ndbi_{stat}"] / df[f"ndvi_{stat}"].add(epsilon)
        df[f"{stat}_ndbi_/_evi"] = df[f"ndbi_{stat}"] / df[f"evi_{stat}"].add(epsilon)
        df[f"{stat}_wvp_/_lwir"] = df[f"wvp_{stat}"] / df[f"lwir_{stat}"].add(epsilon)
        df[f"{stat}_infra_red_combo"] = (
            df[f"nir08_{stat}"] * df[f"swir16_{stat}"] * df[f"swir22_{stat}"]
        )
        df[f"{stat}_relative_ndvi"] = df[f"ndvi_{stat}"] / (
            df[f"ndvi_{stat}"].max() + epsilon
        )
        df[f"{stat}_relative_ndwi"] = df[f"ndwi_{stat}"] / (
            df[f"ndwi_{stat}"].max() + epsilon
        )
        df[f"{stat}_ndvi_ndwi_ratio"] = df[f"ndvi_{stat}"] / df[f"ndwi_{stat}"].add(
            epsilon
        )
        df[f"{stat}_ndvi_evi_ratio"] = df[f"ndvi_{stat}"] / df[f"evi_{stat}"].add(
            epsilon
        )
        df[f"{stat}_ndwi_evi_ratio"] = df[f"ndwi_{stat}"] / df[f"evi_{stat}"].add(
            epsilon
        )
        df[f"{stat}_ndvi_ndwi_diff"] = df[f"ndvi_{stat}"] - df[f"ndwi_{stat}"]
        df[f"{stat}_ndvi_evi_diff"] = df[f"ndvi_{stat}"] - df[f"evi_{stat}"]
        df[f"{stat}_ndwi_evi_diff"] = df[f"ndwi_{stat}"] - df[f"evi_{stat}"]
        df[f"{stat}_combined_spectral_index"] = (
            df[f"ndvi_{stat}"] + df[f"ndwi_{stat}"] + df[f"evi_{stat}"]
        ) / 3

        df[f"{stat}_bsi"] = (
            df[f"swir16_{stat}"] + df[f"swir22_{stat}"] - 2 * df[f"nir08_{stat}"]
        ) / (df[f"swir16_{stat}"] + df[f"swir22_{stat}"] + 2 * df[f"nir08_{stat}"])
        df[f"{stat}_savi"] = ((df[f"nir08_{stat}"] - df[f"red_{stat}"]) * (1 + 0.5)) / (
            df[f"nir08_{stat}"] + df[f"red_{stat}"] + 0.5
        )
        df[f"{stat}_sr"] = df[f"nir08_{stat}"] / df[f"red_{stat}"]
        df[f"{stat}_dsi"] = (df[f"swir22_{stat}"] - df[f"nir08_{stat}"]) / (
            df[f"swir22_{stat}"] + df[f"nir08_{stat}"]
        )
        df[f"{stat}_wvi"] = df[f"wvp_{stat}"] / (
            df[f"swir22_{stat}"] + df[f"swir16_{stat}"] + df[f"nir08_{stat}"]
        )

    df["mean_vci"] = (df["ndvi_mean"] - df["ndvi_min"]) / (
        df["ndvi_max"] - df["ndvi_min"]
    )
    df["median_vci"] = (df["ndvi_median"] - df["ndvi_min"]) / (
        df["ndvi_max"] - df["ndvi_min"]
    )

    return df


def create_train(
    df_features: pd.DataFrame, target: str = "UHI Index", train_size: float = 0.8
) -> tuple:
    """
    Split data into training and test sets.

    Parameters
    ----------
    df_features : pd.DataFrame
        Input dataframe with features and target.
    target : str, optional
        Name of target column, by default "UHI Index"
    train_size : float, optional
        Proportion of data for training, by default 0.8

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    X = df_features.drop(target, axis=1)
    y = df_features[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    return X_train, X_test, y_train, y_test


def round1(
    models: list[dict],
    train_size: float,
    score_func: callable = f_regression,
    percentile: int = 30,
) -> pd.DataFrame:
    """
    Perform first round of model evaluation with feature selection.

    Parameters
    ----------
    models : list[dict]
        List of model dictionaries containing 'model' key with estimator.
    train_size : float
        Proportion of data to use for training.
    score_func : callable, optional
        Scoring function for feature selection, by default f_regression
    percentile : int, optional
        Top percentile of features to keep, by default 30

    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance information.

    Notes
    -----
    This round:
    1. Loads and prepares data
    2. Performs univariate feature selection
    3. Evaluates all models
    4. Returns feature importance from best model
    """
    separator = "-" * 66
    spaces = " " * 24
    equals = "=" * 32

    print(spaces, "Starting round 1", spaces)
    print(separator)
    df = load_data(typ="Train")
    X_train, X_test, y_train, y_test = prepare_data(
        df, typ="Train", train_size=train_size
    )
    select = SelectPercentile(score_func, percentile=percentile)
    select.fit(X_train, y_train)

    X_train = X_train[select.get_feature_names_out()]
    X_test = X_test[X_train.columns]

    for model in tqdm(models):
        insample, outsample = evaluate_model(
            model["model"], X_train, X_test, y_train, y_test
        )
        model["insample"] = insample
        model["outsample"] = outsample

    results = (
        pd.DataFrame(models)
        .sort_values("outsample", ascending=False)
        .reset_index(drop=True)
    )
    print(equals, "Model Scores", equals)
    print(results)
    print(separator)
    best_model = results.iloc[0]
    print(equals, "Best Model", equals)
    print(best_model.model)
    print(separator)

    importance = (
        pd.DataFrame(
            {
                "Features": X_train.columns,
                "Importance": best_model.model.feature_importances_,
            },
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    importance["cumulative_importance"] = (
        importance.Importance.cumsum() / importance.Importance.sum()
    ).round(2)
    print(equals, "Feature Importance", equals)
    print(importance)
    print(separator)
    return importance


def round2(
    models: list[dict],
    pareto_threshold: float,
    importance: pd.DataFrame,
    train_size: float,
) -> tuple:
    """
    Perform second round of model evaluation with Pareto-optimal features.

    Parameters
    ----------
    models : list[dict]
        List of model dictionaries containing 'model' key with estimator.
    pareto_threshold : float
        Cumulative importance threshold for feature selection (0-1).
    importance : pd.DataFrame
        Feature importance dataframe from round1.
    train_size : float
        Proportion of data to use for training.

    Returns
    -------
    tuple
        Best model information and training features.

    Notes
    -----
    This round:
    1. Selects features accounting for pareto_threshold of importance
    2. Evaluates models on reduced feature set
    3. Returns best performing model
    """
    separator = "-" * 66
    spaces = " " * 24
    equals = "=" * 32

    if importance is not None:
        pareto = importance[importance.cumulative_importance <= pareto_threshold]
        print(equals, "Pareto Features", equals)
        print(pareto)

    print(separator)

    print(spaces, "Starting round 2", spaces)
    print(separator)

    df = load_data(typ="Train")
    use_cols = pareto.Features

    X_train, X_test, y_train, y_test = prepare_data(
        df, typ="Train", train_size=train_size
    )
    X_train = X_train[use_cols]
    X_test = X_test[X_train.columns]

    for model in tqdm(models):
        insample, outsample = evaluate_model(
            model["model"], X_train, X_test, y_train, y_test
        )
        model["insample"] = insample
        model["outsample"] = outsample

    results = (
        pd.DataFrame(models)
        .sort_values("outsample", ascending=False)
        .reset_index(drop=True)
    )
    print(equals, "Model Scores", equals)
    print(results.head(1))
    print(separator)
    best_model = results.iloc[0]
    print(equals, "Best Model", equals)
    print(best_model.model)
    print(separator)

    importance = (
        pd.DataFrame(
            {
                "Features": X_train.columns,
                "Importance": best_model.model.feature_importances_,
            },
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance["cumulative_importance"] = (
        importance.Importance.cumsum() / importance.Importance.sum()
    ).round(2)
    print(equals, "Feature Importance", equals)
    print(importance)
    print(separator)
    return best_model, X_train
