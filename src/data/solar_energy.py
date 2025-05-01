import pandas as pd
import pvlib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Any, Dict, List

stats: List[str] = ["mean", "median", "min", "max"]


def get_solar_data(row: pd.Series) -> pd.DataFrame:
    """
    Compute solar position, airmass, and clearsky data for a single row of location and time.

    Parameters
    ----------
    row : pd.Series
        A row containing 'datetime_cnvrt', 'Latitude', and 'Longitude'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing solar position, airmass, and clearsky irradiance data.
    """
    times = row.datetime_cnvrt
    latitude = row.Latitude
    longitude = row.Longitude

    solar_position = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    loc = pvlib.location.Location(latitude, longitude, tz=times.tz)

    airmass = loc.get_airmass(
        times=solar_position.index,
        solar_position=solar_position[["apparent_zenith", "zenith"]],
    )

    clearsky = loc.get_clearsky(
        solar_position.index,
        solar_position=solar_position[
            ["apparent_zenith", "zenith", "apparent_elevation"]
        ],
    )

    df_solar = pd.concat(
        [
            solar_position.reset_index(drop=True),
            airmass.reset_index(drop=True),
            clearsky.reset_index(drop=True),
        ],
        axis=1,
    )
    return df_solar


def evaluate_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate the usefulness of solar variables for predicting the UHI Index using an ExtraTrees model.

    For each statistical summary (mean, median, min, max) of datetime:
    - Converts the datetime to UTC
    - Computes solar features
    - Trains and evaluates a model
    - Generates predictions for the submission dataset to check for its predicting power
    - Saves solar features and predictions to CSV files

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'datetime', 'Latitude', 'Longitude', and 'UHI Index'.

    Returns
    -------
    pd.DataFrame
        Summary of R² scores for each datetime statistic.
    """
    result_list: List[Dict[str, Any]] = []

    for stat in tqdm(stats):
        local_container: Dict[str, Any] = {}
        df_ = df.copy()
        result_dict: Dict[str, Any] = {}

        # Apply datetime aggregation (mean, median, min, max)
        exec(f"df_.datetime = df_.datetime.{stat}()")
        df_["datetime_cnvrt"] = df_.datetime.dt.tz_localize(
            "America/New_York", ambiguous="infer"
        )
        df_["datetime_cnvrt"] = df_["datetime_cnvrt"].dt.tz_convert("UTC")

        solar_data = pd.concat(
            df_.apply(get_solar_data, axis=1).to_numpy().tolist()
        ).reset_index(drop=True)
        filename = f"/content/drive/MyDrive/EY 2025/Train_SolarData{stat.title()}SinceTrainTime.csv"
        solar_data.to_csv(filename, index=False)

        X = solar_data
        y = df_["UHI Index"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        et = ExtraTreesRegressor(n_estimators=250, random_state=0, n_jobs=-1)
        et.fit(X_train, y_train)
        result_dict["stat"] = stat
        result_dict["r2_score"] = r2_score(y_test, et.predict(X_test))

        # Prepare submission data using the same datetime aggregation
        exec(f"local_container['stat_time'] = df_.datetime_cnvrt.{stat}()")
        sub_df = pd.read_csv("/content/Dataset/Submission_template.csv")
        sub_df["datetime_cnvrt"] = local_container["stat_time"]

        solar_data_sub = pd.concat(
            sub_df.apply(get_solar_data, axis=1).to_numpy().tolist()
        ).reset_index(drop=True)
        filename = f"/content/drive/MyDrive/EY 2025/Submission_SolarData{stat.title()}SinceTrainTime.csv"
        solar_data_sub.to_csv(filename, index=False)

        X_sub = solar_data_sub
        sub_df["UHI Index"] = et.predict(X_sub)
        filename = f"/content/SolarData{stat.title()}SinceTrainTime.csv"
        sub_df.iloc[:, :-1].to_csv(filename, index=False)

        result_list.append(result_dict)

    return pd.DataFrame(result_list)
