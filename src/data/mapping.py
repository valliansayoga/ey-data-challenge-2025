import pandas as pd
import numpy as np
import rioxarray as rxr
from pyproj import Proj, Transformer
from tqdm import tqdm
from pathlib import Path
import warnings
from typing import Optional, Tuple

warnings.filterwarnings("ignore")
tqdm.pandas()


def create_csv(tiff_path: Path, output_path: str, input_df: pd.DataFrame) -> None:
    """
    Create a CSV file with satellite feature data mapped to the input dataframe.

    Parameters
    ----------
    tiff_path : Path
        Path to the input TIFF file.
    output_path : str
        Path to save the output CSV.
    input_df : DataFrame
        Input DataFrame containing 'Latitude' and 'Longitude' columns.

    Returns
    -------
    None
    """
    input_df = input_df.copy()
    path = Path(output_path)
    path.resolve().parent.mkdir(parents=True, exist_ok=True)
    mapping = map_satellite_data(tiff_path, input_df)
    mapping.to_csv(output_path, index=False)


def create_dataset(
    df: pd.DataFrame,
    tiff_files: list[Path],
    typ: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate feature dataset by mapping each TIFF file to the input DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with coordinates.
    tiff_files : list of Path
        List of TIFF file paths.
    typ : str, optional
        Output directory/type prefix.
    output_path : str, optional
        Path to save the final combined CSV.

    Returns
    -------
    DataFrame
        Combined DataFrame with original data and mapped features.
    """
    for file in tqdm(tiff_files, desc="Creating CSV features..."):
        create_csv(file, f"{typ}/{typ}_{file.stem}.csv", df)

    features = list(Path(typ).resolve().rglob(f"*{typ}_*.csv"))
    features = [pd.read_csv(f) for f in features if "Final" not in f.name]
    df_features = pd.concat([df, *features], axis=1)

    if output_path:
        df_features.to_csv(output_path, index=False)

    return df_features


def map_point(
    row: pd.Series, data: rxr.rioxarray.raster_array.RasterArray, radius_pixels: int
) -> float:
    """
    Extract the median satellite data value within a radius around a geographic point.

    Parameters
    ----------
    row : Series
        Row with 'Latitude' and 'Longitude'.
    data : RasterArray
        Raster data from the TIFF.
    radius_pixels : int
        Radius in pixels for the extraction window.

    Returns
    -------
    float
        Median value within the window, ignoring NaNs.
    """
    latitudes = row["Latitude"]
    longitudes = row["Longitude"]

    nearest_point = data.sel(x=longitudes, y=latitudes, band=1, method="nearest")
    x_index = np.where(data.x == nearest_point.x)[0][0]
    y_index = np.where(data.y == nearest_point.y)[0][0]

    x_slice = slice(
        max(0, x_index - radius_pixels), min(data.x.size, x_index + radius_pixels + 1)
    )
    y_slice = slice(
        max(0, y_index - radius_pixels), min(data.y.size, y_index + radius_pixels + 1)
    )

    window_data = data.isel(x=x_slice, y=y_slice).values
    median_value = np.nanmedian(window_data)
    return median_value


def map_satellite_data(
    tiff_path: Path,
    input_df: pd.DataFrame,
    radius_meters: float = 50,
    meter_per_pixel: float = 10,
) -> pd.DataFrame:
    """
    Map satellite TIFF data to points in a dataframe within a given radius.

    Parameters
    ----------
    tiff_path : Path
        Path to the TIFF satellite data file.
    input_df : DataFrame
        DataFrame with 'Latitude' and 'Longitude'.
    radius_meters : float, optional
        Radius in meters for window extraction (default is 50).
    meter_per_pixel : float, optional
        Conversion factor from meters to pixels (default is 10).

    Returns
    -------
    DataFrame
        DataFrame with extracted satellite data per point.
    """
    data = rxr.open_rasterio(tiff_path)
    tiff_crs = data.rio.crs
    radius_pixels = int(radius_meters / meter_per_pixel)

    proj_wgs84 = Proj("epsg:4326")
    proj_tiff = Proj(tiff_crs)
    Transformer.from_proj(proj_wgs84, proj_tiff)  # CRS transformer for future use

    values = input_df.apply(map_point, axis=1, data=data, radius_pixels=radius_pixels)
    df = pd.DataFrame(values, columns=[tiff_path.stem])
    return df


def prepare_train_sub(tiff_files: list[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training and submission datasets with satellite features.

    Parameters
    ----------
    tiff_files : list of Path
        List of satellite TIFF file paths.

    Returns
    -------
    tuple of DataFrame
        Tuple containing training and submission feature DataFrames.
    """
    train_path = "/content/Training_data_uhi_index.csv"
    sub_path = "/content/Submission_template.csv"

    train_df = pd.read_csv(train_path)
    sub_df = pd.read_csv(sub_path)

    typ = "Train"
    out_path = f"{typ}/{typ}_Final.csv"
    Path(out_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    print("Creating training data...")
    train_out = create_dataset(
        train_df, tiff_files=tiff_files, typ=typ, output_path=out_path
    )

    assert train_df.shape[0] == pd.read_csv(out_path).shape[0], (
        "Column not same between input and output after concatting the train_df!"
    )

    typ = "Submission"
    out_path = f"{typ}/{typ}_Final.csv"
    Path(out_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    print("Creating submission data...")
    sub_out = create_dataset(
        sub_df, tiff_files=tiff_files, typ=typ, output_path=out_path
    )

    assert sub_df.shape[0] == pd.read_csv(out_path).shape[0], (
        "Column not same between input and output after concatting the sub_df!"
    )

    return train_out, sub_out
