import numpy as np
import geopandas as gpd
import json
from collections import Counter
from typing import List, Optional, Union
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
from shapely.geometry import Point


unmapped_tags: List[str] = list()
no_tags: List[int] = list()
error_point: List[int] = list()


class OSMAPIDataset:
    """
    A class to fetch and manage OSM (OpenStreetMap) data using Overpass API.
    """

    API_URL = "http://overpass-api.de/api/interpreter"

    def __init__(
        self,
        train_latlong: List[tuple[float, float]],
        submission_latlong: List[tuple[float, float]],
    ) -> None:
        """
        Initialize the dataset with training and submission coordinates.

        Parameters
        ----------
        train_latlong : list of tuple
            List of (latitude, longitude) tuples for training data.
        submission_latlong : list of tuple
            List of (latitude, longitude) tuples for submission data.
        """
        self.api_hit_count = 0
        self.train_latlong = train_latlong
        self.submission_latlong = submission_latlong
        self.error_record: List[dict[str, Union[str, int]]] = list()

    def get_osm_data(self, latlong: tuple[float, float], radius: int) -> Optional[dict]:
        """
        Query Overpass API to get OSM data for a single coordinate.

        Parameters
        ----------
        latlong : tuple of float
            Latitude and longitude.
        radius : int
            Search radius in meters.

        Returns
        -------
        dict or None
            JSON response from the API or None on failure.
        """
        self.api_hit_count += 1
        latitude, longitude = latlong
        query = f"""
        [out:json];
        (
            node(around:{radius},{latitude},{longitude});
            way(around:{radius},{latitude},{longitude});
            relation(around:{radius},{latitude},{longitude});
        );
        out body;
        >;
        out skel qt;
        """
        response = requests.get(self.API_URL, params={"data": query})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Unable to fetch data (Status Code: {response.status_code})")
            return None

    def create_osm_json(self, typ: str, radius: int) -> None:
        """
        Query and save OSM data for all coordinates.

        Parameters
        ----------
        typ : {'train', 'submission'}
            Type of dataset.
        radius : int
            Radius to search for OSM elements.
        """
        output_path = Path("/content/drive/MyDrive/EY 2025/OSM")
        if typ == "train":
            latlong = list(enumerate(self.train_latlong))
            output_path = output_path / "Train"
        elif typ == "submission":
            latlong = list(enumerate(self.submission_latlong))
            output_path = output_path / "Submission"
        else:
            raise ValueError("Invalid type. Must be 'train' or 'submission'.")

        output_path.mkdir(parents=True, exist_ok=True)
        check_existing_files = [file.stem for file in output_path.glob("*.json")]

        for i, coord in tqdm(latlong, desc=f"Querying {typ}"):
            if str(i) in check_existing_files:
                continue
            if self.api_hit_count >= 10000:
                print("Reached 10,000 API hits for today.")
                break

            data = self.get_osm_data(coord, radius)
            if data is None:
                self.error_record.append({"type": typ, "index": i})
                continue
            with open(output_path / f"{i}.json", "w") as f:
                json.dump(data["elements"], f, indent=2)

        error_counts = Counter(map(lambda d: d["type"], self.error_record))
        print("Error counts")
        print("-" * 40)
        print(error_counts)

    def __str__(self) -> str:
        error_counts = Counter(map(lambda d: d["type"], self.error_record))
        return (
            f"OSMAPIDataset(train_latlong={len(self.train_latlong)}, "
            f"submission_latlong={len(self.submission_latlong)}, "
            f"train_errors={error_counts.get('train', 0)}, "
            f"submission_errors={error_counts.get('submission', 0)})"
        )

    def __repr__(self) -> str:
        return self.__str__()


def convert_to_point(row: pd.Series) -> Point:
    """
    Convert a DataFrame row with 'lat' and 'lon' to a Shapely Point.

    Parameters
    ----------
    row : Series
        A row with 'lat' and 'lon' columns.

    Returns
    -------
    Point
        Shapely geometry point.
    """
    return Point(row["lon"], row["lat"])


def row_index(row: pd.Series) -> int:
    """
    Extract the index of the row.

    Parameters
    ----------
    row : Series
        DataFrame row.

    Returns
    -------
    int
        Index of the row.
    """
    return row.name


def map_tags(tags: dict) -> Optional[str]:
    """
    Map OSM tags to simplified label types.

    Parameters
    ----------
    tags : dict
        Dictionary of OSM tags.

    Returns
    -------
    str or None
        Mapped tag name.
    """
    for tag in tags:
        if tag == "highway":
            return tag + "_" + tags[tag]
        elif tag == "addr:housenumber":
            return "house"
        elif tag == "natural":
            return tag + "_" + tags[tag]
        elif tag == "amenity":
            return tag + "_" + tags[tag]
        elif tag == "barrier":
            return tag + "_" + tags[tag]
        elif tag == "bridge":
            if tags[tag] == "yes":
                return "bridge"
        elif tag == "railway":
            return tag + "_" + tags[tag]
        elif tag == "footway":
            return "footway"
        elif tag == "tourism":
            return tag + "_" + tags[tag]
        elif tag == "emergency":
            return tag + "_" + tags[tag]
        else:
            unmapped_tags.append(tag)
    return None


def extract_osm_data(row: pd.Series, json_files: dict[int, Path]) -> pd.DataFrame:
    """
    Extract and process OSM data from JSON files.

    Parameters
    ----------
    row : Series
        GeoDataFrame row with geometry and index.
    json_files : dict
        Mapping of row indices to their corresponding JSON file paths.

    Returns
    -------
    DataFrame
        Flattened feature DataFrame with distance statistics for each tag type.
    """
    df_json = pd.read_json(json_files[row.row_index])
    data = df_json[
        (df_json.tags.notnull())
        & (df_json.type != "relation")
        & (df_json.type != "way")
    ].iloc[:, 2:5]

    try:
        data["type"] = data.tags.apply(map_tags)
    except Exception as e:
        print(e)
        no_tags.append(row.row_index)
        return pd.DataFrame({"amenity_bench_count": [np.nan]})

    try:
        data["geometry"] = data.apply(convert_to_point, axis=1)
    except Exception as e:
        print(e)
        error_point.append(row.row_index)
        return pd.DataFrame({"amenity_bench_count": [np.nan]})

    gpd_json = gpd.GeoDataFrame(data, crs="EPSG:4326").to_crs("EPSG:3857")
    gpd_json.drop(columns=["lat", "lon", "tags"], inplace=True)

    geometry_reference = row.geometry
    gpd_json["distance"] = gpd_json.geometry.distance(geometry_reference)

    group = (
        gpd_json.groupby("type")["distance"]
        .agg(["mean", "max", "min", "median", "count", "var", "std"])
        .fillna(-9)
    )

    flat_df = group.stack().to_frame().T
    flat_df.columns = [
        f"{col[0]}_distance_{col[1]}" if col[1] != "count" else f"{col[0]}_{col[1]}"
        for col in flat_df.columns
    ]
    return flat_df
