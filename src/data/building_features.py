import warnings
from typing import Tuple

import geopandas as gpd
import numpy as np
from fiona.drvsupport import supported_drivers
from shapely.geometry import Point
from shapely.ops import unary_union
import pandas as pd

warnings.filterwarnings("ignore")


def get_nearest_N_polygons(point: Point, polygons: gpd.GeoSeries, N: int) -> pd.Series:
    """
    Calculate the distance from a point to the nearest N polygons.

    Parameters
    ----------
    point : Point
        The input geometry point.
    polygons : GeoSeries
        Collection of building polygons.
    N : int
        Number of nearest polygons to consider.

    Returns
    -------
    Series
        Series of distances to the N nearest polygons.
    """
    return polygons.distance(point).sort_values().head(N)


def calculate_stats_distance_to_N_polygons(
    row: gpd.GeoSeries, polygons: gpd.GeoSeries, N: int
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute distance statistics from a point to the nearest N polygons.

    Parameters
    ----------
    row : GeoSeries
        Row containing point geometry.
    polygons : GeoSeries
        Building polygons.
    N : int
        Number of polygons to consider.

    Returns
    -------
    tuple of float
        Mean, median, max, min, standard deviation, and range of distances.
    """
    point = row.geometry
    sorted_distances = get_nearest_N_polygons(point, polygons, N).values
    return (
        np.mean(sorted_distances),
        np.median(sorted_distances),
        np.max(sorted_distances),
        np.min(sorted_distances),
        np.std(sorted_distances),
        np.ptp(sorted_distances),
    )


def calculate_building_area_statistics_in_radius(
    row: gpd.GeoSeries, polygons: gpd.GeoSeries, buffer_radius: float
) -> Tuple[float, ...]:
    """
    Calculate building area and geometry statistics within a radius around a point.

    Parameters
    ----------
    row : GeoSeries
        Row containing point geometry.
    polygons : GeoSeries
        Building polygons.
    buffer_radius : float
        Radius in meters.

    Returns
    -------
    tuple of float
        Density, area density, building count, and various area and circumference stats.
    """
    point = row.geometry
    buffer_zone = point.buffer(buffer_radius)
    nearby_buildings = polygons[polygons.intersects(buffer_zone)]
    count_buildings = len(nearby_buildings)
    density = count_buildings / buffer_zone.area
    building_area = nearby_buildings.geometry.area.sum()
    return (
        density,
        building_area / buffer_zone.area,
        count_buildings,
        np.mean(nearby_buildings.geometry.area),
        np.median(nearby_buildings.geometry.area),
        np.max(nearby_buildings.geometry.area),
        np.min(nearby_buildings.geometry.area),
        np.std(nearby_buildings.geometry.area),
        np.ptp(nearby_buildings.geometry.area),
        np.mean(nearby_buildings.geometry.length),
        np.median(nearby_buildings.geometry.length),
        np.max(nearby_buildings.geometry.length),
        np.min(nearby_buildings.geometry.length),
        np.std(nearby_buildings.geometry.length),
        np.ptp(nearby_buildings.geometry.length),
    )


def calculate_nearest_building_statistics(
    row: gpd.GeoSeries, polygons: gpd.GeoSeries
) -> Tuple[float, float, float, int]:
    """
    Extract nearest building area, intersection area, angle and vertex count.

    Parameters
    ----------
    row : GeoSeries
        Point geometry row.
    polygons : GeoSeries
        Building polygons.

    Returns
    -------
    tuple
        Area, intersection area, angle (degrees), vertex count.
    """
    point = row.geometry
    nearest_index = get_nearest_N_polygons(point, polygons, 1).index
    nearest_polygon = polygons.iloc[nearest_index]
    neighbors = polygons[polygons.intersects(nearest_polygon)]
    intersection_area = nearest_polygon.intersection(unary_union(neighbors)).area
    centroid = nearest_polygon.centroid
    angle = np.arctan2(centroid.y - point.y, centroid.x - point.x).values[0]
    vertices = np.array(
        np.array(nearest_polygon.values[0].geoms[0].exterior.coords.xy[0]).shape[0]
    )
    return (
        nearest_polygon.area.values[0],
        intersection_area.values[0],
        np.degrees(angle),
        vertices,
    )


def calculate_point_relative_position_to_building_boundary(
    row: gpd.GeoSeries, polygons: gpd.GeoSeries
) -> Tuple[float, float, float, int]:
    """
    Determine if point is closer to the edge or center of the building.

    Parameters
    ----------
    row : GeoSeries
        Row containing geometry.
    polygons : GeoSeries
        Building polygons.

    Returns
    -------
    tuple
        Boundary distance, centroid distance, ratio, and distance class.
    """
    point = row.geometry
    nearest_index = get_nearest_N_polygons(point, polygons, 1).index
    nearest_polygon = polygons.iloc[nearest_index]

    boundary_distance = point.distance(nearest_polygon.boundary).values.squeeze()
    centroid_distance = point.distance(nearest_polygon.centroid).values.squeeze()

    if boundary_distance < centroid_distance:
        return (
            boundary_distance,
            centroid_distance,
            centroid_distance / boundary_distance,
            1,
        )
    elif boundary_distance > centroid_distance:
        return (
            boundary_distance,
            centroid_distance,
            centroid_distance / boundary_distance,
            2,
        )
    else:
        return (
            boundary_distance,
            centroid_distance,
            centroid_distance / boundary_distance,
            3,
        )


def calculate_N_nearest_polygon_shape_stats_complexity(
    row: gpd.GeoSeries, polygons: gpd.GeoSeries, N: int
) -> Tuple[float, float, float, float, float, float]:
    """
    Estimate shape complexity of nearest N polygons via vertex count.

    Parameters
    ----------
    row : GeoSeries
        Row with geometry.
    polygons : GeoSeries
        Building polygons.
    N : int
        Number of polygons to include.

    Returns
    -------
    tuple
        Mean, median, max, min, std, and range of vertex counts.
    """
    point = row.geometry
    nearest_index = get_nearest_N_polygons(point, polygons, N).index
    nearest_polygons = polygons.iloc[nearest_index].values

    vertices = np.array(
        [
            np.array(geom.exterior.coords.xy[0]).shape[0]
            for multi in nearest_polygons
            for geom in multi.geoms
        ]
    )
    return (
        vertices.mean(),
        np.median(vertices),
        vertices.max(),
        vertices.min(),
        vertices.std(),
        vertices.max() - vertices.min(),
    )


def get_point(row: pd.Series) -> Point:
    """
    Convert a row of coordinates into a Shapely Point.

    Parameters
    ----------
    row : Series
        Row with 'Latitude' and 'Longitude'.

    Returns
    -------
    Point
        Shapely geometry point.
    """
    return Point(row["Longitude"], row["Latitude"])


def create_building_features(
    df: pd.DataFrame, buffer_radius: float = 100, N: int = 50
) -> gpd.GeoDataFrame:
    """
    Create building-based spatial features for each point in the dataframe.

    Parameters
    ----------
    df : DataFrame
        Input data with 'Latitude' and 'Longitude' columns.
    buffer_radius : float, optional
        Radius in meters to consider for nearby building stats (default is 100).
    N : int, optional
        Number of nearest buildings to include in calculations (default is 50).

    Returns
    -------
    GeoDataFrame
        GeoDataFrame enriched with building features.
    """
    print("Loading footprint...")
    epsg_distance = "32633"
    epsg_read = "4326"
    supported_drivers["LIBKML"] = "rw"
    my_map = gpd.read_file("/content/Dataset/Building_Footprint.kml", driver="LIBKML")
    my_map.to_crs(epsg=epsg_distance, inplace=True)

    print("Creating points...")
    geometric_points = df.apply(get_point, axis=1).to_numpy().tolist()
    geo_locations = gpd.GeoDataFrame(
        df, crs=f"epsg:{epsg_read}", geometry=geometric_points
    )
    geo_locations.to_crs(epsg=epsg_distance, inplace=True)
    polygons = my_map.geometry

    print("Adding 'is_building'...")
    is_building = [
        int(any(my_map.contains(point).values)) for point in geo_locations.geometry
    ]
    geo_locations["is_a_building"] = is_building

    print("Calculating nearest building distance...")
    assert geo_locations.crs == my_map.crs, "CRS must match!"
    geo_locations["nearest_building_distance"] = geo_locations.apply(
        lambda row: my_map.distance(row.geometry).min(), axis=1
    )

    print("Calculating 'stats_distance'...")
    (
        geo_locations["average_distance"],
        geo_locations["median_distance"],
        geo_locations["max_distance"],
        geo_locations["min_distance"],
        geo_locations["std_distance"],
        geo_locations["range_distance"],
    ) = zip(
        *geo_locations.apply(
            calculate_stats_distance_to_N_polygons, axis=1, polygons=polygons, N=N
        )
    )

    print("Calculating 'building stats in a radius'...")
    (
        geo_locations[f"count_{buffer_radius}radius_density"],
        geo_locations[f"area_{buffer_radius}radius_density"],
        geo_locations[f"count_bldng_{buffer_radius}radius"],
        geo_locations[f"average_bldng_{buffer_radius}radius_area"],
        geo_locations[f"median_bldng_{buffer_radius}radius_area"],
        geo_locations[f"max_bldng_{buffer_radius}radius_area"],
        geo_locations[f"min_bldng_{buffer_radius}radius_area"],
        geo_locations[f"std_bldng_{buffer_radius}radius_area"],
        geo_locations[f"range_bldng_{buffer_radius}radius_area"],
        geo_locations[f"average_bldng_{buffer_radius}radius_circumference"],
        geo_locations[f"median_bldng_{buffer_radius}radius_circumference"],
        geo_locations[f"max_bldng_{buffer_radius}radius_circumference"],
        geo_locations[f"min_bldng_{buffer_radius}radius_circumference"],
        geo_locations[f"std_bldng_{buffer_radius}radius_circumference"],
        geo_locations[f"range_bldng_{buffer_radius}radius_circumference"],
    ) = zip(
        *geo_locations.apply(
            calculate_building_area_statistics_in_radius,
            axis=1,
            polygons=polygons,
            buffer_radius=buffer_radius,
        )
    )

    print("Calculating 'nearest building stats'...")
    (
        geo_locations["nearest_bldg_area"],
        geo_locations["nearest_bldg_intersection_area"],
        geo_locations["nearest_bldg_angle"],
        geo_locations["nearest_bldg_vertices"],
    ) = zip(
        *geo_locations.apply(
            calculate_nearest_building_statistics, axis=1, polygons=polygons
        )
    )

    print("Calculating 'boundary relative position'...")
    (
        geo_locations["boundary_distance"],
        geo_locations["centroid_distance"],
        geo_locations["centroid_boundary_ratio"],
        geo_locations["perimeter_class"],
    ) = zip(
        *geo_locations.apply(
            calculate_point_relative_position_to_building_boundary,
            axis=1,
            polygons=polygons,
        )
    )

    print("Calculating 'nearest building complexity'...")
    (
        geo_locations["nearest_bldg_avg_vertices"],
        geo_locations["nearest_bldg_median_vertices"],
        geo_locations["nearest_bldg_max_vertices"],
        geo_locations["nearest_bldg_min_vertices"],
        geo_locations["nearest_bldg_std_vertices"],
        geo_locations["nearest_bldg_range_vertices"],
    ) = zip(
        *geo_locations.apply(
            calculate_N_nearest_polygon_shape_stats_complexity,
            axis=1,
            polygons=polygons,
            N=N,
        )
    )

    print("Done!")
    return geo_locations
