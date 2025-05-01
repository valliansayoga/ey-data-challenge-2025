from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import xarray as xr
from odc.stac import stac_load
from rich import print
from rich.table import Table


class SatelliteData:
    """
    Class to query, access, and visualize satellite imagery using the Microsoft Planetary Computer.
    """

    def __init__(
        self,
        bounds: Optional[tuple] = None,
        time_window: Optional[str] = None,
        collections: Optional[str] = None,
        query_params: Optional[dict] = None,
        resolution: int = 10,
        degrees_per_pixel: float = 111320.0,
        crs: str = "EPSG:4326",
    ) -> None:
        """
        Initialize a satellite data object by querying the STAC API.

        Parameters
        ----------
        bounds : tuple, optional
            Bounding box coordinates in (minx, miny, maxx, maxy).
        time_window : str, optional
            Time range for image collection (e.g., "2020-01-01/2020-12-31").
        collections : str, optional
            STAC collection name (e.g., "sentinel-2-l2a").
        query_params : dict, optional
            Additional query parameters to filter STAC items.
        resolution : int, optional
            Target spatial resolution in meters.
        degrees_per_pixel : float, optional
            Conversion factor to degrees (default is approx. 111320.0 for EPSG:4326).
        crs : str, optional
            Coordinate reference system to project to.
        """
        self.stac = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
        search = self.stac.search(
            bbox=bounds,
            datetime=time_window,
            collections=collections,
            query=query_params,
        )
        self.crs = crs
        self.bounds = bounds
        self.time_window = time_window
        self.collections = collections
        self.query_params = query_params
        self.items = list(search.get_items())
        self.resolution = resolution
        self.scale = resolution / degrees_per_pixel
        if self.__len__() == 0:
            raise ValueError(
                "Your search returned 0 result! Check again your parameters!"
            )

    def __len__(self) -> int:
        """
        Return the number of items retrieved from the STAC search.

        Returns
        -------
        int
            Number of items.
        """
        return len(self.items)

    def show_assets(self) -> pd.DataFrame:
        """
        Display available asset descriptions from the first STAC item.

        Returns
        -------
        DataFrame
            Table of asset names and their descriptions.
        """
        assets_df = {key: [asset.title] for key, asset in self.items[0].assets.items()}
        asset_metadata = pd.DataFrame(assets_df, index=["description"]).T.reset_index()
        asset_metadata.rename(columns={"index": "asset"}, inplace=True)
        return asset_metadata

    def load_bands(self, bands: list[str]) -> xr.core.dataset.Dataset:
        """
        Load selected bands as an xarray dataset.

        Parameters
        ----------
        bands : list of str
            List of band names to load (e.g., ['B04', 'B08']).

        Returns
        -------
        Dataset
            Loaded band data as xarray dataset.
        """
        return stac_load(
            self.items,
            bands=bands,
            crs=self.crs,
            resolution=self.scale,
            chunks={"x": 2048, "y": 2048},
            dtype="int32",
            patch_url=planetary_computer.sign,
            bbox=self.bounds,
        )

    def __str__(self) -> str:
        """
        Print object summary in rich table format.

        Returns
        -------
        str
            String summary.
        """
        table = Table(title=self.collections.upper())
        table.add_column("Attribute", justify="left", style="white", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("Items Found", str(self.__len__()))
        table.add_row("Bounds", str(self.bounds))
        table.add_row("Time Window", self.time_window)
        table.add_row("Resolution", f"{self.resolution} meter(s) per pixel")
        table.add_row("Scale", f"{self.scale}")
        table.add_row("CRS", self.crs)
        print(table)
        return ""

    def __repr__(self) -> str:
        return self.__str__()


def plot_image(item_name: str, cmap: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """
    Plot a satellite image band or composite with color scaling.

    Parameters
    ----------
    item_name : str
        Variable name of xarray data to plot.
    cmap : str
        Colormap to apply.
    vmin : float, optional
        Minimum value for color normalization.
    vmax : float, optional
        Maximum value for color normalization.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    exec(f"{item_name}.plot.imshow(vmin={vmin}, vmax={vmax}, cmap='{cmap}')")
    plt.title(item_name)
    plt.axis("off")
    plt.show()


def export_to_tif(
    item: Union[xr.DataArray, xr.Dataset],
    output_name: str,
    lower_left: Tuple[float, float],
    upper_right: Tuple[float, float],
    crs: str = "epsg:4326",
    count: Optional[int] = None,
) -> None:
    """
    Export xarray image data to GeoTIFF format.

    Parameters
    ----------
    item : xarray DataArray or Dataset
        Image data to export.
    output_name : str
        File path to save GeoTIFF.
    lower_left : tuple of float
        Lower-left corner coordinates (lat, lon).
    upper_right : tuple of float
        Upper-right corner coordinates (lat, lon).
    crs : str, optional
        Coordinate reference system (default is EPSG:4326).
    count : int, optional
        Number of bands to write (needed for multi-band arrays).
    """
    path = Path(output_name)
    path.resolve().parents[0].mkdir(parents=True, exist_ok=True)

    height = item.sizes["latitude"]
    width = item.sizes["longitude"]
    gt = rasterio.transform.from_bounds(
        lower_left[1], lower_left[0], upper_right[1], upper_right[0], width, height
    )
    item.rio.write_crs(crs, inplace=True)
    item.rio.write_transform(transform=gt, inplace=True)

    with rasterio.open(
        output_name,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        crs=crs,
        transform=gt,
        count=count,
        compress="lzw",
        dtype="float64",
    ) as dst:
        if hasattr(item, "to_dataarray"):
            dst.write(item.to_dataarray())
        elif len(item.shape) == 2:
            dst.write(item.values.reshape(1, height, width))
        else:
            dst.write(item)
