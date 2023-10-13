"""Some helper functions."""
from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, TypeVar, cast

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from shapely import Polygon

from curviriver.exceptions import GeometryError, InputTypeError

if TYPE_CHECKING:
    from shapely import MultiPolygon

    GDFTYPE = TypeVar("GDFTYPE", gpd.GeoDataFrame, gpd.GeoSeries)

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
coord_dtype = np.dtype([("x", np.float64), ("y", np.float64)])


def _get_area_range(mp: MultiPolygon) -> float:
    """Get the range of areas of polygons in a multipolygon."""
    if np.isclose(mp.area, 0.0):
        return 0.0
    return np.ptp([g.area for g in mp.geoms]) / mp.area


def _get_larges(mp: MultiPolygon) -> Polygon:
    """Get the largest polygon from a multipolygon."""
    return Polygon(
        mp.geoms[
            np.argmax([g.area for g in mp.geoms])
        ].exterior  # pyright: ignore[reportOptionalMemberAccess]
    )


def multi_explode(gdf: GDFTYPE) -> GDFTYPE:
    """Convert multi-part geometries to single-part and fill polygon holes, if any.

    Notes
    -----
    This function tries to convert multi-geometries to their constituents by
    first checking if multiploygons can be directly converted using
    their exterior boundaries. If not, it will try to remove those small
    sub-polygons that their area is less than 1% of the total area
    of the multipolygon. If this fails, the multi-geometries will be exploded.
    Thus, the number of rows in the output GeoDataFrame may be larger than
    the input GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with single-part geometries.

    Returns
    -------
    geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with polygons.
    """
    if not isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise InputTypeError("gdf", "GeoDataFrame or GeoSeries")

    gdf_prj = cast("GDFTYPE", gdf.copy())
    if isinstance(gdf_prj, gpd.GeoSeries):
        gdf_prj = gpd.GeoDataFrame(gdf_prj.to_frame("geometry"))

    mp_idx = gdf_prj.loc[gdf_prj.geom_type == "MultiPolygon"].index
    if mp_idx.size > 0:
        geo_mp = gdf_prj.loc[mp_idx, "geometry"]
        geo_mp = cast("gpd.GeoSeries", geo_mp)
        idx = {i: g.geoms[0] for i, g in geo_mp.geometry.items() if len(g.geoms) == 1}
        gdf_prj.loc[list(idx), "geometry"] = list(idx.values())
        if len(idx) < len(geo_mp):
            area_rng = geo_mp.map(_get_area_range)
            mp_idx = area_rng[area_rng >= 0.99].index  # pyright: ignore[reportGeneralTypeIssues]
            if mp_idx.size > 0:
                gdf_prj.loc[mp_idx, "geometry"] = geo_mp.map(_get_larges)

    if gdf_prj.geom_type.str.contains("Multi").any():
        gdf_prj["multipart"] = [
            list(g.geoms) if "Multi" in g.type else [g] for g in gdf_prj.geometry
        ]
        gdf_prj = gdf_prj.explode("multipart")
        gdf_prj = cast("gpd.GeoDataFrame", gdf_prj)
        gdf_prj = gdf_prj.set_geometry("multipart", drop=True)
        gdf_prj = cast("gpd.GeoDataFrame", gdf_prj)

    if not gdf_prj.is_simple.all():
        gdf_prj["geometry"] = gdf_prj.buffer(0)

    if isinstance(gdf, gpd.GeoSeries):
        return gdf_prj.geometry

    return gdf_prj
