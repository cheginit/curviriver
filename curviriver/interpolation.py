"""Interpolating point cloud values to a grid using Inverse Distance Weighting (IDW)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Literal, cast, overload

import cytoolz.curried as tlz
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import scipy.interpolate as sci
import shapely
from scipy.spatial import KDTree
from shapely import LineString
from sklearn.cluster import DBSCAN

from curviriver.exceptions import (
    InputTypeError,
    InputValueError,
    MatchingCRSError,
    ParallelImportError,
)

FloatArray = npt.NDArray[np.float64]


__all__ = [
    "extract_xsections",
    "pc_average_distance",
    "idw_line_interpolation",
    "idw_point_interpolation",
    "xs_hydraulics",
]

if TYPE_CHECKING:
    from scipy.interpolate import BSpline
    from scipy.spatial import cKDTree

FloatArray = npt.NDArray[np.float64]


def extract_xsections(
    point_cloud: FloatArray, centerline: LineString
) -> Generator[FloatArray, None, None]:
    # Calculate the distance along the centerline for each projected point
    distances_along_centerline = shapely.line_locate_point(
        centerline, shapely.points(point_cloud[:, :2])
    )

    # Cluster points based on their distances along the centerline
    tree = KDTree(point_cloud)
    distances, _ = tree.query(point_cloud, k=2)
    non_zero_distances = distances[:, 1]
    eps = np.mean(non_zero_distances) + np.std(non_zero_distances)
    clustering = DBSCAN(eps=eps, min_samples=2).fit(distances_along_centerline.reshape(-1, 1))
    labels = clustering.labels_

    for label in np.unique(labels):
        if label != -1:
            yield point_cloud[labels == label]


def pc_average_distance(
    point_cloud: FloatArray, centerline: LineString | None = None, cross_section: bool = False
) -> np.float64:
    if not cross_section:
        tree = KDTree(point_cloud)
        distances, _ = tree.query(point_cloud, k=2)
        non_zero_distances = distances[:, 1]
        avg_distance = np.mean(non_zero_distances)
        return avg_distance

    if centerline is None:
        raise ValueError("Centerline must be provided when cross_section is True")

    cross_sections = extract_xsections(point_cloud, centerline)
    prev_section = next(cross_sections, None)

    if prev_section is None:
        return np.float64(0)

    intra_section_distances = []
    inter_section_distances = []
    for current_section in cross_sections:
        # Intra-section distances for the previous section
        tree = KDTree(prev_section)
        distances, _ = tree.query(prev_section, k=2)
        non_zero_distances = distances[:, 1]
        intra_section_distances.append(np.mean(non_zero_distances))

        # Inter-section distances between the previous and current sections
        distances, _ = tree.query(current_section)
        inter_section_distances.append(np.mean(distances))

        prev_section = current_section

    # Intra-section distances for the last section
    tree = KDTree(prev_section)
    distances, _ = tree.query(prev_section, k=2)
    non_zero_distances = distances[:, 1]
    intra_section_distances.append(np.mean(non_zero_distances))

    intra_section_avg = (
        np.mean(intra_section_distances) if intra_section_distances else np.float64(0)
    )
    inter_section_avg = (
        np.mean(inter_section_distances) if inter_section_distances else np.float64(0)
    )

    overall_avg_distance = (intra_section_avg + inter_section_avg) * 0.5
    return overall_avg_distance


@overload
def _idw(
    xy: zip[tuple[float, float]] | FloatArray,
    tree: cKDTree[None],
    values: FloatArray,
    search_radius: np.float64,
    to_linestring: Literal[False],
) -> FloatArray:
    ...


@overload
def _idw(
    xy: zip[tuple[float, float]] | FloatArray,
    tree: cKDTree[None],
    values: FloatArray,
    search_radius: np.float64,
    to_linestring: Literal[True],
) -> LineString:
    ...


def _idw(
    xy: zip[tuple[float, float]] | FloatArray,
    tree: cKDTree[None],
    values: FloatArray,
    search_radius: np.float64,
    to_linestring: bool,
) -> LineString | FloatArray:
    points = tuple(xy)
    interp_values = []
    for point in points:
        indices = tree.query_ball_point(point, r=search_radius, p=2)
        if not indices:
            interp_values.append(np.nan)
            continue
        local_distances = np.linalg.norm(tree.data[indices] - np.array(point), axis=1)
        weights = 1 / (local_distances**2)
        interp_values.append(np.dot(weights, values[indices]) / np.sum(weights))

    if to_linestring:
        return LineString(np.c_[points, interp_values])
    return np.c_[points, interp_values]


def _get_tree(
    point_cloud: gpd.GeoDataFrame,
    value_col: str,
) -> tuple[cKDTree[None], FloatArray]:
    if not isinstance(point_cloud, gpd.GeoDataFrame):
        raise InputTypeError("point_cloud", "GeoDataFrames")

    if value_col not in point_cloud.columns:
        raise InputValueError("value_col", list(point_cloud.columns))

    if not point_cloud.geom_type.eq("Point").all():
        raise InputTypeError("point_cloud", "geometries of type Point")

    tree = KDTree(point_cloud.get_coordinates().to_numpy())
    values = point_cloud[value_col].to_numpy()
    values = cast("FloatArray", values)
    return tree, values


@overload
def idw_line_interpolation(
    point_cloud: gpd.GeoDataFrame,
    xs_lines: gpd.GeoDataFrame | gpd.GeoSeries,
    value_col: str,
    centerline: LineString | None = None,
    pc_xsection: bool = False,
    search_radius_coeff: float = 3.0,
    grid_points: Literal[False] = False,
    parallel: bool = False,
) -> gpd.GeoSeries:
    ...


@overload
def idw_line_interpolation(
    point_cloud: gpd.GeoDataFrame,
    xs_lines: gpd.GeoDataFrame | gpd.GeoSeries,
    value_col: str,
    centerline: LineString | None = None,
    pc_xsection: bool = False,
    search_radius_coeff: float = 3.0,
    grid_points: Literal[True] = True,
    parallel: bool = False,
) -> gpd.GeoDataFrame:
    ...


def idw_line_interpolation(
    point_cloud: gpd.GeoDataFrame,
    xs_lines: gpd.GeoDataFrame | gpd.GeoSeries,
    value_col: str,
    centerline: LineString | None = None,
    pc_xsection: bool = False,
    search_radius_coeff: float = 3.0,
    grid_points: bool = False,
    parallel: bool = False,
) -> gpd.GeoDataFrame | gpd.GeoSeries:
    """Interpolate point cloud values to a grid using Inverse Distance Weighting (IDW).

    Parameters
    ----------
    point_cloud : geopandas.GeoDataFrame
        Point cloud with values to be interpolated.
        Using a projected CRS is highly recommended.
    xs_lines : geopandas.GeoDataFrame or geopandas.GeoSeries
        Cross-section lines to which values will be interpolated.
        Using a projected CRS is highly recommended.
    value_col : str
        Name of the column in point_cloud containing values to be interpolated.
    centerline : shapely.LineString, optional
        Centerline of ``point_cloud`` convex hull. Required when ``pc_xsection=True``.
    pc_xsection : bool, optional
        Whether the point cloud is from surveys done perpendicular to the centerline,
        instead of many points along the centerline. Default is ``False``.
    search_radius_coeff : float, optional
        Coefficient to multiply the average distance between the closest and second
        closest points to estimate the search radius. Default is 3.
    grid_points: bool, optional
        If ``True``, the interpolated values will be returned as a GeoDataFrame
        with the grid points as geometries. Default is ``False``.
    parallel : bool, optional
        Whether to use parallel processing, defaults to ``False``.
        For parallel processing, ``joblib`` package must be installed.

    Returns
    -------
    geopandas.GeoSeries or geopandas.GeoDataFrame
        If ``grid_points=False`` a GeoSeries of 3D line where the Z-coordinate
        is the interpolated value. If ``grid_points=True`` a GeoDataFrame of
        2D points with a column called ``value_col`` containing the interpolated value.
    """
    tree, values = _get_tree(point_cloud, value_col)
    if not isinstance(xs_lines, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise InputTypeError("xs_lines", "GeoDataFrames or GeoSeries")

    if not xs_lines.geom_type.eq("LineString").all():
        raise InputTypeError("xs_lines", "geometries of type LineString")

    if point_cloud.crs != xs_lines.crs:
        raise MatchingCRSError

    avg_distance = pc_average_distance(tree.data, centerline, cross_section=pc_xsection)
    search_radius = search_radius_coeff * avg_distance

    if parallel:
        try:
            from joblib import Parallel, delayed
        except ImportError as ex:
            raise ParallelImportError from ex

        interpolated_vals = Parallel(n_jobs=-1)(
            delayed(_idw)(zip(*line.coords.xy), tree, values, search_radius, True)
            for line in xs_lines.geometry
        )
        interpolated_vals = cast("list[LineString]", interpolated_vals)
    else:
        interpolated_vals = [
            _idw(zip(*line.coords.xy), tree, values, search_radius, True)
            for line in xs_lines.geometry
        ]
    xs_interp = gpd.GeoSeries(interpolated_vals, crs=point_cloud.crs)

    if grid_points:
        x, y, z = shapely.get_coordinates(xs_interp, include_z=True).T
        return gpd.GeoDataFrame(
            {value_col: z}, geometry=gpd.points_from_xy(x, y, crs=xs_interp.crs)
        )
    return xs_interp


def idw_point_interpolation(
    point_cloud: FloatArray,
    grid_points: FloatArray | zip[tuple[float, float]],
    centerline: LineString | None = None,
    pc_xsection: bool = False,
    search_radius_coeff: float = 3.0,
    parallel: bool = False,
) -> FloatArray:
    """Interpolate point cloud values to a grid using Inverse Distance Weighting (IDW).

    Parameters
    ----------
    point_cloud : numpy.ndarray
        Point cloud with values to be interpolated. The array must have 3 columns
        with the X, Y coordinates of the points as the first two columns and the
        values to be interpolated as the third column.
    grid_points : numpy.ndarray
        Grid points to which values will be interpolated. The array must have 2
        columns with the X, Y coordinates of the points. The coordinates must be
        in the same CRS as point_cloud.
    centerline : shapely.LineString, optional
        Centerline of ``point_cloud`` convex hull. Required when ``pc_xsection=True``.
    pc_xsection : bool, optional
        Whether the point cloud is from surveys done perpendicular to the centerline,
        instead of many points along the centerline. Default is ``False``.
    search_radius_coeff : float, optional
        Coefficient to multiply the average distance between the closest and second
        closest points to estimate the search radius. Default is 3.
    parallel : bool, optional
        Whether to use parallel processing, defaults to ``False``.
        For parallel processing, ``joblib`` package must be installed.

    Returns
    -------
    numpy.ndarray
        Grid points with a third column containing the interpolated values.
    """
    tree = KDTree(point_cloud[:, :2])
    values = point_cloud[:, 2]
    avg_distance = pc_average_distance(tree.data, centerline, cross_section=pc_xsection)
    search_radius = search_radius_coeff * avg_distance

    if parallel:
        try:
            from joblib import Parallel, delayed
        except ImportError as ex:
            raise ParallelImportError from ex

        xys = tlz.partition_all(500, grid_points)
        results = Parallel(n_jobs=-1)(
            delayed(_idw)(xy, tree, values, search_radius, False) for xy in xys
        )
        results = cast("list[FloatArray]", results)
        return np.concatenate(results)
    return _idw(grid_points, tree, values, search_radius, False)


def xs_hydraulics(line: LineString) -> tuple[float, float]:
    """Compute cross-section hydraulics properties.

    Parameters
    ----------
    line : shapely.LineString
        Cross-section line with Z-coordinate.

    Returns
    -------
    area : float
        Cross-sectional area.
    preimeter : float
        Cross-sectional perimeter.
    """
    if not isinstance(line, LineString):
        raise InputTypeError("line", "LineString")
    if not line.has_z:
        raise InputValueError("line", "LineString with Z-coordinate")

    x, y, z = shapely.get_coordinates(line, include_z=True).T
    dis = np.hypot(np.diff(x), np.diff(y)).cumsum()
    dis = np.insert(dis, 0, 0)
    k = np.clip(3, 1, x.size - 1)
    spl = sci.make_interp_spline(dis, z, k)
    spl = cast("BSpline", spl)
    area = spl.integrate(dis[0], dis[-1]).item()  # pyright: ignore[reportGeneralTypeIssues]
    preimeter = dis[-1]
    return area, preimeter
