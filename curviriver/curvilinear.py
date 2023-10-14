"""Generate curvilinear mesh from a polygon."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import shapely
from scipy.spatial import Voronoi
from shapely import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    ops,
)

from curviriver import smoothing
from curviriver.exceptions import (
    InputRangeError,
    InputTypeError,
    LineIntersectionError,
    NoIntersectionError,
    NoMainCenterlineError,
    TooFewRidgesError,
)

if TYPE_CHECKING:
    CRSTYPE = Union[int, str, pyproj.CRS]

__all__ = ["poly_centerline", "line_extension", "line_xsection", "poly_segmentize"]


def _interpolate_line(
    line: LinearRing, x_min: float, y_min: float, interpolation_distance: float
) -> list[tuple[float, float]]:
    first_point = (line.xy[0][0] - x_min, line.xy[1][0] - y_min)
    last_point = (line.xy[0][-1] - x_min, line.xy[1][-1] - y_min)

    intermediate_points = []
    length_tot = line.length
    distance = interpolation_distance
    while distance < length_tot:
        point = line.interpolate(distance)
        intermediate_points.append((point.x - x_min, point.y - y_min))
        distance += interpolation_distance
    return [first_point, *intermediate_points, last_point]


def _poly_centerline(
    geometry: Polygon | MultiPolygon, interpolation_distance: float
) -> MultiLineString:
    """Create centerline from a polygon.

    This function is based on the
    `Centerline <https://github.com/fitodic/centerline>`__
    package (MIT License).

    Parameters
    ----------
    geometry : shapely.Polygon or shapely.MultiPolygon
        Input geometry which can be either ``Polygon``` or ``MultiPolygon``.
    interpolation_distance : float
        Densify the input geometry's border by placing additional
        points at this distance.

    Returns
    -------
    shapely.MultiLineString
        Centerline of the input geometry
    """
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise InputTypeError("line", "Polygon or MultiPolygon")

    x_min = np.floor(min(geometry.envelope.exterior.xy[0]))
    y_min = np.floor(min(geometry.envelope.exterior.xy[1]))
    polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]

    points = []
    for poly in polygons:
        points.extend(_interpolate_line(poly.exterior, x_min, y_min, interpolation_distance))
        if poly.interiors:
            points.extend(
                _interpolate_line(pts, x_min, y_min, interpolation_distance)
                for pts in poly.interiors
            )

    voronoi_diagram = Voronoi(np.array(points, "f8"))
    vertices = voronoi_diagram.vertices
    ridges = voronoi_diagram.ridge_vertices

    c_min = np.array([x_min, y_min])
    linestrings = []
    for ridge in ridges:
        # Check if the ridge is finite
        if -1 not in ridge:
            line = LineString((vertices[ridge[0]] + c_min, vertices[ridge[1]] + c_min))
            if line.within(geometry) and line.coords[0]:
                linestrings.append(line)

    if len(linestrings) < 2:
        raise TooFewRidgesError

    return shapely.line_merge(shapely.unary_union(linestrings))


def _extraplolation(p1: tuple[float, float], p2: tuple[float, float]) -> LineString:
    """Create a line extrapolated in p1 -> p2 direction."""
    ratio = 2
    a = p1
    b = (p1[0] + ratio * (p2[0] - p1[0]), p1[1] + ratio * (p2[1] - p1[1]))
    return LineString([a, b])


def line_extension(
    line: LineString, poly: Polygon | MultiPolygon, both_ends: bool = True
) -> LineString:
    """Extend a line to the boundary of a (multi)polygon.

    Parameters
    ----------
    line : shapely.LineString
        Line to be extended.
    poly : shapely.Polygon or shapely.MultiPolygon
        Polygon to which the line will be extended.
    both_ends : bool, optional
        Whether to extend both ends of the line, defaults to ``True``.

    Returns
    -------
    shapely.LineString
        Extended line.
    """
    if not isinstance(line, LineString):
        raise InputTypeError("line", "LineString")

    if not isinstance(poly, (Polygon, MultiPolygon)):
        raise InputTypeError("poly", "Polygon or MultiPolygon")

    if not line.intersects(poly):
        raise InputTypeError("line", "LineString that intersects with ``poly``")

    # Only need the boundary intersection
    p_exterior = LinearRing(poly.exterior.coords)
    if isinstance(line.intersects(p_exterior), LineString):
        raise LineIntersectionError
    l_coords = list(line.coords)
    l_coords = cast("list[tuple[float, float]]", l_coords)
    while True:
        # Only use the last two points
        l_extraploated = _extraplolation(*l_coords[-2:])
        intersection_points = p_exterior.intersection(l_extraploated)

        if not isinstance(intersection_points, (Point, MultiPoint)):
            new_point_coords = cast("tuple[float, float]", l_extraploated.coords[1])
            l_coords.append(new_point_coords)
            continue

        if isinstance(intersection_points, Point):
            new_point_coords = next(iter(intersection_points.coords))
        elif isinstance(intersection_points, MultiPoint):
            # Use the point closest to the last point
            last_point = Point(l_coords[-1])
            distances = [last_point.distance(point) for point in intersection_points.geoms]
            new_point_coords = list(intersection_points)[distances.index(min(distances))].coords[0]
        else:
            raise NoIntersectionError
        new_point_coords = cast("tuple[float, float]", new_point_coords)
        l_coords.append(new_point_coords)
        break
    line_extended = LineString(l_coords)
    if both_ends:
        return line_extension(line_extended.reverse(), poly, both_ends=False)
    return line_extended


def _longest_path(multi_line: MultiLineString) -> LineString:
    """Find the longest path among all pairs of leaf nodes using Dijkstra's algorithm."""
    net = nx.Graph()

    # Create graph using only the first and last coordinates of each line
    for i, line in enumerate(multi_line.geoms):
        start, end = line.coords[0], line.coords[-1]
        net.add_edge(start, end, weight=1 / line.length, index=i)

    # Identify leaf nodes
    leaf_nodes = [
        node for node, deg in nx.degree(net) if deg == 1  # pyright: ignore[reportGeneralTypeIssues]
    ]
    longest_path = []
    longest_path_length = 0

    # Find the longest path among all pairs of leaf nodes
    for source in leaf_nodes:
        length, path = nx.single_source_dijkstra(net, source, weight="weight")
        for target in leaf_nodes:
            if source == target:
                continue
            path_length = length.get(target, None)  # pyright: ignore[reportGeneralTypeIssues]
            if path_length is not None and path_length > longest_path_length:
                longest_path = path[target]
                longest_path_length = path_length

    # Fetch original lines
    original_lines = [
        multi_line.geoms[net[u][v]["index"]] for u, v in zip(longest_path[:-1], longest_path[1:])
    ]
    main_line = ops.linemerge(original_lines)
    if isinstance(main_line, MultiLineString):
        raise NoMainCenterlineError
    return main_line


def poly_centerline(geometry: Polygon) -> LineString:
    """Create centerline from a polygon.

    This function is based on the
    `Centerline <https://github.com/fitodic/centerline>`__
    package (MIT License).

    Parameters
    ----------
    geometry : shapely.Polygon or shapely.MultiPolygon
        Input geometry which can be either ``Polygon``` or ``MultiPolygon``.

    Returns
    -------
    shapely.LineString
        Centerline of the input geometry
    """
    centerlines = []
    centerline = None
    for c in np.linspace(0.1, 1, 10):
        centerline_interp = geometry.area / geometry.length * c
        centerlines.append(
            _poly_centerline(geometry, centerline_interp).simplify(centerline_interp)
        )
        if isinstance(centerlines[-1], LineString):
            centerline = centerlines[-1]
            break
    if centerline is None:
        centerline = _longest_path(centerlines[-1])

    return line_extension(centerline, geometry)


def __get_idx(d_sp: npt.NDArray[np.float64], distance: float) -> npt.NDArray[np.int64]:
    """Get the index of the closest points based on a given distance."""
    dis = pd.DataFrame(d_sp, columns=["distance"]).reset_index()
    bins = np.arange(0, dis["distance"].max() + distance, distance)
    grouper = pd.cut(dis["distance"], bins)
    idx = dis.groupby(grouper, observed=True).last()["index"].to_numpy("int64")
    return np.append(0, idx)


def __get_spline_params(
    line: LineString, n_seg: int, distance: float, crs: CRSTYPE
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Get Spline parameters (x, y, phi)."""
    _n_seg = n_seg
    spline = smoothing.smooth_linestring(line, crs, _n_seg, degree=5)
    idx = __get_idx(spline.distance, distance)
    while np.isnan(idx).any():
        _n_seg *= 2
        spline = smoothing.smooth_linestring(line, crs, _n_seg, degree=5)
        idx = __get_idx(spline.distance, distance)
    return spline.x[idx], spline.y[idx], spline.phi[idx], spline.distance[idx]


def __get_perpendicular(
    line: LineString, n_seg: int, distance: float, half_width: float, crs: str | int | pyproj.CRS
) -> list[LineString]:
    """Get perpendiculars to a line."""
    x, y, phi, dis = __get_spline_params(line, n_seg, distance, crs)
    x_l = x - half_width * np.sin(phi)
    x_r = x + half_width * np.sin(phi)
    y_l = y + half_width * np.cos(phi)
    y_r = y - half_width * np.cos(phi)
    if np.diff(dis)[-1] < 0.25 * distance:
        x_l = np.delete(x_l, -2)
        x_r = np.delete(x_r, -2)
        y_l = np.delete(y_l, -2)
        y_r = np.delete(y_r, -2)
    return [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in zip(x_l, y_l, x_r, y_r)]


def line_xsection(line: LineString, distance: float, width: float, crs: CRSTYPE) -> gpd.GeoSeries:
    """Get cross-sections along the line at a given spacing.

    Parameters
    ----------
    line : shapely.LineString
        A line along which the cross-sections will be generated.
    distance : float
        The distance between two consecutive cross-sections.
    width : float
        The width of the cross-section.
    crs : str or int or pyproj.CRS
        The CRS of the input line. Using projected CRS is highly recommended.

    Returns
    -------
    geopandas.GeoSeries
        Cross-sections along the line, sorted by line direction.
    """
    n_seg = int(np.ceil(line.length / distance)) * 100
    half_width = width * 0.5
    main_split = __get_perpendicular(line, n_seg, distance, half_width, crs)
    return gpd.GeoSeries(main_split, crs=pyproj.CRS(crs))


def poly_segmentize(
    poly: shapely.Polygon,
    crs: CRSTYPE,
    spacing_streamwise: float,
    xs_npts: int,
) -> gpd.GeoSeries:
    """Segmentize a polygon into a curvilinear grid.

    Parameters
    ----------
    poly : shapely.Polygon
        Polygon to convert to a grid of transects.
    crs : int, str, or pyproj.CRS
        Coordinate reference system of the polygon. Using projected CRS is
        highly recommended.
    spacing_streamwise : float
        Spacing between cross-sections along the polygon's centerline.
    xs_npts : int
        Number of points along each cross-section.

    Returns
    -------
    gpd.GeoSeries
        Cross-sections as a GeoSeries of LineStrings.
    """
    if not isinstance(poly, Polygon):
        raise InputTypeError("poly", "Polygon")

    centerline = poly_centerline(poly)
    if spacing_streamwise > centerline.length:
        raise InputRangeError(
            "spacing_streamwise", f"less than the length of the centerline: {centerline.length}"
        )
    width = poly.area / centerline.length
    while True:
        xs = line_xsection(centerline, spacing_streamwise, width, crs=crs)
        if xs.crosses(poly).all():  # pyright: ignore[reportGeneralTypeIssues]
            break
        width *= 2
    xs = xs.iloc[1:-1].intersection(poly)
    xs = gpd.GeoSeries(
        [
            xs.iloc[0].parallel_offset(spacing_streamwise, "right"),
            *xs.geometry,
            xs.iloc[-1].parallel_offset(spacing_streamwise, "left"),
        ],
        crs=xs.crs,
    )
    return gpd.GeoSeries(
        [
            shapely.LineString([line.interpolate(d) for d in np.linspace(0, line.length, xs_npts)])
            for line in xs.geometry
        ],
        crs=xs.crs,
    )
