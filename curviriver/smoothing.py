"""Smoothing functions for LineString objects or a set of coordinates.

This module is taken from the `pygeoutils` package, which is available at
`HyRiver <https://docs.hyriver.io/>`__.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, Union, cast

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import scipy.interpolate as sci
import shapely
from shapely import LineString, MultiLineString, Point

from curviriver.exceptions import InputRangeError, InputTypeError

FloatArray = npt.NDArray[np.float64]

if TYPE_CHECKING:
    import pyproj
    from scipy.interpolate import BSpline

    GDFTYPE = TypeVar("GDFTYPE", gpd.GeoDataFrame, gpd.GeoSeries)
    CRSTYPE = Union[int, str, pyproj.CRS]

__all__ = [
    "make_bspline",
    "smooth_linestring",
    "interpolate_na",
]


@dataclass
class Spline:
    """Provide attributes of an interpolated B-spline.

    Attributes
    ----------
    x : numpy.ndarray
        The x-coordinates of the interpolated points.
    y : numpy.ndarray
        The y-coordinates of the interpolated points.
    phi : numpy.ndarray
        Angle of the tangent of the B-spline curve.
    curvature : numpy.ndarray
        Curvature of the B-spline curve.
    radius : numpy.ndarray
        Radius of curvature of the B-spline.
    distance : numpy.ndarray
        Total distance of each point along the B-spline from the start point.
    line : shapely.LineString
        The B-spline as a shapely.LineString.
    """

    x: FloatArray
    y: FloatArray
    phi: FloatArray
    curvature: FloatArray
    radius: FloatArray
    distance: FloatArray

    @property
    def line(self) -> LineString:
        """Convert the B-spline to shapely.LineString."""
        return LineString(zip(self.x, self.y))


def _adjust_boundaries(arr: FloatArray) -> FloatArray:
    """Adjust the boundaries of an array."""
    arr[0] = arr[1]
    arr[-1] = arr[-2]
    return arr


def bspline_curvature(
    bspline: BSpline, konts: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute the curvature of a B-spline curve.

    Notes
    -----
    The formula for the curvature of a B-spline curve is:

    .. math::

        \kappa = \frac{\dot{x}\ddot{y} - \ddot{x}\dot{y}}{(\dot{x}^2 + \dot{y}^2)^{3/2}}

    where :math:`\dot{x}` and :math:`\dot{y}` are the first derivatives of the
    B-spline curve and :math:`\ddot{x}` and :math:`\ddot{y}` are the second
    derivatives of the B-spline curve. Also, the radius of curvature is:

    .. math::

        \rho = \frac{1}{|\kappa|}

    Parameters
    ----------
    bspline : scipy.interpolate.BSpline
        B-spline curve.
    konts : numpy.ndarray
        Knots along the B-spline curve to compute the curvature at. The knots
        must be strictly increasing.

    Returns
    -------
    phi : numpy.ndarray
        Angle of the tangent of the B-spline curve.
    curvature : numpy.ndarray
        Curvature of the B-spline curve.
    radius : numpy.ndarray
        Radius of curvature of the B-spline curve.
    """
    dx, dy = bspline.derivative(1)(konts).T
    dx = _adjust_boundaries(dx)
    dy = _adjust_boundaries(dy)
    phi = np.arctan2(dy, dx)

    if bspline.k >= 2:
        ddx, ddy = bspline.derivative(2)(konts).T
    else:
        ddx = np.zeros_like(dx)
        ddy = np.zeros_like(dy)
    ddx = _adjust_boundaries(ddx)
    ddy = _adjust_boundaries(ddy)
    curvature = (dx * ddy - ddx * dy) / np.float_power(np.square(dx) + np.square(dy), 1.5)
    curvature[~np.isfinite(curvature)] = 0
    with np.errstate(divide="ignore"):
        radius = np.reciprocal(np.abs(curvature))
    return phi, curvature, radius


def make_bspline(x: FloatArray, y: FloatArray, n_pts: int, k: int = 3) -> Spline:
    """Create a B-spline curve from a set of points.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates of the points.
    y : numpy.ndarray
        y-coordinates of the points.
    n_pts : int
        Number of points in the output spline curve.
    k : int, optional
        Degree of the spline. Should be an odd number less than the number of
        points and greater than 1. Default is 3.

    Returns
    -------
    :class:`Spline`
        A Spline object with ``x``, ``y``, ``phi``, ``radius``, ``distance``,
        and ``line`` attributes. The ``line`` attribute returns the B-spline
        as a ``shapely.LineString``.
    """
    k = np.clip(k, 1, x.size - 1)
    konts = np.hypot(np.diff(x), np.diff(y)).cumsum()
    konts = np.insert(konts, 0, 0)
    spl = sci.make_interp_spline(konts, np.c_[x, y], k)
    spl = cast("BSpline", spl)

    konts = np.linspace(konts[0], konts[-1], n_pts)
    x_sp, y_sp = spl(konts).T
    x_sp = cast("FloatArray", x_sp)
    y_sp = cast("FloatArray", y_sp)
    phi_sp, curv_sp, rad_sp = bspline_curvature(spl, konts)
    d_sp = np.hypot(np.diff(x_sp), np.diff(y_sp)).cumsum()
    d_sp = np.insert(d_sp, 0, 0)
    if n_pts < 3:
        idx = np.r_[:n_pts]
        return Spline(x_sp[idx], y_sp[idx], phi_sp[idx], curv_sp[idx], rad_sp[idx], d_sp[idx])

    return Spline(x_sp, y_sp, phi_sp, curv_sp, rad_sp, d_sp)


class GeoBSpline:
    """Create B-spline from a GeoDataFrame of points.

    Parameters
    ----------
    points : geopandas.GeoDataFrame or geopandas.GeoSeries
        Input points as a ``GeoDataFrame`` or ``GeoSeries``. The results
        will be more accurate if the CRS is projected.
    npts_sp : int
        Number of points in the output spline curve.
    degree : int, optional
        Degree of the spline. Should be less than the number of points and
        greater than 1. Default is 3.

    Examples
    --------
    >>> import geopandas as gpd
    >>> xl, yl = zip(
    ...     *[
    ...         (-97.06138, 32.837),
    ...         (-97.06133, 32.836),
    ...         (-97.06124, 32.834),
    ...         (-97.06127, 32.832),
    ...     ]
    ... )
    >>> pts = gpd.GeoSeries(gpd.points_from_xy(xl, yl, crs=4326))
    >>> sp = GeoBSpline(pts.to_crs(3857), 5).spline
    >>> pts_sp = gpd.GeoSeries(gpd.points_from_xy(sp.x, sp.y, crs=3857))
    >>> pts_sp = pts_sp.to_crs(4326)
    >>> list(zip(pts_sp.x, pts_sp.y))
    [(-97.06138, 32.837),
    (-97.06132, 32.83575),
    (-97.06126, 32.83450),
    (-97.06123, 32.83325),
    (-97.06127, 32.83200)]
    """

    def __init__(self, points: GDFTYPE, n_pts: int, degree: int = 3) -> None:
        self.degree = degree

        if any(points.geom_type != "Point"):
            raise InputTypeError("points.geom_type", "Point")
        self.points = points

        if n_pts < 1:
            raise InputRangeError("n_pts", ">= 1")
        self.n_pts = n_pts

        tx, ty = zip(*(g.xy for g in points.geometry))
        self.x_ln = np.array(tx, dtype="f8").squeeze()
        self.y_ln = np.array(ty, dtype="f8").squeeze()
        self.npts_ln = self.x_ln.size
        if self.npts_ln < self.degree:
            raise InputRangeError("degree", f"< {self.npts_ln}")
        self._spline = make_bspline(self.x_ln, self.y_ln, self.n_pts, self.degree)

    @property
    def spline(self) -> Spline:
        """Get the spline as a ``Spline`` object."""
        return self._spline


def smooth_linestring(
    line: LineString | MultiLineString, crs: CRSTYPE, n_pts: int, degree: int = 3
) -> Spline:
    """Smooth a line using B-spline interpolation.

    Parameters
    ----------
    line : shapely.LineString, shapely.MultiLineString
        Line to smooth. Note that if ``line`` is ``MultiLineString``
        it will be merged into a single ``LineString``. If the merge
        fails, an exception will be raised.
    crs : int, str, or pyproj.CRS
        CRS of the input line. It must be a projected CRS.
    n_pts : int
        Number of points in the output spline curve.
    degree : int, optional
        Degree of the spline. Should be less than the number of points and
        greater than 1. Default is 3.

    Returns
    -------
    :class:`Spline`
        A :class:`Spline` object with ``x``, ``y``, ``phi``, ``radius``,
        ``distance``, and ``line`` attributes. The ``line`` attribute
        returns the B-spline as a shapely.LineString.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import shapely
    >>> line = shapely.LineString(
    ...     [
    ...         (-97.06138, 32.837),
    ...         (-97.06133, 32.836),
    ...         (-97.06124, 32.834),
    ...         (-97.06127, 32.832),
    ...     ]
    ... )
    >>> sp = smooth_linestring(line, 4326, 5)
    >>> list(zip(*sp.line.xy))
    [(-97.06138, 32.837),
    (-97.06132, 32.83575),
    (-97.06126, 32.83450),
    (-97.06123, 32.83325),
    (-97.06127, 32.83200)]
    """
    line = shapely.line_merge(line)
    if not isinstance(line, LineString):
        raise InputTypeError("line", "LineString")

    points = gpd.GeoSeries([Point(xy) for xy in zip(*line.xy)], crs=crs)
    return GeoBSpline(points, n_pts, degree).spline


def interpolate_na(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    fill_value: float,
) -> npt.NDArray[np.float64]:
    """Interpolate NaN values in ``z`` using B-spline interpolation.

    Notes
    -----
    The input ``x``, ``y``, and ``z`` must be 1D arrays of the same size.

    Parameters
    ----------
    x : numpy.ndarray
        1D arrays representing the X-coordinates of the points.
    y : numpy.ndarray
        1D arrays representing the Y-coordinates of the points.
    z : numpy.ndarray
        1D arrays representing the Z-coordinates of the points.
    fill_value : float
        Value to use to fill NaNs at the beginning and end of the array.

    Returns
    -------
    numpy.ndarray
        The input array ``z`` with NaN values interpolated or filled.

    Raises
    ------
    ValueError: If input arrays x, y, and z are not 1D or not of the same size.
    """
    # Check input constraints
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        raise InputTypeError("x/y/z", "1D arrays.")
    if x.size != y.size or x.size != z.size:
        raise InputTypeError("x/y/z", "1D arrays of the same size")

    nan_mask = np.isnan(z)
    if nan_mask.sum() == 0:
        return z

    if nan_mask.all():
        return np.full_like(z, fill_value)

    first_non_nan, last_non_nan = np.argwhere(~nan_mask)[[0, -1], 0]

    # Fill beginning and end with fill_value
    z = z.copy()
    z[:first_non_nan] = fill_value
    z[last_non_nan + 1 :] = fill_value

    # Interpolate NaNs in the middle
    interp_slice = np.r_[first_non_nan : last_non_nan + 1]
    interp_notna = ~nan_mask[interp_slice]
    r = np.hypot(np.diff(x[interp_slice]), np.diff(y[interp_slice])).cumsum()
    r = np.insert(r, 0, 0)
    # Apply B-spline interpolation to fill NaNs in the interpolation range
    bspline = make_bspline(r[interp_notna], z[interp_slice][interp_notna], len(interp_slice))
    z[interp_slice] = bspline.y
    return z
