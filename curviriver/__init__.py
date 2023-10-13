"""Top-level API for curviriver."""
from importlib.metadata import PackageNotFoundError, version

from curviriver import exceptions
from curviriver.curvilinear import line_extension, line_xsection, poly_centerline, poly_segmentize
from curviriver.interpolation import (
    extract_xsections,
    idw_line_interpolation,
    idw_point_interpolation,
    pc_average_distance,
    xs_hydraulics,
)
from curviriver.smoothing import interpolate_na, make_bspline, smooth_linestring

try:
    __version__ = version("curviriver")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "poly_centerline",
    "poly_segmentize",
    "line_extension",
    "line_xsection",
    "idw_line_interpolation",
    "idw_point_interpolation",
    "xs_hydraulics",
    "make_bspline",
    "smooth_linestring",
    "interpolate_na",
    "extract_xsections",
    "pc_average_distance",
    "exceptions",
]
