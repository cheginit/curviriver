import numpy as np
import pytest

import curviriver as cr
from curviriver.exceptions import InputTypeError


def test_dimension_check():
    with pytest.raises(InputTypeError, match="1D arrays."):
        cr.interpolate_na(np.array([[1, 2], [3, 4]]), np.array([1, 2]), np.array([1, 2]), 0.0)


def test_size_check():
    with pytest.raises(InputTypeError, match="same size"):
        cr.interpolate_na(np.array([1, 2]), np.array([1, 2]), np.array([1, 2, 3]), 0.0)


def test_no_nans():
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 2, 3])
    z = np.array([1, 2, 3, 4])
    assert np.isclose(cr.interpolate_na(x, y, z, 0.0), z).all()


def test_all_nans():
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 2, 3])
    z = np.array([np.nan, np.nan, np.nan, np.nan])
    expected = np.array([0.0, 0.0, 0.0, 0.0])
    assert np.isclose(cr.interpolate_na(x, y, z, 0.0), expected).all()


def test_interpolate_middle():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 2, 3, 4, 5])
    z = np.array([0.0, 1.0, np.nan, np.nan, 4.0, 5.0])
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(cr.interpolate_na(x, y, z, 0.0), expected).all()


def test_fill_begin_end():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 2, 3, 4, 5])
    z = np.array([np.nan, np.nan, 2.0, 3.0, 4.0, np.nan])
    expected = np.array([0.0, 0.0, 2.0, 3.0, 4.0, 0.0])
    assert np.isclose(cr.interpolate_na(x, y, z, 0.0), expected).all()


def test_mixed_nans():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 2, 3, 4, 5])
    z = np.array([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan])
    expected = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
    assert np.isclose(cr.interpolate_na(x, y, z, 0.0), expected).all()
