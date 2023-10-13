"""Configuration for pytest."""
from pathlib import Path

import geopandas as gpd
import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add pygeohydro namespace for doctest."""
    import curviriver as cr

    doctest_namespace["cr"] = cr


@pytest.fixture()
def ehydro():
    """Return a curviriver object."""
    url = "/".join(
        (
            "https://ehydrotest.blob.core.usgovcloudapi.net",
            "ehydro-surveys/CEMVN/SW_04_SWP_20230914_CS.ZIP",
        )
    )
    return gpd.read_file(f"zip+{url}!{Path(url).stem}.gdb")
