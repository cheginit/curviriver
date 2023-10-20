.. image:: https://raw.githubusercontent.com/cheginit/curviriver/main/doc/source/_static/logo-text.png
    :target: https://curviriver.readthedocs.io

|

CurviRiver: Curvilinear Mesh Generator for Rivers
=================================================

.. image:: https://github.com/cheginit/curviriver/actions/workflows/test.yml/badge.svg
   :target: https://github.com/cheginit/curviriver/actions/workflows/test.yml
   :alt: CI

.. image:: https://img.shields.io/pypi/v/curviriver.svg
    :target: https://pypi.python.org/pypi/curviriver
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/curviriver.svg
    :target: https://anaconda.org/conda-forge/curviriver
    :alt: Conda Version

.. image:: https://codecov.io/gh/cheginit/curviriver/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/curviriver
    :alt: CodeCov

.. image:: https://img.shields.io/pypi/pyversions/curviriver.svg
    :target: https://pypi.python.org/pypi/curviriver
    :alt: Python Versions

|

.. image:: https://static.pepy.tech/badge/curviriver
    :target: https://pepy.tech/project/curviriver
    :alt: Downloads

.. image:: https://www.codefactor.io/repository/github/cheginit/curviriver/badge/main
    :target: https://www.codefactor.io/repository/github/cheginit/curviriver/overview/main
    :alt: CodeFactor

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

|

Features
--------

CurviRiver takes as input a Polygon of a river segment and generates a 2D or 3D
curvilinear mesh that can be used for hydrodynamic or hydrological modeling.
The mesh is generated in three main steps:

- Determining the centerline of the input Polygon using Voroni diagram
  Dijkstra's algorithm,
- Smoothing the generated centerline with a B-spline curve,
- Computing the tangent angles of the centerline at each point along
  the centerline and generating cross-sections perpendicular to the
  centerline at given intervals,
- Generating a 2D mesh from vertices of the cross-sections,
- Generating a 3D mesh if depth data is provided, by determining the depth of
  2D mesh vertices from the depth data using Inverse Distance Weighting (IDW).

Installation
------------

You can install CurviRiver using ``pip``:

.. code-block:: console

    $ pip install curviriver

or using ``conda`` (``mamba``):

.. code-block:: console

    $ conda install -c conda-forge curviriver

Quick start
-----------

We demonstrate capabilities of CurviRiver by generating a
curvilinear mesh along a portion of the Columbia River and
interpolating
`eHydro <https://www.sam.usace.army.mil/Missions/Spatial-Data-Branch/eHYDRO/>`__
topobathy data on to the generated mesh vertices. Please visit the
`example gallery <https://curviriver.readthedocs.io/en/latest/gallery.html>`__
for more examples.

First, we use `PyGeoHydro <https://docs.hyriver.io/readme/pygeohydro.html>`__
to retrieve eHydro data for a part of the Columbia River that topobathy data are
available. We get both the survey outline and the bathymetry data.
Then, we use the survey outline polygon to generate a curvilinear mesh.
We use the ``poly_segmentize`` function for this purpose that has two
parameters: Spacing in streamwise direction and number of points in
cross-stream direction. The function returns a ``geopandas.GeoSeries``
of the cross-sections, vertices of which are the mesh points.

.. code:: python

    from pygeohydro import EHydro
    import curviriver as cr

    ehydro = EHydro("outlines")
    geom = ehydro.survey_grid.loc[ehydro.survey_grid["OBJECTID"] == 210, "geometry"].iloc[0]
    outline = ehydro.bygeom(geom, ehydro.survey_grid.crs)

    poly = outline.convex_hull.unary_union
    spacing_streamwise = 2000
    xs_npts = 5
    stream = cr.poly_segmentize(poly, outline.crs, spacing_streamwise, xs_npts)

.. image:: https://raw.githubusercontent.com/cheginit/curviriver/main/doc/source/_static/curvilinear.png
  :target: https://github.com/cheginit/curviriver

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/cheginit/curviriver/blob/main/CONTRIBUTING.rst>`__
file for instructions.

Support
-------

The work for this project is funded by USGS through Water Resources Research Institutes
(`WRRI <https://water.usgs.gov/wrri/index.php>`__).
