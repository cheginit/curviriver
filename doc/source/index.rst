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

.. image:: https://raw.githubusercontent.com/cheginit/curviriver/main/doc/source/_static/curvilinear.png
  :target: https://github.com/cheginit/curviriver

Support
-------

The work for this project is funded by USGS through Water Resources Research Institutes
(`WRRI <https://water.usgs.gov/wrri/index.php>`__).

.. toctree::
    :hidden:

    usage
    gallery
    autoapi/index
    changelog
    contributing
    authors
    license
