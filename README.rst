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

- Determining the centerline of the input Polygon using Voroni diagram,
- Generating a B-spline curve from the centerline,
- Computing the angle of the centerline at each point and generating
  cross-sections perpendicular to the centerline at given intervals,
- Generate a 2D mesh from vertices of the cross-sections,
- If depth data is provided, generate a 3D mesh by determining the depth of
  each vertex from the depth data using Inverse Distance Weighting (IDW).

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

.. code:: python

    import geopandas as gpd
    import curviriver as cr


.. image:: https://raw.githubusercontent.com/cheginit/curviriver/main/doc/source/_static/demo.png
  :target: https://github.com/cheginit/curviriver

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/cheginit/pygeoogc/blob/main/CONTRIBUTING.rst>`__
file for instructions.

Support
-------

The work for this project is funded by USGS through Water Resources Research Institutes
(`WRRI <https://water.usgs.gov/wrri/index.php>`__).
