GeoMesher: Meshing a GeoDataFrame using Gmsh
============================================

.. image:: https://github.com/cheginit/geomesher/actions/workflows/test.yml/badge.svg
   :target: https://github.com/cheginit/geomesher/actions/workflows/test.yml
   :alt: CI

.. image:: https://img.shields.io/pypi/v/geomesher.svg
    :target: https://pypi.python.org/pypi/geomesher
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/geomesher.svg
    :target: https://anaconda.org/conda-forge/geomesher
    :alt: Conda Version

.. image:: https://codecov.io/gh/cheginit/geomesher/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/geomesher
    :alt: CodeCov

.. image:: https://img.shields.io/pypi/pyversions/geomesher.svg
    :target: https://pypi.python.org/pypi/geomesher
    :alt: Python Versions

|

.. image:: https://static.pepy.tech/badge/geomesher
    :target: https://pepy.tech/project/geomesher
    :alt: Downloads

.. image:: https://www.codefactor.io/repository/github/cheginit/geomesher/badge/main
    :target: https://www.codefactor.io/repository/github/cheginit/geomesher/overview/main
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

GeoMesher is a fork of `pandamesh <https://github.com/Deltares/pandamesh>`__. The original
package included two mesh generators: `Triangle <https://www.cs.cmu.edu/~quake/triangle.html>`__
and `Gmsh <https://gmsh.info/>`__. This fork only includes the Gmsh mesh generator since
Triangle seems to be not maintained anymore. Also, GeoMesher adds the following new
functionalities:

* A new method for returning the generated mesh as a GeoDataFrame.
* A new function called ``gdf_mesher`` that can generate a mesh from a GeoDataFrame
  with a single function call and with sane defaults for the mesh generator.
* Remap a scalar field from the source GeoDataFrame to the generated mesh,
  using an area weighted interpolation method
  (based on `Tobler <https://github.com/pysal/tobler>`__).
* Handle ``MultiPolygon`` geometries in the source GeoDataFrame.

Note that the remapping functionality of GeoMesher is a simple areal interpolation method.
For more advanced interpolation methods, please use `Tobler <https://pysal.org/tobler/index.html>`__.

.. image:: https://raw.githubusercontent.com/cheginit/geomesher/main/doc/source/_static/demo.png
  :target: https://github.com/cheginit/geomesher

Credits
-------

GeoMesher is a fork of `pandamesh <https://github.com/Deltares/pandamesh>`__ (MIT License)
and uses one of the modules in
`Tobler <https://pysal.org/tobler/index.html>`__ (BSD-3-Clause License).

.. toctree::
    :hidden:

    usage
    gallery
    autoapi/index
    changelog
    contributing
    authors
    license
