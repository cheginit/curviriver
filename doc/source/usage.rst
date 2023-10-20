Getting Started
===============

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
