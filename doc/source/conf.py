"""Sphinx configuration."""
import inspect
import os
import sys
from pathlib import Path
from textwrap import dedent, indent

import yaml
from sphinx.application import Sphinx
from sphinx.util import logging

import curviriver

LOGGER = logging.getLogger("conf")

print(f"xarray: {curviriver.__version__}, {curviriver.__file__}")

nbsphinx_allow_errors = False

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinx.ext.linkcode",
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "jupyter_sphinx",
    "sphinx_design",
    "sphinx_favicon",
    "numpydoc",
]

extlinks = {
    "issue": ("https://github.com/cheginit/curviriver/issues/%s", "GH%s"),
    "pull": ("https://github.com/cheginit/curviriver/pull/%s", "PR%s"),
}

# sphinx-copybutton configurations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# nbsphinx configurations
nbsphinx_timeout = 600
nbsphinx_execute = "never"
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None).rsplit("/", 1)[-1] %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::

        This page was generated from `{{ docname }}`__.
        Interactive online version:
        :raw-html:`<a href="https://mybinder.org/v2/gh/cheginit/curviriver/main?urlpath=lab/tree/doc/source/examples/{{ docname }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`

    __ https://github.com/cheginit/curviriver/tree/main/doc/source/examples/{{ docname }}
"""

# autoapi configurations
autoapi_dirs = [
    "../../curviriver",
]
autoapi_ignore = [
    "**ipynb_checkpoints**",
    "**tests**",
    "**conf.py",
    "**conftest.py",
    "**noxfile.py",
    "**exceptions.py",
    "**cli.py",
]
autoapi_options = ["members"]
autoapi_member_order = "groupwise"
modindex_common_prefix = ["curviriver."]

# -- Options for autosummary/autodoc output ------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"
autodoc_typehints_description_target = "documented"

# Napoleon configurations
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "file-like": ":term:`file-like <file-like object>`",
    # objects without namespace: curviriver
    "MeshAlgorithm": "~curviriver.mesher.MeshAlgorithm",
    "SubdivisionAlgorithm": "~curviriver.mesher.SubdivisionAlgorithm",
    "FieldCombination": "~curviriver.mesher.FieldCombination",
}

# General information about the project.
project = "CurviRiver"
copyright = "2023, Taher Chegini"
today_fmt = "%Y-%m-%d"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Sitemap -----------------------------------------------------------------

# ReadTheDocs has its own way of generating sitemaps, etc.
if not os.environ.get("READTHEDOCS"):
    extensions += ["sphinx_sitemap"]

    html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
    sitemap_locales = [None]
    sitemap_url_scheme = "{link}"

# -- Internationalization ----------------------------------------------------

# specifying the natural language populates some key tags
language = "en"

# -- MyST options ------------------------------------------------------------

# This allows us to use ::: to denote directives, useful for admonitions
myst_enable_extensions = ["colon_fence", "linkify", "substitution"]
myst_heading_anchors = 2
myst_substitutions = {"rtd": "[Read the Docs](https://readthedocs.org/)"}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_sourcelink_suffix = ""
html_last_updated_fmt = ""  # to reveal the build date in the pages meta

# Define the json_url for our version switcher.
json_url = "https://curviriver.readthedocs.io/en/latest/_static/switcher.json"

release = curviriver.__version__

# Define the version we use for matching in the version switcher.
version_match = os.environ.get("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# If it's "latest" → change to "dev" (that's what we want the switcher to call it)
if not version_match or version_match.isdigit() or version_match == "latest":
    # For local development, infer the version to match from the package.
    if "dev" in release or "rc" in release:
        version_match = "dev"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = "v" + release

html_theme_options = {
    "external_links": [
        {
            "url": "https://docs.hyriver.io/",
            "name": "HyRiver",
        },
    ],
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "Twitter",
            "url": "https://twitter.com/_taher_",
            "icon": "fa-brands fa-twitter",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/cheginit/curviriver",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/curviriver",
            "icon": "fa-custom fa-pypi",
        },
    ],
    # alternative way to set twitter and github header icons
    # "github_url": "https://github.com/pydata/pydata-sphinx-theme",
    # "twitter_url": "https://twitter.com/PyData",
    "logo": {
        "text": "CurviRiver",
        "image_dark": "_static/logo.png",
        "alt_text": "CurviRiver",
    },
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "navbar_align": "left",  # [left, content, right] For testing that the navbar items align properly
    "navbar_center": ["version-switcher", "navbar-nav"],
    # "announcement": "https://raw.githubusercontent.com/pydata/pydata-sphinx-theme/main/docs/_templates/custom-template.html",
    "show_version_warning_banner": False,
    # "show_nav_level": 2,
    # "navbar_start": ["navbar-logo"],
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # "navbar_persistent": ["search-button"],
    # "primary_sidebar_end": ["custom-template.html", "sidebar-ethical-ads.html"],
    # "article_footer_items": ["test.html", "test.html"],
    # "content_footer_items": ["test.html", "test.html"],
    "footer_start": ["copyright.html"],
    "footer_center": ["sphinx-version.html"],
    # "secondary_sidebar_items": ["page-toc.html"],  # Remove the source buttons
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
}

html_context = {
    "github_user": "cheginit",
    "github_repo": "curviriver",
    "github_version": "main",
    "doc_path": "doc",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom-icon.js"]
todo_include_todos = True

# -- favicon options ---------------------------------------------------------

# see https://sphinx-favicon.readthedocs.io for more information about the
# sphinx-favicon extension
favicons = [
    # generic icons compatible with most browsers
    "favicon-32x32.png",
    "favicon-16x16.png",
    {"rel": "shortcut icon", "sizes": "any", "href": "favicon.ico"},
]

# configuration for opengraph
description = "Generate mesh from geospatial data using Gmsh."
ogp_site_url = "https://curviriver.readthedocs.io/en/latest"
ogp_image = "https://raw.githubusercontent.com/cheginit/curviriver/main/doc/source/_static/logo.png"
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary" />',
    '<meta name="twitter:site" content="@_taher_" />',
    '<meta name="twitter:creator" content="@_taher_" />',
    f'<meta name="twitter:description" content="{description}" />',
    f'<meta name="twitter:image" content="{ogp_image}" />',
    f'<meta name="twitter:image:alt" content="{description}" />',
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
    "scikit-learn": ("https://scikit-learn.org/stable", None),
}


# based on numpy doc/source/conf.py
def linkcode_resolve(domain: str, info: dict[str, str]):
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        source = []
        lineno = None

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    fn = os.path.relpath(fn, start=Path(curviriver.__file__).parent)

    if "+" in release:
        return f"https://github.com/cheginit/curviriver/blob/main/curviriver/{fn}{linespec}"

    return f"https://github.com/cheginit/curviriver/blob/v{release}/curviriver/{fn}{linespec}"


def update_gallery(app: Sphinx):
    """Update the gallery page.

    Taken from https://github.com/pydata/xarray/blob/main/doc/conf.py
    """
    LOGGER.info("Updating gallery page...")

    gallery = yaml.safe_load(Path(app.srcdir, "gallery.yml").read_bytes())

    items = [
        f"""
         .. grid-item-card::
            :text-align: center
            :link: {item['path']}

            .. image:: {item['thumbnail']}
                :alt: {item['title']}
            +++
            {item['title']}
        """
        for item in gallery
    ]

    items_md = indent(dedent("\n".join(items)), prefix="    ")
    markdown = f"""
.. grid:: 1 2 2 2
    :gutter: 2

    {items_md}
"""
    Path(app.srcdir, "examples-gallery.txt").write_text(markdown)
    LOGGER.info("Gallery page updated.")


def setup_to_main(app: Sphinx, pagename: str, templatename: str, context, doctree) -> None:
    """Add a function that jinja can access for returning an "edit this page" link pointing to `main`."""

    def to_main(link: str) -> str:
        """Transform "edit on github" links and make sure they always point to the main branch.

        Args:
            link: the link to the github edit interface

        Returns
        -------
            the link to the tip of the main branch for the same file
        """
        links = link.split("/")
        idx = links.index("edit")
        return "/".join(links[: idx + 1]) + "/main/" + "/".join(links[idx + 2 :])

    # Disable edit button for docstring generated pages
    if "generated" in pagename:
        context["theme_use_edit_page_button"] = False
    context["to_main"] = to_main


def setup(app: Sphinx):
    """Add custom configuration to sphinx app.

    Args:
        app: the Sphinx application
    Returns:
        the 2 parallel parameters set to ``True``.
    """
    app.connect("html-page-context", setup_to_main)
    app.connect("builder-inited", update_gallery)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
