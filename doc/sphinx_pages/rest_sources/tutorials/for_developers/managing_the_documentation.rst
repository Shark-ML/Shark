The Shark Documentation System
==============================


.. contents:: Contents:


Shark uses the well-established `Doxygen <http://www.doxygen.org>`_ C++ documentation
system in combination with `Sphinx <http://sphinx.pocoo.org/>`_, another popular
documentation system originating from the Python world. Doxygen is used to extract
documentation from the C++ sources, generate inheritance diagrams, etc. To generate
the entire surrounding documentation including these tutorials, Sphinx is used.


Building the documentation
++++++++++++++++++++++++++


Setting up Sphinx and Doxylink
------------------------------

This section will tell you how to **build the documentation on your computer**, and
also how to first install the tools needed for it. Besides Doxygen, we rely on two
relevant Python modules, namely Sphinx and Doxylink (aka sphinxcontrib-doxylink).
Since this tutorial page is created by Sphinx, you will most likely read it off a
webserver or as part of a Shark package including the generated documentation pages.
After having built the documentation yourself, you will be able to read it from your
local folder, too.

.. admonition:: Quick-Start Installation Summary:

    **Installation:** **1.** Make sure Doxygen, Graphviz, Python, Sphinx, and
    Doxylink (aka sphinxcontrib-doxylink) are properly installed on your system,
    with PATH properly set for ``sphinx-build`` and PYTHONPATH for Doxylink. **2.**
    Call ``ccmake -DOPT_COMPILE_DOCUMENTATION=ON <SHARK_SRC_DIR>`` from where you
    want the documentation to be built (e.g., in- or out-of-source).

    or ``cmake .`` in ``<SHARK_SRC_DIR>/doc``, or call
    ``ccmake -DOPT_COMPILE_DOCUMENTATION=ON .`` or ``cmake -DOPT_COMPILE_DOCUMENTATION=ON .``
    in ``<SHARK_SRC_DIR>``

.. note:: These instructions are currently only tested on Mac OS X and Linux. Please
    contact us to mutually find out how this will work on Windows, or share your success
    stories with us.

#. We assume you have Doxygen installed. Note that in our current setup, Doxygen is also configured
   to use the `Graphviz <http://www.graphviz.org/>`_ libraries for rendering inheritence graphs,
   among others. This implies that we either assume Graphviz installed, or that you have to
   manually edit your Shark.dox(.in) file in the main documentation folder, setting ``HAVE_DOT = NO``.

#. First, please get an overview for yourself over what Python versions you have installed
   on your system. In the simplest of worlds, you would have one single Python version (either
   2 or 3). In all other cases, keep in the back of your head that you will want to install
   the necessary Python packages for the same Python versions which you will later use to call
   Sphinx when building the tutorials.

#. Get an instance of `Doxylink (a.k.a. sphinxcontrib-doxylink)
   <http://pypi.python.org/pypi/sphinxcontrib-doxylink>`_ working on your system. Since
   Doxylink depends on Sphinx, getting Doxylink will also pull in Sphinx and its dependencies.
   So, for example when using ``pip``, a command like ``pip install sphinxcontrib-doxylink``
   should get you good to go. You can specify a custom target directory via e.g.
   ``pip install --install-option="--prefix=/home/user/path/to/your/pip_python_packages"
   sphinxcontrib_doxylink``.

#. As a precautionary measure, review your ``PYTHONPATH`` and ``PATH`` environment variables:
   depending on how and where you installed Doxylink and the other dependencies, you may
   have to add the locations of the corresponding python packages (e.g. for the example above,
   ``/home/user/path/to/your/pip_python_packages/lib/python3.3/site-packages``)
   to your ``PYTHONPATH``
   and the locations of the ``sphinx-build`` executable to your ``PATH``.

#. Next decide where you want to build your documentation to, and issue
   ``cmake -DOPT_COMPILE_DOCUMENTATION=ON <SHARK_SRC_DIR>`` or ``ccmake
   -DOPT_COMPILE_DOCUMENTATION=ON <SHARK_SRC_DIR>`` from there. The same comments
   on in- and out-of-source builds apply as in the :doc:`installation instructions
   <../../getting_started/installation>`. As stated there, we suggest to not build
   the documentation with any debug or release builds of the library, but to build
   it independently either in- or out-of-source.

#. See what happens when you issue ``make doc`` in the directory you chose before
   (i.e., from where you issued the cmake commands from the previous point). If
   Doxygen is installed and working, the interesting part will now be whether all
   Python components needed by Sphinx and Doxylink are working. This should in
   theory be the case. If not, try one of the following:

   * go back one step and examine your ``PYTHONPATH`` and ``PATH`` environment
     variables again. For debugging purposes, it can also help to issue ``make doc``
     with a user-controlled pythonpath, e.g. as in ::

         PYTHONPATH=/path/where/pip/puts/your/python2.7/site-packages/ make doc

     In extreme cases, you could add an alias from "sphinx-build" to the correct location
     of your sphinx-build executable on your system.

   * you can manually edit ``<SHARK_SRC_DIR>/doc/sphinx_pages/conf.py`` to print out information,
     like the path that Sphinx is seeing, etc.

   * in very lost causes with multiple python versions and dependencies, you might
     want to look into the `virtualenv <http://www.virtualenv.org>`_
     tool for managing different Python installations.

#. You know that you are done when ``make doc`` exits with ``build succeeded. Built target doc``,
   and when you can successfully view the page ``<SHARK_SRC_DIR>/doc/index.html``.

.. admonition:: Note on just building the Sphinx part of the documentation

   The subfolder ``doc/sphinx_pages`` consists of an additional Makefile steering
   the Sphinx document generation process only. Here, you can issue ``make clean``
   and ``make html``



.. todo::

    explain the new tut2rst and tpp to cpp workings. (iinm, the tpp to cpp happens
    when doing ccmake . in the shark_source dir.)


MathJax
+++++++

To render Latex equations on both Doxygen and Sphinx pages, we rely on MathJax
instead of static images locally generated by Latex. MathJax was chosen because
vertical alignment of equations rendered by a local latex installation is a pain.
To make a user's browser MathJax-capable, we follow a two-fold approach: first,
as a default location for the ``MathJax.js`` file, we use an URL to the MathJax
content delivery network -- that is, all users simply load the default online
version of MathJax. However, users with a local installation of the documentation
may also want to read the docs when working offline. Thus, both the Doxygen
``header.html`` template and the Sphinx ``layout.html`` template include a JS
snippet that tells the docs to fallback to a local MathJax installation. **However**,
this local MathJax installation is not provided together with Shark and must be
inserted by the user to a specific location. MathJax
_must_ be located in ``<SHARK_SRC_DIR>/contrib/mathjax`` (to be precise, MathJax must
be installed such that ``MathJax.js`` lives in that folder). The reason we do
not distribute MathJax with Shark is that notably the popular Firefox browser does
not support the MathJax web fonts (because of a strict same-origin policy even for
local sites), thus needs to fallback to image fonts, and these image fonts are
140MB in size. Since we did not want to distribute these along with Shark, any
users that want offline equation support for the docs are advised to install
MathJax themselves to the abovementioned location.

..
   old, from when we distributed mathjax as well / for documentation of how to get a small MJ inst.
   A local version of MathJax was chosen because otherwise, seeing equations in the
   docs would rely on an internet connection. Since a standard MathJax
   installation is huge (150MB or so), we crop some of its functionality:
   the folders ``docs``, ``test``, and ``unpacked`` are deleted. Then, the
   biggest culprit, ``fonts/HTML-CSS/TeX/png``, is also removed. Finally,
   all config files in the ``config`` folder except the standard
   ``TeX-AMS-MML_HTMLorMML.js`` are deleted, and the standard file is
   renamed to avoid confusion. Also, the option ``imageFont:null`` is added
   in order to stop complaints about missing png fonts. As a result of deleting
   the png fonts, old IE users will miss out, but we take this risk for the
   sake of saving 140 MB of space.



Maintenance and update issues
+++++++++++++++++++++++++++++

Below comes information related to Python, Doxylink, and their updates.


Note for Python3.3 users
------------------------

Unfortunately, at the time of this writing, docutils does not support Python3.3,
see `this bug report and patch <http://sourceforge.net/tracker/?func=detail&aid=3541369&group_id=38414&atid=422030>`_ .
Python 3.3 users should thus apply the patch from the link above to their docutils
source tree.



For developers
++++++++++++++

The information below is only relevant for developers
who write tutorials and/or wish to alter the appearence
of the overall documentation system.


Writing tutorials
-----------------

See the dedicated tutorial on :doc:`writing_tutorials`.


Changing the documentation's appearance
---------------------------------------


Important files and paths
&&&&&&&&&&&&&&&&&&&&&&&&&

The general appearance of the Sphinx pages is governed by a
"Sphinx theme" and potentially additional CSS stylings and
other files needed for styling. Both are located in
``doc/sphinx_pages/ini_custom_themes``. The file ``theme.conf``
is the Sphinx theme and derived from the ``sphinxdoc`` theme
with minor adaptations. The file static/mt_sphinx_deriv.css_t
is the stylefile, which still holds some Sphinx directives
which will be replaced to create the truly static
``mt_sphinx_deriv.css`` for the html pages.

In ``doc/sphinx_pages/templates`` you can find the Sphinx/Jinja2
templates which are used to steer the page layout in addition
to the theme-based styling.

The folder ``doc/sphinx_pages/static`` holds additional images,
icons, and sprites needed by the templates.

For doxygen, the header and footer layout is goverened by the
files in ``doc/doxygen_pages/templates``, and the CSS styling
in ``doc/doxygen_pages/css``.


Assorted hints
&&&&&&&&&&&&&&

Miscellanea:

* When linking to boost documentation pages for which you have
  copied the link from your browser, always replace the current
  version number (e.g., 1_54_0) in the link with "release". This
  will always redirect to the most recent version.

Sphinx and Doxygen html header injection
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

The Shark homepage injects a css menu header (based on
`the mollio templates <http://www.mollio.org>`_) into
the documentation generated by both Sphinx and by Doxygen.
If you change the menu layout, remember to change it
**synchronously** in two locations:
``<SHARK_SRC_DIR>/doc/sphinx_pages/templates/layout.html``
for all Sphinx pages and
``<SHARK_SRC_DIR>/doc/doxygen_pages/templates/header.html``
for all Doxygen pages.


Sphinx and Doxygen connection
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Doxygen creates documentation for the classes, namespaces, functions, variables, etc.,
used in Shark. For the surrounding tutorials (like this page), we use the Sphinx
documentation system, which was originally conceived for the Python world. In order
to be able to automatically reference pages in the doxygen documentation from within
the Sphinx tutorial pages, we use the excellent and highly recommended Sphinx-Doxygen
bridge "Doxylink" by Matt Williams. You can find the documentation for Doxylink
`here <http://packages.python.org/sphinxcontrib-doxylink/>`_ and its PyPI package
page `there <http://pypi.python.org/pypi/sphinxcontrib-doxylink>`__ .


In ``<SHARK_SRC_DIR>/doc/sphinx_pages/conf.py`` the variable ``doxylink`` defines additional
Sphinx markup roles and links them to a Doxygen tag file. At the moment, the only role
is ``:doxy:``, and it links to the global overall tag file for the entire Shark library.



