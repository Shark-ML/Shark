

The Shark documentation system
==============================


.. contents:: Contents:


Shark uses the well-established `Doxygen <http://www.doxygen.org>`_ documentation
system in combination with `Sphinx <http://sphinx.pocoo.org/>`_, another documentation
system originating from the Python world. Doxygen is used to extract documentation
from the C++ sources, generate inheritance diagrams, etc. Sphinx is used to generate
the entire surrounding documentation.


Building the documentation
++++++++++++++++++++++++++


Setting up Sphinx and Doxylink
------------------------------

This section will tell you how to **build the documentation on your computer**, and
also how to first install the tools needed for it. We rely on a number of different
Python modules, namely Sphinx and Doxylink, which in turn rely on other packages again,
e.g., Docutils, Jinja2, Pygments, and Pyparsing. Since this tutorial page is created by
Sphinx, you will most likely read it off a webserver or as part of a Shark package
including the generated documentation pages.

.. note:: These instructions are currently tested only for Mac OS X and Linux users. Please
    contact us to mutually find out how this will work on Windows, or share your successful
    commands with us.

#. We assume you have Doxygen installed. Note that in our current setup, Doxygen is also configured
   to use the Graphviz (http://www.graphviz.org/) libraries for rendering inheritence graphs,
   among others. This implies that we either assume Graphviz installed, or that you have to
   manually edit your Shark.dox file in the main documentation folder, setting ``HAVE_DOT = NO``.

#. First, please get an overview for yourself over what Python versions you have installed on your system.
   In an ideal world, you would have one single Python version (either 2 or 3). We have however witnessed
   successful installation of all needed dependencies on systems where python-2.5, -2.6, -2.7, and -3.2
   were installed alongside each other, but this can make things (much) harder, e.g., execute Sphinx with
   a different Python version than for which its dependencies had been installed, etc.

#. In the shark home directory, go to the sub-directory ``contrib/doxylink/sources``

#. There, try to issue the command ``PYTHONPATH=../build/lib/python python setup.py install
   --home=../build --record list_of_installed_files.txt`` (where the last argument ``record``
   could be omitted, but may provide useful debugging information and/or help during
   uninstallation/cleanup).

   .. note::

      On a Mac, instead try ``PYTHONPATH=../build/lib/python: python setup.py install
      --home=../build --record list_of_installed_files.txt`` (with an added colon after
      the target Python path as the only difference).

   .. note::

      On some systems, the ``python`` command might invoke python 3, and python 2 be invoked
      by python2. In such a case, replace the above command by the appropriate version, e.g.,
      ``PYTHONPATH=../build/lib/python python2 setup.py install --home=../build --record
      list_of_installed_files.txt`` .

   This will on some platforms (especially those with not-too-often used Python installations)
   result in an error ``Traceback (most recent call last): File "setup.py", line 3, in <module>
   from setuptools import setup, find_packages ImportError: No module named setuptools`` . In
   such a case, please obtain setuptools in the way suited for your distribution or platform
   (e.g., apt-get, etc.). When you then successfully installed setuptools, again try to issue
   ``PYTHONPATH=../build/lib/python python setup.py install --home=../build --record
   list_of_installed_files.txt`` , or, on a Mac potentially ``PYTHONPATH=../build/lib/python:
   python setup.py install --home=../build --record list_of_installed_files.txt`` .

   Another problem that can occur is that your locally installed Python dependencies cannot
   be resolved. In such a case, consider adding the respective directory to your PYTHONPATH
   variable. This can for example be achieved by adding ``export
   PYTHONPATH=$PYTHONPATH:/path/to/your/Shark3/contrib/doxylink/build/lib/python/``
   to your .bashrc . Note that this setting of the PYTHONPATH would be for the everyday use
   of Sphinx and others, whereas the note further above just refers to temporarily setting
   the PYTHONPATH when installing the dependencies.

#. See what happens when you issue ``make doc`` in the ``doc/`` subdirectory of the main Shark
   directory (you will possibly have to use ``ccmake .`` or ``cmake -i`` again to configure the
   installation first). If Doxygen is installed and working, the interesting part will now be
   whether all Python components needed by Sphinx and Doxylink are working. This should in
   theory be the case. There have however cases been reported where the executable provided by
   Sphinx (``sphinx-build``) was either not recognized, not included in the system path, or
   itself did not find the underlying Python Sphinx module. In such a case, consider

   * installing Sphinx through your distribution again.
   * examining, relative to the $SHARKHOME directory, both ``contrib/doxylink/build/bin`` as
     well as ``contrib/doxylink/build/lib/python/Sphinx-1.0.7-py2.7.egg/sphinx`` and maybe
     adding a manual alias from ``sphinx-build`` to the correct executable
     ``contrib/doxylink/build/bin/sphinx-build``, or including the former directory to your
     Python and/or system path.
   * it can also help to issue ``make doc`` with a user-controlled pythonpath, e.g. as in ::

         PYTHONPATH=/usr/lib/python2.7/site-packages/:/path/to/your/Shark3/contrib/doxylink/build/lib/python make doc

   * in extremely lost causes, you might want to look into the `virtualenv <http://www.virtualenv.org>`_
     tool for managing different Python installations.

#. You know that you are done when ``make doc`` exits with ``build succeeded. Built target doc``,
   and when you can successfully view the page ``$SHARKHOME/doc/index.html``.

In general, if you run into troubles, you should try to make sure all dependencies are installed
and accessible. The most relevant dependencies are Sphinx and Doxylink, which in turn rely on a
number of tools, e.g., Docutils, Jinja2, Pygments, and Pyparsing, which however should be taken
care of automatically by setuptools. You can check upon the added installations by examining the
directory tree under ``contrib/doxylink/`` with e.g. ``ls -R``. Then it usually boils down to
either installing what's missing or making the path known in the correct manner. Good luck!


MathJax
+++++++

To render Latex equations on both Doxygen and Sphinx pages, we rely on MathJax
instead of static images locally generated by Latex. MathJax was chosen because
vertical alignment of equations rendered by a local latex installation is a pain.
To make a user's browser MathJax-capable, we follow a two-fold approach: first,
as a default location for the ``MathJax.js`` file, we use an URL to the MathJax
content delivery network -- that is, all users simply load the default online
version of MathJax. However, users with a local installation of the documentation
should also be able to read the docs when working offline. Thus, both the Doxygen
``header.html`` template and the Sphinx ``layout.html`` template include a JS
snippet that tells the docs to fallback to a local MathJax installation. This
_must_ be located in ``$Sharkhome/contrib/mathjax`` (to be precise, MathJax must
be installed such that ``MathJax.js`` lives in that folder). The reason we do
not distribute MathJax with Shark is that notably the Firefox browser does not
support the MathJax web fonts (because of a strict same-origin policy even for
local sites), thus needs to fallback to image fonts, and these image fonts are
140MB in size. Since we did not want to distribute these along with Shark, any
users that want offline equation support for the docs are advised to install
MathJax themselves to the abovementioned location. Thanks.

..
   comment in once the firefox same-origin thing is fixed:
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

Below is useful information related to Python and Doxylink updates with respect
to necessary user intervention.


When your Python version is upgraded
------------------------------------

In case some of your Python dependencies for the Doxygen-Sphinx-Doxylink have
been pulled in when installing Doxylink, you may need to re-issue the command
``PYTHONPATH=../build/lib/python python setup.py install --home=../build --record
list_of_installed_files.txt`` in ``contrib/doxylink/sources`` after every Python
upgrade that took place through your distribution.

Note for Python3.3 users
------------------------

Unfortunately, at the time of this writing, docutils does not support Python3.3,
see `this bug report and patch <http://sourceforge.net/tracker/?func=detail&aid=3541369&group_id=38414&atid=422030>`_ .
Python 3.3 users should thus apply the patch from the link above to their docutils
source tree. If installed by hand according to the above instructions, this will be
located in ``contrib/doxylink/build/lib/python/docutils-0.9.1-py3.3.egg/docutils``.

When a new Doxylink version comes out
-------------------------------------

When a new Doxylink version is released, there are two aspects:

#. First, the users' aspect: the new version may or may not add functionality
   which is needed by the current SVN version of Shark. If it does (and also
   in general), it is  advisable for anyone building the Shark docs to upgrade
   their local Doxylink installation. This can simply be done by following the
   above same instructions as for installing Doxylink in the first place. In
   particular, in ``contrib/doxylink/sources`` issueing
   ``PYTHONPATH=../build/lib/python python setup.py install
   --home=../build --record list_of_installed_files.txt`` should be all you
   need to do.

#. Second, the shark developers' aspect: since Shark for convenience provides
   the Doxylink sources, someone needs to update the files in
   ``contrib/doxylink/sources/`` such that they reflect the update. However,
   we do not include all files, for example omit the ``test/`` and ``doc/``
   directories. Thus, only the important/present files/changes should be
   propagated manually. Using meld on the shark ``contrib/doxylink`` folder
   and a newly checked out birkendfeld repository in combination with svn
   list commands will usually help do the trick quickly.






For developers
++++++++++++++

The information below is only relevant for developers
who write tutorials and/or wish to alter the appearence
of the overall documentation system.


Writing tutorials
-----------------

In general, simply see the documentation for Sphinx and reStructuredText-files
to understand the syntax which tutorials have to adhere to. In general, it is
easiest to start with an existing file and simply copy its style. Below we
point out notable caveats when working with Sphinx and rst-files:

* Do not use the main heading type, that is, the underline type
  for the h1-heading twice, because this will look ugly in the
  document. In other words, whichever symbol you chose to underline
  the main page heading should not get used a second time from then
  on.

  To promote homogeneity, we advise that the following conventions are
  being followed for heading levels, almost aligning with that for the
  official Python documentation (except skipping the ``=`` to avoid
  confusion with tables):

  * ``h1`` headings use ``#``
  * ``h2`` headings use ``*``
  * ``h3`` headings use ``-``
  * ``h4`` headings use ``^``

  Unfortunately, this convention is followed in close to none of the
  current tutorials, but it cannot be wrong to have a convention, right?

* Do not reference doxygen class names in headings (via the ``:doxy:`` role).
  Also, do not include inline-code-markup (``like so`` -- source: ````like so```` )
  within headings. Instead, use a single ``'`` (not a `````).

* If you add new pages to the tutorials, first decide
  what the correct new order should be. Then add the new
  tutorial according to this same order both to the index.rst
  as well as to the tutorials.rst page. In other words, all
  tutorials should appear in the same order in both files.


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

Sphinx and Doxygen html header injection
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

The Shark homepage injects a css menu header (based on
`the mollio templates <http://www.mollio.org>`_) into
the documentation generated by both Sphinx and by Doxygen.
If you change the menu layout, remember to change it
**synchronously** in two locations:
``${SHARKHOME}/doc/sphinx_pages/templates/layout.html``
for all Sphinx pages and
``${SHARKHOME}/doc/doxygen_pages/templates/header.html``
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
page `here <http://pypi.python.org/pypi/sphinxcontrib-doxylink>`__ .


In ``${SHARKHOME}/doc/sphinx_pages/conf.py`` the variable ``doxylink`` defines additional
Sphinx markup roles and links them to a Doxygen tag file. At the moment, the only role
is ``:doxy:``, and it links to the global overall tag file for the entire Shark library.



