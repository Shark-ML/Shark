Writing Shark tutorials
=======================

So you want to write a tutorial or a similar prose documentation text.
Simply do the following:

#. Look at the header menu of this page. Decide in which section your
   document should appear. For example, the "About Shark" section.

#. Navigate to your Shark base directory. From there to ``doc/sphinx_pages/rest_sources``,
   and then into the subdirectory that corresponds to the site section
   you chose in the previous step. For example,
   ``$SHARKHOME/doc/sphinx_pages/rest_sources/about_shark``.

#. Create an empty file with a meaningful name for your new document,
   ending on ``.rst``. For example, ``new_text.rst``.

#. Write your documentation text in the reStructuredText markup language.
   See the original documentation `here <http://docutils.sourceforge.net/rst.html>`_
   and the primer included in the Sphinx documentation `there <http://sphinx.pocoo.org/rest.html>`_.
   In addition to the standard syntax, Sphinx adds `special markup constructs
   <http://sphinx.pocoo.org/markup/index.html>`_. Alternatively, navigate to
   an existing html file on the Shark homepage, press "Show page source" on
   the right and copy along.

#. Almost finished! We only need to "register" your new document
   with the Sphinx system. Usually this is done by adding the relative
   path to the ``toctree`` directive at the top of ``$SHARKHOME/doc/sphinx_pages/index.rst``.
   In our example, we'd add the line ::

      rest_sources/about_shark/new_text

   to the top of ``index.rst``. In very rare cases, the page is included locally from
   another sub-page. Note that the page will only be included in the Sphinx navigation
   tree if a toctree directive on the top of the page is used, so exploit this
   peculiarity consciously.

#. If you want to look at your page, do a ``make html`` in ``$SHARKHOME/doc/sphinx_pages``
   or a ``make doc`` in ``$SHARKHOME``. Some warnings about non-included documents are
   expected (unrelated: in rare cases a ``make clean html`` seems to be necessary). Then
   open ``$SHARKHOME/doc/sphinx_pages/build/html/rest_sources/about_shark/new_text.html``
   in your browser -- done!

.tut and .tpp files
-------------------

In order to synchronize tutorials and corresponding source code files,
we introduced a simple, but effective mechanism.  Instead of a
``.cpp`` source files and a ``.rst`` files referring to them, we
maintain ``.tpp`` source files and a ``.tut`` files, respectively.
A ``.tpp`` file is a source file with additional tags for source code
blocks, for example in ``RFTutorial.tpp``: ::

  //###begin<includes>
  #include <shark/Data/Csv.h> //importing the file
  #include <shark/Algorithms/Trainers/RFTrainer.h> //the random forest trainer
  #include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //zero one loss for evaluation
  //###end<includes>

A ``.tut`` file can refer to such blocks: ::

  ..sharkcode<Supervised/RFTutorial.tpp,includes>

The Shark build system automatically extracts these blocks and
produces stripped  ``.cpp`` source files, where the tags are removed,
and ``.rst`` files. In the latter, the references are replaced by the
corresponding source code blocks.



Formatting requirements and conventions
---------------------------------------

Ideally, the following formatting conventions apply to tutorial .rst sources:

* Do not use the h1 heading underline symbol twice, it will look ugly
  in the document.

* Each document needs at least one other heading after/besides the h1 heading.

* Ideally, use the following symbols for the different heading levels
  (almost aligning with the official Python documentation except
  skipping ``=`` to avoid confusion with tables):

  * ``h1`` headings use ``#``
  * ``h2`` headings use ``*``
  * ``h3`` headings use ``-``
  * ``h4`` headings use ``^``

  Unfortunately, this convention is followed in close to none of the
  current tutorials. But it cannot be wrong to have a convention, right?

* Do not reference doxygen class names in headings (via the ``:doxy:`` role).
  Also, do not include inline-code-markup (``like so`` -- source: ````like so```` )
  within headings. Instead, use a single ``'`` (not a `````).

* If you add new pages to the tutorials, first decide what the correct
  new order should be. Then add the new tutorial in that order both to
  the index.rst and the tutorials.rst page. In other words, all tutorials
  should appear in the same order in both files.


Linking to other pages
----------------------

Here's an internal link (here, to the tutorials): :doc:`../tutorials`
The syntax for this link was: ``:doc:`tutorials```,
also see `the sphinx documentation <http://sphinx.pocoo.org/markup/inline.html#cross-referencing-documents>`_.

Here we link to a simple plain text file which we stored in a central place (but could also reside anywhere relative
to this document, and it gets copied automagically): :download:`a plain file </static/text_files/test.txt>`. The code for
this link was ``:download:`a plain file </static/text_files/test.txt>```

You can link to the outside world like this: Shark uses `CMake <http://www.cmake.org/>`_. The markup
for this link was ```CMake <http://www.cmake.org/>`_.``

Linking to Doxygen
------------------

Here's a doxylink:

    :doxy:`choleskyDecomposition`

The syntax for this link is:
    ``:doxy:`choleskyDecomposition```

