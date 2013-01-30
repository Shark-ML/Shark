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
      
   to the top of ``index.rst``. In rarer cases, the page is included locally from
   another sub-page. Note that the page will only be included in the Sphinx navigation
   tree if a toctree directive on the top of the page is used, so exploit this 
   peculiarity consciously.
   
#. If you want to look at your page, do a ``make html`` in ``$SHARKHOME/doc/sphinx_pages``
   or a ``make doc`` in ``$SHARKHOME``. Some warnings about non-included documents are
   expected (unrelated: in rare cases a ``make clean html`` seems to be necessary). Then open 
   ``$SHARKHOME/doc/sphinx_pages/build/html/rest_sources/about_shark/new_text.html``
   in your browser -- done!
   

Linking to other pages
----------------------

Here's an internal link (by chance, to the tutorials): :doc:`../tutorials`
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

