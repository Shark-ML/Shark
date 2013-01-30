Credits and Copyright
=====================

.. _label_for_citing_shark:

Citing Shark
------------

We kindly ask you to cite Shark in academic work as:

.. container:: cibox

	Christian Igel, Verena Heidrich-Meisner, and Tobias Glasmachers.
	`Shark <http://jmlr.csail.mit.edu/papers/v9/igel08a.html>`_.
	Journal of Machine Learning Research 9, pp. 993-996, 2008

The article's bibtex entry reads: ::

	@Article{shark08,
		author = {Christian Igel and Verena Heidrich-Meisner and Tobias Glasmachers},
		title = {Shark},
		journal = {Journal of Machine Learning Research},
		year = {2008},
		volume = {9},
		pages = {993-996}
	}

License
-------

The Shark library is made available under the `GNU General Public
License <http://www.gnu.org/licenses/gpl.html>`_ as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

Hosting institutions
--------------------

The Shark machine learning library is jointly maintained by researchers from

* `Institut für Neuroinformatik (INI), Ruhr-Universität Bochum, Germany <http://www.ini.rub.de/>`_
* `Department of Computer Science (DIKU), University of Copenhagen, Denmark <http://www.diku.dk/>`_



Present and previous contributors
----------------------------------

Shark is currently developed and maintained by the following people (in alphabetical order)::

	Asja Fischer, Tobias Glasmachers, Kasper Nybo Hansen,
	Christian Igel, Oswin Krause, Bill Li, Trinh Xuan Tuan,
	Matthias Tuma, Thomas Voss


The following people have been contributors to past versions of
Shark (in alphabetical order)::

	Rüdiger Alberts, Lukas Arnold, Thomas Bücher, Eduard Diner,
	Verena Heidrich-Meisner, Michael Hüsken, Martin Kreutz,
	Marc Nunkesser, Tatsuya Okabe, Stefan Roth, Pavel Saviankou,
	Bernhard Sendhoff, Peter Stagge, Thorsten Suttorp,
	Marc Toussaint, Björn Weghenkel, Stefan Wiegand, Aimin Zhou

Past and present supporters
---------------------------

The development of the Shark library is or has been supported by the following institutions or companies:

* `Institut für Neuroinformatik (INI), Ruhr-Universität Bochum, Germany <http://www.ini.rub.de/>`_
* `Department of Computer Science (DIKU), University of Copenhagen, Denmark <http://www.diku.dk/>`_
* `nisys GmbH <http://www.nisys.de/>`_
* `Honda Research Institute Europe (HRI-EU) <http://world.honda.com/group/HondaResearchInstituteEurope/>`_

Third-party software
--------------------

Shark makes use of, includes, links to, or relies on third-party software
listed below. These are provided by their respective vendors under licenses
compatible with the GPLv3 of Shark.
When modifying, building on, or re-distributing Shark, it is your
responsibility to also ensure that the rights of the respective third-party
copyright holders are respected in addition to meeting the requirements of
the Shark GPLv3 license.


Source code
+++++++++++

* Shark uses and links against parts of the `Boost software libraries <http://www.boost.org>`_.
  The Boost libraries are not shipped with Shark and have to be installed
  independently. The Boost libraries are available under the `Boost software
  license <http://www.boost.org/LICENSE_1_0.txt>`_, which is compatible with
  the GPL.

* Shark relies on `CMake <http://www.cmake.org/>`_ as its build system.
  CMake is not included with Shark, but needs to be installed externally.
  However, CMake configuration files are provided with Shark for convenience.
  CMake is available  under the 3-clause (new/modified) `BSD license
  <http://www.opensource.org/licenses/bsd-license.php>`_.

* The Shark machine learning library ships with the simplex solver from
  GLPK, the GNU Linear Programming Kit (http://www.gnu.org/software/glpk/),
  version 4.45. GLPK is NOT part of the Shark machine learning library, but
  its simplex solver is used by Shark and thus distributed within the
  package for convenience. GLPK code is directly linked into the Shark
  library. A slightly modified subset of GLPK can be found in the header
  file GLPK.h and the source file GLPK.cpp. The GLPK is available under the
  `GNU General Public License <http://www.gnu.org/licenses/gpl.html>`_.


Documentation
+++++++++++++
* Shark uses the `Doxygen documentation system <http://www.doxygen.org>`_.
  Doxygen is not included with Shark, but Doxygen configuration files are
  provided for convenience. Doxygen is available under the
  `GNU General Public License <http://www.gnu.org/licenses/gpl.html>`_.
* Shark also uses the `Sphinx documentation system <http://sphinx.pocoo.org/>`_.
  Sphinx is not included with Shark, but Sphinx configuration files are
  provided for convenience. Sphinx is available under the
  2-clause (simplified/Free) `BSD license
  <http://www.opensource.org/licenses/bsd-license.php>`_.
* The Shark documentation links between Sphinx and Doxygen using
  `Doxylink <http://pypi.python.org/pypi/sphinxcontrib-doxylink>`_ written
  by Matt Williams and released under a 2-clause (simplified/Free) `BSD license
  <http://www.opensource.org/licenses/bsd-license.php>`_. Doxylink is not included
  in Shark, but the Shark documentation relies on Doxylink's functionality.
* The website header is derived from the `Mollio <http://mollio.org/>`_ set
  of html/css templates. Mollio is triple-licensed under the
  `CC BY <http://creativecommons.org/licenses/by/2.5/>`_, the
  `GPLv2 <http://www.gnu.org/licenses/gpl-2.0.html>`_, and the
  `CPL <http://www.opensource.org/licenses/cpl1.0.php>`_.
* The page icon in the local table of contents is one of Nicolas Gallagher's
  `pure CSS GUI icons <http://nicolasgallagher.com/pure-css-gui-icons/>`_.
  Nicolas Gallagher's work is dual-licensed under an
  `MIT <http://www.opensource.org/licenses/mit-license.php>`_ and
  `GNU GPLv2 <http://www.gnu.org/licenses/gpl-2.0.html>`_
  license.
