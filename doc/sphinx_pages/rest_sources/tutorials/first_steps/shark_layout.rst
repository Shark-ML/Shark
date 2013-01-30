Shark directory layout
======================

The below diagram gives an overview over the Shark directory structure
after *having extracted the source code package of shark*.

.. note::
   The locations and/or contents of the listed folders may
   differ for the pre-compiled binary packages and installers. The contents
   of the ``include/`` folder however, are the same across all installation
   packages.

.. image:: ../images/shark_directory_structure.png
  :height: 600px
  :target: ../../../_images/shark_directory_structure.png
  :alt: Shark directory structure after installing from source.

The ``include/`` folder's structure well mirrors the Shark library's design
choices:

* The ``Core/`` Folder provides basic interfaces and functionality
  for math and other utility functions in shark.
* In ``LinAlg/``, basic vector and matrix functionality is implemented
  (e.g., by wrapping the Boost C++ uBLAS libraries).
* The ``Data/`` folder for example provides import/export routines,
  and also sets up a :doxy:`Data` class especially suited for
  machine learning tasks: subsets (e.g., for cross-validation) are
  lazy copies of the original set.
* The folders ``Fuzzy/``, ``Network/``, ``Rng/``, ``Statistics/`` all
  implement specialized functionality pertaining to Fuzzy Logic, HTTP
  protocols (for RESTful APIs), random number generation, and various
  statistical tests or distributions, respectively.

* Currently, the only algorithms implemented in the folder ``Unsupervised/``
  are Restricted Boltzmann machines, but this will be expanded in the future.

* Finally, one of the most important design aspects of Shark is the
  ``"Model"``-``"ObjectiveFunction"``-``"Optimizer"`` trias. This can be seen as roughly
  corresponding to the three remaining folders ``Models/``, ``ObjectiveFunctions/``,
  and ``Algorithms/``

.. todo::
  This mindmap will be extended in scope and provided as an interactive,
  browsable applet in the official release of Shark.
