.. highlight:: bash

Installing Shark
================

.. contents:: Contents:


Quickstart for  Linux, MacOS, and other Unix-based systems
**********************************************************


To install Shark, get the sources

.. code-block:: none

      svn co https://svn.code.sf.net/p/shark-project/code/trunk/Shark
      
or a source-code release as described :doc:`here
<../downloads/downloads>`.
Then build the library:

.. code-block:: none

      cd Shark
      cmake -DOPT_ENABLE_ATLAS=ON -DOPT_ENABLE_OPENMP=ON
      make

Shark relies on `Boost <http://www.boost.org>`_ and uses `CMake
<http://www.cmake.org/>`_.
Furthermore, Shark can make use of the linear algebra library ATLAS
(if you do not want to use ATLAS, just remove the
``-DOPT_ENABLE_ATLAS=ON`` option). Under **Ubuntu**, you install all
required packages by  
``sudo apt-get install cmake cmake-curses-gui libatlas-base-dev
libboost-all-dev``.
Under **MacOS** using MacPorts, you get the required packages by
``sudo port install boost cmake atlas``. Note that Shark does
not support compilers such as GCC 4.2.1. So you
may need something such as ``sudo port install gcc48`` beforehand.
Per default, Shark is built as a static library. It may be necessary
to build a shared library (e.g., if only Boost shared libraries are
available on your system). To do so, simply add
`-DOPT_DYNAMIC_LIBRARY=ON``.

Detailed installation instructions
**********************************

The following guide explains how to install Shark in detail.
If you run into problems, please have a look at the :doc:`installation
troubleshooting page <../getting_started/troubleshooting>`
and the more general :doc:`FAQ <../faq/faq>`.


Once done installing, verify your Shark installation by running the Shark test suite (see below).
After successful installation and validation, there is a guide to the documentation available
:doc:`here <../getting_started/using_the_documentation>`.

Requirements
------------

Shark relies on `Boost <http://www.boost.org>`_  Version 1.48 or higher.
For compiling the library, you need `CMake <http://www.cmake.org/>`_
(at least version 2.8)
and a C++ that is not too old (e.g., GCC > 4.6). 


.. Installing pre-built Shark binary packages
   ------------------------------------------

    We provide pre-built binaries of Shark to be directly installed, see the :doc:`Downloads page <../downloads/downloads>`.
    We offer installers for **MS Windows 64 bit Visual Studio 2010**, **MS Windows 32 bit Visual Studio 2010**, **MS Windows
    64 bit Visual Studio 2008**, **MS Windows 32 bit Visual Studio 2008**, a **MacOS X 64 bit diskimage**, as well as a
    **Linux 32 bit Debian/Ubuntu package** and a **Linux 64 bit Debian/Ubuntu package**.


   Building Shark from source
   --------------------------

.. If your platform is not supported by the binary packages, or if you want an up-to-date version
   from the SVN repositories, you have to build Shark from source.

   At the moment, the only way to install Shark is from the source.


Download and unpack sources
---------------------------

Either download and unpack the latest official Shark source-code release from :doc:`here
<../downloads/downloads>`, or check out the current SVN version via (a ``Shark`` directory
will be created as a subfolder -- if you want the tree contents directly in the current
directory, add a space and period ``.`` to the end of the command):

.. code-block:: none

      svn co https://svn.code.sf.net/p/shark-project/code/trunk/Shark

Building Shark with Linux, MacOS, and other Unix-based systems
--------------------------------------------------------------

In the following, ``<SHARK_SRC_DIR>`` will denote the main Shark
directory, which will usually be the ``Shark/`` folder in the
directory into which you checked out the SVN snapshot or extracted
the Shark source package.

.. It should contain a ``CMakeLists.txt`` file as well as an ``include/`` and ``src/`` directory.



..    **Installation:** **1.** Configure the build using ``ccmake <SHARK_SRC_DIR>``
      (plus optional build configuration variables, see below). **2.** Call ``make``
      **3.** Call ``make test`` to verify the build **4.** Optionally call ``make
      install``. Done!

      **Time requirements:** Building plus testing can take between 15 and 120 minutes, depending on your architecture
      and build options. You can pass the ``-jN`` flag to both ``make`` and ``make test`` to use ``N`` cores and speed
      things up.

      **Space requirements:** A full installation (with debug and release libraries, examples, tests, and documentation)
      can take up around 4.5 GB. This reduces dramatically when not building the tests and examples, and/or when only
      building the release variant of Shark (but we still strongly encourage you to use the debug version with your newly
      written code).


The first step is to configure the build. In all of the below we use the
command ``ccmake`` for this. If you are not familiar with ``cmake``,
see `More details on CMake`_.

#. **Configuring the build using CMake:** Regardless if from a separate build directory
   or the main Shark folder, to enter the curses-based configuration menu of CMake, simply
   issue::

       ccmake <SHARK_SRC_DIR>

   If you have a custom/manual Boost installation, please identify your boost include and
   library directories and use instead::

       ccmake -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_INCLUDEDIR=/path/to/boost/include/ -DBOOST_LIBRARYDIR=/path/to/boost/lib/ <SHARK_SRC_DIR>

   If ``ccmake`` is installed, the above command will produce a console-style menu in
   which you can easily change those installation options you wish to alter.
   First, you may have to press ``c`` to configure the system and populate the menu.
   Next, navigate through the rows with your arrow keys, press enter to change those
   options you wish to, and then press ``c`` twice to configure the installation,
   followed by ``g`` to generate the according makefile, and ``q`` to quit ``ccmake``.


   There are several different build options (see bottom of page) you will
   encounter in the ``ccmake`` menu, and the (arguably) three most important ones are:

   * the path to the Boost library (if installed to a custom location),
   * the desired Shark build type (Debug or Release).
   * the installation path (prefix) for Shark when later (and optionally)
     calling ``make install``. By default it is ``/usr/local/`` (usually requiring ``sudo make install``).

   **If unsure, leave everything as is (perhaps put the build type to ``Debug``), and
   see** :doc:`the troubleshooting page <../getting_started/troubleshooting>` **if
   things go awry.** But, even if no options are changed, the CMake configuration
   system must still be configured in this way once.


   Besides the Boost installation path, the most important build option will
   be ``CMAKE_BUILD_TYPE``, which defines your build type (Debug or Release).

   .. admonition:: Note on Shark build type (CMAKE_BUILD_TYPE)

      Choosing the ``Debug`` build type for ``CMAKE_BUILD_TYPE`` enables a lot of type,
      size, and safety checks, but makes Shark much slower. An empty value for the
      ``CMAKE_BUILD_TYPE``, or the value ``Release`` will build the fast release version
      of the library, but without many safety checks. Whatever option you choose, you
      can repeat the build process choosing the other option and get both a release as
      well as a debug version of the library on your system. If you are new to Shark and
      want to try some *existing* examples to see how fast Shark is, please use or link
      to the release version. If you are new to Shark and want to write your own programs
      using Shark, it might make your life a lot easier if you start by linking to the
      debug version until you are sure your code is sane.

   For a detailed explanation of all other optional Shark build options
   (starting with ``OPT_``), please see the section `Shark CMake Options`_
   below.
   
   Shark supports both in-place builds (where the generated files are
   put in the Shark directory) and out-of-source builds (where the
   generated files are put in a completely different directory and the
   source tree remains unchanged). This choice is handled by the CMake
   build system (for full details, see their documentation `here
   <http://www.cmake.org/Wiki/CMake_FAQ#What_is_an_.22out-of-source.22_build.3F>`_
   ).

   In short, ``ccmake`` should be called *from the directory in which you want the build
   files to end up*. The argument to ``ccmake`` should be *the path to your Shark source
   directory* (``<SHARK_SRC_DIR>``), which contains the main CMakeLists.txt file for Shark.
   When calling ccmake from an outside directory (i.e., when building out-of-source) *after
   previous in-place builds*, you must first delete any leftover CMakeCache.txt file from
   the Shark source directory.

   In general, out-of-source builds have the advantage that you can have e.g. one folder
   for Debug and one for Release builds. In the following, the
   generic placeholder ``<SHARK_SRC_DIR>`` can either be just the current directory
   (e.g., just the dot or period "``.``") in case of in-place builds, or the path to
   your Shark main directory in case of out-of-source builds. In-place builds will not
   mess with the SVN repository, because all corresponding ``svn:ignore`` properties
   are set in the repository by default. In addition to the build tree location, you
   also have the opportunity to specify an installation directory to which the library
   will be installed upon issuing ``make install`` after compilation (see below).

   In our view, the most recommendable setup is to have two out-of-source build directories
   for one debug and one release build, but configure both of these not to build the
   documentation. The documentation can instead be conveniently built in-place
   in ``<SHARK_SRC_DIR>/doc`` by issuing ``ccmake .`` there. See the :doc:`documentation
   tutorial <../tutorials/for_developers/managing_the_documentation>` for more information.

#. Run ``make`` (or e.g. ``make -j4`` to distribute the build on 4 cores).

#. That's it: you are done and have a working Shark installation at your disposal!
   Now preferably enter ``make test`` (or ``ctest``) to verify that everything works fine.

#. When you are happy with the outcome, you can run ``make install`` to install Shark at the
   previously chosen prefix/path. If you don't install Shark this way, the library files
   will simply linger in the ``lib/`` subdirectory, which is fine. Note however, that there
   might be some additional commands carried out as part of ``make install`` (e.g., data
   files needed for the example tutorials may not get copied to the proper location),
   but this can also be done manually as needed. That is, you are fine using and
   linking to files in the build directory for most tasks - just remember to manually
   copy any data files that are reported as missing when running certain examples.
   ``locate`` may be your friend here.


Building Shark with Microsoft Windows
-------------------------------------

There are several ways to compile Shark under Windows.  If you are
using Microsoft Visual Studio, the perhaps easiest way is to download
`CMake <http://www.cmake.org/>`_. Navigate with the GUI into the Shark
directory and generate the required project files. Then open the
project with Visual Studio.  The simple procedure is explained in the
following tutorial video:

.. raw:: html

  <iframe width="560" height="345" src="http://www.youtube.com/embed/JzPNcRfVfzo" frameborder="0" allowfullscreen></iframe>

In general, Windows users are advised to add NOMINMAX to their pre-processor
defines in order to prevent windows.h from polluting the global namespace with
min and max macros.

Alternatively, you can use a Unix/GNU-like framework under Microsoft
Windows. The installation in general works as described in
`Building Shark with Linux, MacOS, and other Unix-based Systems`_,
but also see :ref:`label_for_findboost` for instructions for MinGW.



Building Shark with ATLAS backend
---------------------------------

ATLAS is an optimized linear algebra library. Using it as a backend to the shark routines can give speed-ups of factor 5-10
for big problems. Enabling ATLAS is simple. On most Unix systems, only the option "OPT_ENABLE_ATLAS" must be set to true.
If ATLAS is not placed in a standard path, you will have to tell Shark where the libraries can be found. For this, the ``ccmake``
call above must be changed to::

  ccmake -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_INCLUDEDIR=/path/to/boost/include/ -DBOOST_LIBRARYDIR=/path/to/boost/lib/ -DATLAS_ROOT:Path=/path/to/ATLAS/ -DOPT_ENABLE_ATLAS=ON <SHARK_SRC_DIR>

Enabling ATLAS support will change the auto-generated :ref:`CMake files for projects using Shark
<label_for_cmake_example_project>` to automatically use the ATLAS library as well.

See :doc:`the troubleshooting page <../getting_started/troubleshooting>` for information on how
to verify that Shark is using ATLAS.


More details on CMake
*********************

The Shark machine learning library relies on `CMake
<http://www.cmake.org/>`_ as primary build system. CMake takes a file
CMakeLists.txt as input and produces compiler- and IDE-specific
projects. The range of supported compilers and IDEs includes but is
not limited to:

* Classic Makefiles
* Microsoft Visual Studio 2005/2008/2010
* Apple XCode
* Eclipse with CDT

Using CMake
-----------

On MacOs and Linux ``ccmake`` offers a frontend for ``cmake``.  If it
is not installed on your system, either consider adding it (sometimes
in a package called ``cmake-curses-gui`` or similar), or fall back to
the wizard mode of CMake: instead of the above command, simply use the
alternative ``cmake -i``, which will query you on the command line. If
you already know well the relevant configuration options, you can also
pass them directly to ``cmake`` (without the ``-i``), as in for
example ``cmake -D CMAKE_BUILD_TYPE=Debug -D
OPT_COMPILE_DOCUMENTATION:BOOL=OFF -DBoost_NO_SYSTEM_PATHS=TRUE ...``,
etc.  Of course, you can also use the QT GUI-version of CMake
(``cmake-gui``); and of course, you can also pass options directly to
``ccmake`` in the above way.


The Shark CMake setup generates the following targets (where target means that you
can add the corresponding keyword to the ``make`` command, e.g., ``make doc`` etc.):

* Empty or default target: Builds the library and all tests.
* ``test``: Runs the unit test suite of the library.
* ``package``: Packages the library, including header files, documentation, unit tests and examples.
* ``install``: Installs the library, including header files, documentation, unit tests and examples to ${CMAKE_INSTALL_PREFIX}.

To build a specific target, see your favorite IDE's documentation. In case of Makefiles, add the target name after the make command.

The documentation has its own CMake project in the ``doc/`` subfolder.
It can be built by issuing ``make doc`` there (in-place build of the documentation),
and we recommend separating the
library build process from the documentation build process. See the :doc:`documentation
tutorial <../tutorials/for_developers/managing_the_documentation>` for more information.

.. _label_for_cmake_options:


Shark CMake Options
-------------------

The Shark CMake setup offers the following options for configuring the build process of the library:

* OPT_COMPILE_DOCUMENTATION (DEFAULT: OFF): Controls whether the documentation is built. If enabled, Doxygen and Sphinx are required.
  See the :doc:`documentation tutorial <../tutorials/for_developers/managing_the_documentation>` for more information.

* OPT_COMPILE_EXAMPLES (DEFAULT: OFF): Controls whether the examples accompanying the library are built.

* OPT_DYNAMIC_LIBRARY (DEFAULT: OFF): If enabled, Shark is built as a shared library. Otherwise, a static
  library is produced. We recommend to use the standard installation option (static) at first. When this
  works, feel free to include Shark in your LD_LIBRARY_PATH or the like to support dynamic linking. Also
  note that the space requirements do not drop that dramatically when choosing the dynamic option.

* OPT_ENABLE_NETWORKING (DEFAULT: OFF): Controls whether the networking component (HTTP server) and accompanying unit tests as well as examples are built.

* OPT_ENABLE_OPENMP (DEFAULT: OFF): Controls whether OpenMP is enabled for the build.

* OPT_INSTALL_DOCUMENTATION (DEFAULT: OFF): Controls whether the documentation is installed. Depends on OPT_COMPILE_DOCUMENTATION.

* OPT_LOG_TEST_OUTPUT (DEFAULT: OFF): Controls whether results of the unit tests are logged for further processing or report generation.

* OPT_MAKE_TESTS (DEFAULT: ON): Controls whether to build all tests.

* OPT_OFFICIAL_RELEASE (DEFAULT: OFF): Enabled only for official releases.

* OPT_ENABLE_ATLAS(DEFAULT: OFF): Let Shark use ATLAS as backend for the linear algebra routines. This is highly recommended if available!
