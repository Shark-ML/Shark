.. highlight:: bash

Installing Shark
================

.. contents:: Contents:


Downloads
**********************************************************
We have two source packages available:

`Shark-4.0.0.zip <https://github.com/Shark-ML/Shark/archive/v4.0.0.zip>`_

`Shark-4.0.0.tar.gz <https://github.com/Shark-ML/Shark/archive/v4.0.0.tar.gz>`_

You can also get the latest snapshot from github:

.. code-block:: none

      git clone https://github.com/Shark-ML/Shark/


CMake Options
**********************************************************
The cmake file of Shark has several options. See the following table
for a list of options. The default values are **bold**:

======================= ===================== ===============================================
Option           	    Values                Effect
======================= ===================== ===============================================
CMAKE_BUILD_TYPE        Debug/**Release**     Builds Shark in Debug or Release mode.
                                              In debug, the linbrary is called shark_debug
CMAKE_INSTALL_PREFIX    Path **/usr/local**   Installation path for Shark
BUILD_DOCUMENTATION     ON/**OFF**            Builds the documentation, see "build documentation"
BUILD_EXAMPLES          **ON**/OFF            Builds the examples
BUILD_TESTING           **ON**/OFF            Builds the tests
BUILD_SHARED_LIBS      	ON/**OFF**            Builds Shark as shared library 
Boost_USE_STATIC_LIBS   ON/**OFF**            Searches and uses the static boost libraries,
                                              Be aware, that linking static Boost 
                                              libraries to a dynamic Shark
					      can result in problems during build!
BOOST_ROOT              Path                  Path to boost, if it is not installed in a default path.
ENABLE_CBLAS            **ON**/OFF            Searches for a linear algebra library on the system
CBLAS_LIBRARY_PATH      Path **/usr/include** Sets the path to the cblas include directory.
CBLAS_INCLUDE_PATH      Path **/usr/lib64/**  Sets the path to the cblas library directory.
ENABLE_OPENMP           **ON**/OFF            Enables OpenMP support if supported by the platform
ENABLE_SIMD	        **ON**/OFF            Enables SIMD in linear algebra
ENABLE_OPENCL           ON/**OFF**            Enables OpenCL support if boost.compute is available. EXPERIMENTAL!
ENABLE_CLBLAST          ON/**OFF**            Uses CLBLAST as OpenCL linear algebra backend. EXPERIMENTAL!
======================= ===================== ===============================================

To change options, either use one of the cmake guis (e.g., ccmake) or add the options to the cmake call.
Choosing another path to boost and disabling OpenMP would look like::

	cmake "-DBOOST_ROOT=/path/to/boost" "-DENABLE_OPENMP=OFF" ../

An introduction on how to run cmake can be found `here <http://cmake.org/runningcmake/>`__.


Linux, MacOS, and other Unix-based systems
**********************************************************

Packages
---------------------------------------------

We are supporting packages for the following platforms:

* Arch Linux
	- current development version downloadable from AUR as packages ``shark-ml-atlas-git`` and ``shark-ml-git``
	  for a shark version with and without ATLAS.
* More to come!

Installation
---------------------------------------------

On may systems, installing shark is as easy as:

.. code-block:: none

	mkdir Shark/build/
	cd Shark/build
	cmake ../
	make
	
Dependencies
---------------------------------------------

Shark relies on `Boost <http://www.boost.org>`_ and uses `CMake
<http://www.cmake.org/>`__.
Furthermore, Shark can make use of different linear algebra libraries.
On MacOsX, Accelerate is used by default. On Linux and Windows, ATLAS, CBLAS and openblas
is used if available.
Under **Ubuntu**, you install all required packages by::
	
	sudo apt-get install cmake cmake-curses-gui libatlas-base-dev libboost-all-dev
	
Under **MacOS** using MacPorts, you get the required packages by::

	sudo port install boost cmake

Building the documentation
----------------------------------------------------

This section will tell you how to **build the documentation on your computer**, and
also how to first install the tools needed for it. Besides Doxygen, we rely on two
relevant Python modules, namely Sphinx and Doxylink (aka sphinxcontrib-doxylink).
Since this tutorial page is created by Sphinx, you will most likely read it off a
webserver or as part of a Shark package including the generated documentation pages.
After having built the documentation yourself, you will be able to read it from your
local folder, too.

#. Make sure Doxygen, Graphviz, Python and Sphinx are properly installed on your system,
#. run cmake in your build directory and set ``BUILD_DOCUMENTATION`` to ``ON``
#. run ``make doc``


Windows and Visual Studio
**********************************************************

Dependencies
---------------------------------

First start by download and installing:

* `CMake <https://cmake.org/download/>`__
* The most recent `boost binaries <http://sourceforge.net/projects/boost/files/boost-binaries/>`__

Setting Up Boost
----------------------------------


For simplicity, we assume that you installed boost in ``C:\locale\boost_1.59\``.
The boost libraries will be located in a subfolder whose name depends on your compiler (e.g.
``C:\locale\boost_1.59\lib64-msvc-12.0``). Next, you need to tell Windows
about the location of the binaries as otherwise compilation will work, but running the compiled binaries
will not be possible. 

For this you go to
My Computer>Properties>Advanced>Environment Variables>System Variables>Path>Edit>Variable Value
and add ``;C:\locale\boost_1.59\lib64-msvc-12.0`` (or your equivalent path) to the end.

Installing Shark
-----------------------

* Download the sources and unpack them
* Open the CMake GUI
* Next to "Where is the source code" set the path to the unpacked Shark location
* Next to "Where to build the directory" set the path to where you want the Visual Studio project files to be
* Click on "Add Entry"
* Add an Entry BOOST_ROOT of type PATH and set it to your boost intall directory (e.g. ``C:\locale\boost_1.59``)
* Add an Entry BOOST_LIBRARYDIR of type PATH and set it to your boost library directory (e.g. ``C:\locale\boost_1.59\lib64-msvc-12.0``)
* Set the right Visual Studio compiler and click on Configure (possibly twice) and then on generate



Troubleshooting the installation procedure
==========================================


Getting Shark to find Boost
**********************************************************


If you obtained Boost through your package manager, Shark should (in theory) be able to find
Boost automatically. If Boost is not found automatically (e.g., after issuing ``make`` the
compilation aborts with long error messages involving the word "boost"; or the Makefile
cannot even be generated because the Shark CMake configuration does not complete), also
possibly because you compiled/installed Boost yourself to some custom location, you can
try the following:

  * The most proven approach is to invoke CMake with the additional options
    ``-DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_INCLUDEDIR=/path/to/boost/include/ -DBOOST_LIBRARYDIR=/path/to/boost/lib/``.
    Here, ``DBOOST_LIBRARYDIR`` could for example be ``/opt/local/include`` or ``/home/user/mine/boost_153/lib``, etc.
    This should work in 90% of all cases.
