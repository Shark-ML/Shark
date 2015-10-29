.. highlight:: bash

Installing Shark
================

.. contents:: Contents:


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

To install Shark, get the sources from our :doc:`Downloads page <../downloads/downloads>`
	
Then build the library::

	mkdir Shark/build/
	cd Shark/build
	cmake ../
	make
	
Dependencies
---------------------------------------------

Shark relies on `Boost <http://www.boost.org>`_ and uses `CMake
<http://www.cmake.org/>`__.
Furthermore, Shark can make use of different linear algebra libraries.
On MacOsX, Accelerate is used by default. On Linux and Windows, ATLAS
is used if available.
Under **Ubuntu**, you install all required packages by::
	
	sudo apt-get install cmake cmake-curses-gui libatlas-base-dev libboost-all-dev
	
Under **MacOS** using MacPorts, you get the required packages by::

	sudo port install boost cmake


CMake Options
-------------------------------------------------------------
The cmake file of Shark has several options. See the following table
for a list of options. The default values are **bold**:

======================= ===================== ===============================================
Option           	    Values                Effect
======================= ===================== ===============================================
BUILD_SHARED_LIBS      	ON/**OFF**            Builds Shark as shared library 
Boost_USE_STATIC_LIBS   ON/**OFF**            Searches and uses the static boost libraries,
                                              Be aware, that linking static Boost 
                                              libraries to a dynamic Shark
					      can result in problems during build!
BOOST_ROOT              Path                  Path to boost, if it is not installed in a default
                                              path.
ENABLE_ATLAS            **ON**/OFF            Enables ATLAS as linear algebra library if found;
                                              ignored on MacOSX as Accelerate is favourable
ATLAS_ROOT              Path                  Additional path to search for an ATLAS
                                              installation, if ATLAS is not installed in a
                                              system path
ATLAS_FULL_LAPACK       ON/**OFF**            Indicates whether ATLAS comes with the full
                                              LAPACK support 
ENABLE_ACCELERATE       **ON**/OFF            Enables Accelerate as linear algebra library,
                                              only enabled on MacOSX
ENABLE_OPENMP           **ON**/OFF            Enables OpenMP support if supported by the 
                                              platform
BUILD_DOCUMENTATION     ON/**OFF**            Builds the documentation, requires doxygen
BUILD_EXAMPLES          **ON**/OFF            Builds the examples
BUILD_TESTING           **ON**/OFF            Builds the tests
CMAKE_BUILD_TYPE        Debug/**Release**     Builds Shark in Debug or Release mode.
                                              In debug, the linbrary is called shark_debug
CMAKE_INSTALL_PREFIX    Path **/usr/local**   Installation path for Shark

======================= ===================== ===============================================

To change options, either use one of the cmake guis (e.g., ccmake) or add the options to the cmake call.
Choosing another path to boost and disabling OpenMP would look like::

	cmake "-DBOOST_ROOT=/path/to/boost" "-DENABLE_OPENMP=OFF" ../

An introduction on how to run cmake can be found `here <http://cmake.org/runningcmake/>`__.


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

* Download the sources from our :doc:`Downloads page <../downloads/downloads>` and unpack them
* Open the CMake GUI
* Next to "Where is the source code" set the path to the unpacked Shark location
* Next to "Where to build the directory" set the path to where you want the Visual Studio project files to be
* Click on "Add Entry"
* Add an Entry BOOST_ROOT of type PATH and set it to your boost intall directory (e.g. ``C:\locale\boost_1.59``)
* Add an Entry BOOST_LIBRARYDIR of type PATH and set it to your boost library directory (e.g. ``C:\locale\boost_1.59\lib64-msvc-12.0``)
* Set the right Visual Studio compiler and click on Configure (possibly twice) and then on generate
