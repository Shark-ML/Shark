.. highlight:: bash

Installing Shark
================

.. contents:: Contents:


Linux, MacOS, and other Unix-based systems
**********************************************************

To install Shark, get the sources::

	svn co https://svn.code.sf.net/p/shark-project/code/trunk/Shark
	
or a source-code release as described :doc:`here
<../downloads/downloads>`.
Then build the library::

	mkdir Shark/build/
	cd Shark/build
	cmake ../
	make
	
Dependencies
---------------------------------------------

Shark relies on `Boost <http://www.boost.org>`_ and uses `CMake
<http://www.cmake.org/>`_.
Furthermore, Shark can make use of different linear algebra libraries.
On MacOsX, Accelerate is used by default. On linux and Windows, ATLAS
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

To enable, either
use one of the cmake guis or add the options to the cmake call.
Choosing another path to boost and disabling OpenMP would look like::

	cmake "-DBOOST_ROOT=/path/to/boost" "-DENABLE_OPENMP=OFF" ../
