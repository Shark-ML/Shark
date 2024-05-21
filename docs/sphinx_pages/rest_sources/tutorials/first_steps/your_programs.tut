Your Shark Programs
===================

To use the Shark library's functionality, you usually write your own
C++ programs and link them against the Shark library. We below give an example configuration for
CMake, which we recommend using (see
`here <http://cmake.org/runningcmake/>`_ for an introduction). A minimal setup, compiling a program ``ExampleProject`` from
an input file ``HelloWorld.cpp`` is given by

.. code-block:: none

	cmake_minimum_required(VERSION 2.8)

	project(ExampleProject)

	# Find the Shark libraries and includes
	# set Shark_DIR to the proper location of Shark
	find_package(Shark REQUIRED)
	include(${SHARK_USE_FILE})

	# Executable
	add_executable(ExampleProject HelloWorld.cpp)
        set_property(TARGET ExampleProject PROPERTY CXX_STANDARD 11)
	target_link_libraries(ExampleProject ${SHARK_LIBRARIES})


You can find the template CMakeLists.txt in your example folder at
``ExampleProject/CMakeLists.txt``.  It automatically links to all
libraries used by your Shark build configuration and sets the required
compiler flags.

In the following, we describe the command line usage of the file for linux.
If shark was installed, chances are that cmake knows where to find shark
and a simple call to

.. code-block:: none

	cmake .
	make
	
will build the project. If Shark is not installed, or installed to a non-standard path, 
the path to the proper ``SharkConfig.cmake`` file is needed

.. code-block:: none

	cmake "-DShark_DIR=/Path/To/Shark/" .
	
If shark was not installed, ``/Path/To/Shark/`` is simply the build directory,
otherwise it is ``/Shark/Install/Directory/lib/CMake/Shark``. 
For example, if the install directory is ``~/``, the command
is

.. code-block:: none

	cmake "-DShark_DIR=~/lib/CMake/Shark" .


You can easily change the build options using cmake or its guis (e.g., ccmake), see :doc:`../../installation` for specific options.





