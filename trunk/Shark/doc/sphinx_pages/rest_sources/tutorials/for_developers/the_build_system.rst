The Shark Build System Setup
============================

Some information on Shark's configuration options for CMake is already provided
on the :doc:`installation page <../../getting_started/installation>`.
We here give supplemental information for developers.

How to Add Examples, Unit Tests and New Files
---------------------------------------------

How to add a unit test to the existing setup?
	Under the assumption of your test implementation residing in Shark/Test/mytest.cpp, add a line reading
	``SHARK_ADD_TEST( mytest.cpp mytest )`` to the file Shark/Test/CMakeLists.txt.
	
How to add a new example to the existing setup?
	Under the assumption of your example implementation residing in Shark/examples/myexamples.cpp, add a line reading
	``SHARK_ADD_EXAMPLE( myexample.cpp myexample )`` to the file Shark/examples/CMakeLists.txt.
	
How to add header and/or source files to the library?
	Under the assumption of a file myfile.h/cpp residing in Shark/include/shark/Core add the lines include/shark/Core/myfile.h and include/shark/Core/myfile.cpp
	to CORE_HEADERS and CORE_SRCS in Shark/CMakeLists.txt. In general, your files need to be added to the toplevel CMakeLists.txt to be compiled and/or
	installed. For specific instructions regarding unit tests and examples, see the instructions listed before.
