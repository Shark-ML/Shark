.. highlight:: bash

Troubleshooting the installation procedure
==========================================

.. contents:: Contents:

This page lists possible problems that might occur during the Shark installation procedure. The usual,
out-of-the-box installation procedure is explained :doc:`here <../getting_started/installation>`.




.. _label_for_findboost:

Getting Shark to find Boost
---------------------------



If you obtained Boost through your package manager, or alternatively compiled Boost in accordance
with the below directions, Shark should be able to find Boost automatically, at the very least
if you also invoke CMake with the options ``-DBOOST_ROOT:Path=/path/to/boost -DBoost_NO_SYSTEM_PATHS=TRUE``.
If there are still problems (e.g., after issuing ``make``, the compilation aborts with long error
messages involving the word "boost"; or the Makefile cannot even be generated due to the Shark
CMake configuration not being able to complete), then consider any and/or all of the following:

  * Check if everything is maybe actually working anyways. There is a known issue with CMake,
    namely that CMake can list ``Boost_DIR`` as ``Boost_DIR-NOTFOUND``, even if Boost was
    actually found and everything is working. So, if this is true for you, just ignore the
    ``Boost_DIR-NOTFOUND`` (NB: The same constellation can happen with CMake finding/not-finding
    the HDF5 package as well).

  * In at least one case, ensuring a restart between Boost installation and invoking CMake
    solved the issue.

  * De-installing concurrent lower versions of Boost.

  * Checkout the latest Shark version, delete any CMakeCache.txt in your shark source/main
    directory, and then examine the output of in your build directory (may be the source
    directory) issuing ``rm -rf CMakeCache.txt; cmake /path/to/shark/main/dir -DBoost_DEBUG=1``,
    where the last call may possibly be augmented by hints for the boost root path, etc., as
    described above. Also, possibly examine the contents of your CMakeCache.txt.

  * Possibly specify the Boost include and lib directory separately, as in
    ``-DBoost_INCLUDE_DIRS=/path_to_boost_includes -DBoost_LIBRARY_DIRS=/path_to_boost_libs`` .

  * A more drastic measure is to add ::

		SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "...")
		SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "...")

    to your ``CMakeLists.txt`` before ``FIND_PACKAGE( Boost  )``, where the
    dots ``...`` have to be replaced by the proper paths on your system.

  * An even more drastic measure is manually replacing all incorrect occurrences of
    Boost library paths in the file ``CMakeCache.txt``, saving the file, and then
    re-invoking make.




Notes on installing Boost manually
----------------------------------


As said in the installation instructions, Shark does not work with a Boost version older than 1.45.
If you cannot simply update Boost using the software manager of your system, please download Boost
from `here <http://www.boost.org/users/download/>`_ and follow the `installation instructions
<http://www.boost.org/doc/libs/release/more/getting_started>`_ for a manual compilation/installation.

We here aggregate a few notes on a manual Boost installation, in the hope that they will stay in
sync with the respective current Boost installation procedure:

* After uncompressing the tarball, we recommend passing a user-defined target prefix path to the
  bootstrapping script::

	  ./bootstrap.sh --prefix=/your/target/path/to/new/boost/installation/

* In the next step, decide whether or not you want to compile Boost with MPI support. If so,
  you have to add a line ``using mpi ;`` to the file ``tools/build/v2/user-config.jam``.

  .. note::
	As of Boost 1.48, there can be a problem with MPI-enabled builds in Boost, resulting in
	the error message "duplicate name" for the mpi libraries. The current best solution, if
	you run into this problem, is to not use MPI. Also see http://trac.macports.org/ticket/31864 ,
	http://stackoverflow.com/questions/6577440/building-boost-1-46-1-with-openmpi , and
	http://boost.2283326.n4.nabble.com/BJam-error-quot-Duplicate-name-of-actual-target-quot-with-MPI-Python-bindings-td2692836.html ,
	and/or do a search engine query for ``boost build error "duplicate name" mpi``.


* In order for the Shark installation CMake files to find your custom Boost installation,
  we most highly recommend building Boost with build type ``complete`` and a ``tagged``
  installation layout. This means that, as the next and final command for your manual Boost
  installation, you should issue ::

	./b2 --build-type=complete --layout=tagged --prefix=/your/target/path/to/new/boost/installation/ install

  if you opted for the MPI option above, and ::

	./b2 --build-type=complete --layout=tagged --prefix=/your/target/path/to/new/boost/installation/ --without-mpi install

  if you do not want to compile Boost with MPI support.


* If you want, you can validate your Boost installation with the two validation programs provided on
  the Boost installation page. Note, however, that the second one may fail with a segfault due to a
  known problem on the Boost side.

* If you followed the above steps, that is, issued the two commands::

    ./bootstrap.sh --prefix=/your/target/path/to/new/boost/installation/
    ./b2 --build-type=complete --layout=tagged --prefix=/your/target/path/to/new/boost/installation/ --without-mpi install

  then Shark should really be able to find your Boost installation right away, provided you pass the two options ::

	  -DBOOST_ROOT:Path=/your/target/path/to/new/boost/installation/ -DBoost_NO_SYSTEM_PATHS=TRUE

  to the CMake configuration system. Otherwise, see the above entry :ref:`label_for_findboost`.


Further notes on building Shark
-------------------------------

* See the above entry :ref:`label_for_findboost` for comments on providing hints
  to the Shark CMake configuration about the location of Boost.

* ``make install`` and the data folder

  .. todo::

    elaborate on the relevance of make install for the examples and example data folder.

* In order to build dynamically linked libraries..

  .. todo::

    elaborate on dyn-linked libs, LD_LIBRARY_PATH, etc.




If you encounter problems with HDF5
-----------------------------------


One quite common error is an error message saying that the HDF5 package
does not have a component called ``HL``. This message appears when the
high-level API is not included in your HDF5 version. One remedy is to
install it via your distributions' packaging system. Another solution,
if you do not need HDF5 support, is to simply (and brutally) comment out
the HDF5 section in the main Shark ``CMakeLists.txt``.



If you encounter problems with the pre-compiled Shark binaries
--------------------------------------------------------------



It might happen that you encounter a problem or bug in one of the pre-compiled
Shark binaries. If that is the case, drop a mail to the mailinglist, ask if it
is fixed in the SVN repository, and possibly compile from the latest sources.

In the past, we also received some notes from Microsoft Windows users about
currently unexplained issues with the Windows binaries, which can usually be
resolved by building from source. We are currently looking further into such
reports.



If you encounter problems with the latest SVN sources
-----------------------------------------------------


It is our policy that code in the SVN repository should always compile,
although there may be incomplete functionality at times. However, at rare
times it does happen that a current SVN snapshot does not compile on one
or more platforms. In such a case, you can examine the commit logs, inqure
with the mailing list, or try one or two commits further up or down.


.. todo::

	add link to build server for people to check status of SVN HEAD


.. _label_for_mingw:

Notes on installing Shark under MinGW
--------------------------------------

While it is our goal that Shark should compile and run under MinGW, this
happens to be a rarely encountered setup among our users, and also none of
the core developers are working under MinGW. For this reason, installation
support for Shark under MinGW is still somewhat anectdotal. However,
there are two documented cases of successful Shark installations
under MinGW. Out-of-the-box, you will probably encounter problems related
to CMake identifying the platform as Windows and setting options not
suited for MinGW. Below are different workaround strategies:

* For one user, it was enough to comment out the line containing ::

	SET( DISABLE_WARNINGS "/wd4250 /wd4251 /wd4275 /wd4800 /wd4308" )

  in the main Shark CMakeLists.txt file, then deleting CMakeCache.txt
  and starting over.

* Another user manually disallowed the ``WIN32`` mode for CMake
  by inserting ::

	SET (WIN32 OFF)

  in the main Shark CMakeLists.txt file, for example directly under the
  line ::

	CMAKE_POLICY(SET CMP0003 NEW)

  For unclear reasons, that same user also got errors related to
  "Linking CXX executable bin/Logger.exe". This was solved by using
  a modified cmake command along the lines of ::

	cmake -G 'MSYS Makefiles' -D CMAKE_BUILD_TYPE=Release -D PTHREAD_LIBRARY=/path_to_boost_libs/libboost_thread-mgw47-mt-1_51.dll.a -D Boost_INCLUDE_DIRS=/path_to_boost_includes -D Boost_LIBRARY_DIRS=/path_to_boost_libs -D	OPT_DYNAMIC_LIBRARY:BOOL=OFF -D OPT_MAKE_TESTS:BOOL=ON -D OPT_COMPILE_EXAMPLES:BOOL=ON -D OPT_COMPILE_DOCUMENTATION:BOOL=OFF <SHARK_SRC_DIR>

  Here, the relevant option is ``-D PTHREAD_LIBRARY=/path_to_boost_libs/libboost_thread-mgw47-mt-1_51.dll.a``,
  but unfortunately, it is not exactly clear what causes the problem with the logger and why this option is
  needed exactly. The rest of the long command is included to serve as an example for configuring the Shark
  build directly via cmake under MinGW in general.

We welcome any other reports on MinGW issues, solutions, and success stories.



Notes on installing ATLAS manually
----------------------------------

First note that Shark only uses the ATLAS routines itself, and no additional installation of
LAPACK beyond the LAPACK-functionality that ATLAS itself supports is needed.

Second, when building ATLAS, we recommend passing the CPU cycle-per-second value, the bitwidth
of your system (32 or 64 bit), and the installation destination prefix. In summary, from
a dedicated build directory (not the source or destination directory) issue::

    ../relative/path/to/atlas/source/dir/configure -D c -DPentiumCPS=<your MHz> -b 64 --prefix=/desired/installation/path

where ``<your MHz>`` is your processor's speed in MHz (see e.g. ``/proc/cpuinfo`` on Unix machines),
and 64 should be changed to 32 on 32bit platforms accordingly.

From here, continue with the ``make build; make check; make time; make install`` as outlined in the
ATLAS installation guide. Overall, we thus recommend the same minimal install as outlined in
`"Basic Steps of an ATLAS install" <http://math-atlas.sourceforge.net/atlas_install/atlas_install.html#SECTION00033000000000000000>`_.

In addition, we second the recommendation to
`turn off CPU throttling <http://math-atlas.sourceforge.net/atlas_install/atlas_install.html#SECTION00032000000000000000>`_
when building ATLAS.

Please also see the section on :ref:`LinAlg and ATLAS <label_for_linalg_atlas>` in the LinAlg vector and matrix tutorial.

Verifying your ATLAS installation
*********************************

If you followed the instructions on the installation page for building Shark with ATLAS, that is, passed the correct ATLAS
path to CMake, and also set the option "OPT_ENABLE_ATLAS" to "ON", you should then note a significant speed-up in the
supported linear algebra operations. To verify this, simply run the LinAlg_FastProd test in a way that the output does
not get suppressed (as usually done by ctest): either call the test binary directly, ::

	./Test/bin/LinAlg_FastProd

or simply call ctest with the ``-V`` (verbose) option ::

	ctest -V -R FastProd

You will see a comparative output of running times for different operations either calling the ATLAS-enabled fast_prod,
or the non-ATLAS, ublas axpy_prod. On one of our development machines, we get the following as part of the output without
ATLAS enabled::

	Benchmarking matrix matrix prod for medium sized matrices
	fast_prod AX: 3.90975
	fast_prod A^TX: 4.19306
	fast_prod AX^T: 3.66976
	fast_prod A^TX^T: 3.73642

And this equivalent of it with ATLAS enabled::

	Benchmarking matrix matrix prod for medium sized matrices
	fast_prod AX: 0.716619
	fast_prod A^TX: 0.713287
	fast_prod AX^T: 0.693288
	fast_prod A^TX^T: 0.693288

As you can see, the latter gives a substantial speed-up, and we can be sure that the ATLAS backend is indeed working.
