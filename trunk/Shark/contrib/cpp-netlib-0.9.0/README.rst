C++ Network Library
===================

Introduction
------------

cpp-netlib is a collection of network related routines/implementations
geared towards providing a robust cross-platform networking library.
cpp-netlib offers the following implementations:

  *  Common Message Type -- A generic message type which can be used
     to encapsulate and store message related information, used by all
     network implementations as the primary means of data exchange.
  *  Network protocol message parsers -- A collection of parsers which
     generate message objects from strings.
  *  Adapters and Wrappers -- A collection of Adapters and wrappers aimed
     towards making the message type STL friendly.
  *  Network protocol client and server implementations -- A collection 
     of network protocol implementations that include embeddable client 
     and server types.

This library is released under the Boost Software License (please see
http://boost.org/LICENSE_1_0.txt or the accompanying LICENSE_1_0.txt file
for the full text.

Downloading cpp-netlib
----------------------

You can find official release packages of the library at::

    http://github.com/cpp-netlib/cpp-netlib/downloads

Building and Installing
-----------------------

Since cpp-netlib is a header-only library, there is nothing to build. To install
cpp-netlib, you can choose to copy the contents of the ``boost`` directory into
an existing Boost [#]_ distribution or to a different location. All that is
required is for projects that use cpp-netlib when building, have the directory
where cpp-netlib is installed as part of the include paths.

.. [#] http://www.boost.org/

The recommended installation procedure would be to follow the steps below::

    # On Linux/Mac, consider the `$` character as the shell prompt
    $ sudo mkdir -p /usr/local/include/cpp-netlib
    $ sudo cp -r cpp-netlib/boost /usr/local/include/cpp-netlib

Now don't forget to add ``/usr/local/include/cpp-netlib`` in your project's
compiler include directories to start using cpp-netlib in your projects.

Running Tests
-------------

If you want to run the tests that come with cpp-netlib, there are a few things
you will need. These are:

  * A compiler (GCC 4.x or Clang 2.8)
  * A build tool (CMake [#]_ recommended, Boost.Build also an option)
  * OpenSSL headers (optional with CMake, mandatory for Boost.Build)
  * Python 2.6

.. note:: This assumes that you have the cpp-netlib distribution package
   unpacked somwhere in your home directory. This specifically assumes that you
   have cpp-netlib at the toplevel of your home directory.
.. [#] http://www.cmake.org/

Building with CMake
~~~~~~~~~~~~~~~~~~~

To build and run the tests with CMake, you will need to have CMake version 2.8
or higher installed appropriately in your system.

::

    $ cmake --version
    cmake version 2.8.1

Inside the cpp-netlib directory, you can issue the following statements to
configure and generate the Makefiles, and build the tests::

    $ cd ~/cpp-netlib      # we're assuming it's where cpp-netlib is
    $ cmake -DCMAKE_BUILD_TYPE=Debug \
    >    -CMAKE_C_COMPILER=clang     \
    >    -CMAKE_CXX_COMPILER=clang++ \
    >    .

.. note:: This uses the source directory as the build directory as well. At the
   time of this writing, cpp-netlib is meant to be tested in the same directory
   where the source files are, because of the way the tests depend on Python
   being installed and having access to Python scripts during the build.

Once CMake is done with generating the Makefiles and configuring the project,
you can now build the tests and run them::

    $ cd ~/cpp-netlib
    $ make
    $ make test

If for some reason some of the tests fail, you can send the files in
``Testing/Temporary/`` as attachments to the cpp-netlib `developers mailing
list`_.

.. _`developers mailing list`: https://lists.sourceforge.net/lists/listinfo/cpp-netlib-devel

Building with Boost.Build
~~~~~~~~~~~~~~~~~~~~~~~~~

If you don't already have Boost.Build set up on your system, follow the steps
indicated in the Boost Getting Started Guide [#]_ -- you will particularly want
to copy the ``bjam`` executable to a directory that is already in your ``PATH``
so that you don't have to go hunting for it all the time. A good place to put it
is in ``/usr/local/bin``.

.. [#] http://www.boost.org/doc/libs/1_44_0/more/getting_started/index.html

Building and running the tests can be as simple as doing the following::

    $ cd ~/cpp-netlib
    $ bjam

Doing this will already build all the tests and run them as they are built. In
case you encounter any problems and would like to report it to the developers,
please do the following::

    $ cd ~/cpp-netlib
    $ bjam 2>&1 >build-test.log

And then attach the ``build-test.log`` file to the email you will send to the
cpp-netlib `developers mailing list`_.

.. _`developers mailing list`: https://lists.sourceforge.net/lists/listinfo/cpp-netlib-devel

Hacking on cpp-netlib
---------------------

cpp-netlib is being developed with the git_ distributed SCM system.
cpp-netlib is hosted on GitHub_ following the GitHub recommended practice of
forking the repository and submitting pull requests to the source repository.
You can read more about the forking_ process and submitting `pull requests`_ if
you're not familiar with either process yet.

.. _git: http://git-scm.com/
.. _GitHub: http://github.com/
.. _forking: http://help.github.com/forking/
.. _`pull requests`: http://help.github.com/pull-requests/

Because cpp-netlib is released under the `Boost Software License`_ it is
recommended that any file you make changes to bear your copyright notice
alongside the original authors' copyright notices on the file. Typically the
copyright notices are at the top of each file in the project.

.. _`Boost Software License`: http://www.boost.org/LICENSE_1_0.txt

At the time of writing, there are no coding conventions being followed but if
you write in the general style that is already existing in the project that
would be greatly appreciated. Copious amounts of comments will be called out,
but code that is not self-explanatory typically at least requires a rationale
documentation in comments explaining "why" the code is written that way.

The main "upstream" repository is the one hosted by the original maintainer of
the project (Dean Michael Berris) at http://github.com/mikhailberis/cpp-netlib.
The "official" release repository is maintained at
http://github.com/cpp-netlib/cpp-netlib -- which is a fork of the upstream
repository. It is recommended that forks be made against the upstream repostory
and pull requests be submitted against the upstream repository so that patches
and other implementations can be curated by the original maintainer.

Contact and Commercial Support
------------------------------

In case you have any questions or would like to make feature requests, you can
contact the development team through the `developers mailing list`_
or by filing issues at http://github.com/mikhailberis/cpp-netlib/issues.

.. _`developers mailing list`:
   https://lists.sourceforge.net/lists/listinfo/cpp-netlib-devel

You can reach the maintainers of the project through::

    Dean Michael Berris
    mikhailberis@gmail.com

    Glyn Matthews

    Mike Dickey

At this time, paid commercial support is available for cpp-netlib being offered
by the maintainers. In case you have any questions, please feel free to contact
any one of the maintainers above or anybody on the developers mailing list.

