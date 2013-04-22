Frequently asked questions
==========================

.. contents:: Contents:

General
-------

Is there a mailing list for users of the Shark library?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Yes, the mailing list is available at
https://lists.sourceforge.net/lists/listinfo/shark-project-user.

.. _help:

Where can I get help if my problem is not covered by the available documentation or this FAQ?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Multiple support channels are available for the Shark machine learning library:

* Shark Mailing List Archive: http://sourceforge.net/mailarchive/forum.php?forum_name=shark-project-user
* Shark Mailing List Interface: https://lists.sourceforge.net/lists/listinfo/shark-project-user
* Shark bug tracker: http://sourceforge.net/tracker/?group_id=93510&atid=604542
* Shark feature request tracker: http://sourceforge.net/tracker/?group_id=93510&atid=604545

Whenever you are reporting a bug, you should briefly scan over our below bug
reporting guidelines to help our developers in resolving the issue:

* Check the `bug database <http://sourceforge.net/tracker/?group_id=93510&atid=604542>`_
  if your problem has already been reported.

* Provide information on the Shark version you are using and the platform you are working on.
  To this end, you can execute the program ``Version`` that is part of the default Shark
  installation (located in the ``bin/`` sub-folder of your installation directory). It will
  give you an output similar to::

    Shark Machine Learning Library Ver. 3.0.0
    Official release: false
    Platform: Mac OS
    Compiler: GNU C++ version 4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2335.15.00)
    C++ Std. Lib.: GNU libstdc++ version 20070719
    Boost Ver.: 1.44.0
    Build Type: Release
    OpenMP Enabled: false

  You can directly copy this information into your bug report.

* If the problem is related to the early stages of using CMake to configure
  the Shark installation (such as not finding the Boost installation), please
  checkout the latest Shark version and then generate debug output from the
  CMake configuration run by first deleting any CMakeCache.txt in your shark
  source/main directory, and then issuing in your build directory (may be the
  source directory) ``rm -rf CMakeCache.txt; cmake /path/to/shark/main/dir -DBoost_DEBUG=1``.
  Please attach the output of this to your bug report if indicated, and possibly
  the contents of your CMakeCache.txt.

* Try to provide us with a minimal test case for reproducing the problem.



Installation
------------

How do I install the Shark library?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Please click on :doc:`Getting started <../getting_started/installation>`.

My installation fails - what now?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Please first see :doc:`the dedicated installation troubleshooting page <../getting_started/troubleshooting>`.
If that didn't help, see the above section :ref:`help`.

I am a MacOS-user - where do i get a recent compiler?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Apple decided not to distribute recent versions of gcc with Xcode anymore. Instead the clang compiler
is distributed, which is also supported by shark. To solve the problem, either install a new version of gcc
using macports or use the clang compiler, for example by issuing 

   export CC=clang
   export CXX=clang++

before using cmake. Note however that clang does not support OpenMp yet.


Do I need root/administrator access to install the Shark library?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

No, root/administrator access is not required. However, if you want to install the library to a
central location (such as ``/usr/lib/`` on Linux) you will of course need write access to that directory.
Otherwise, simply select local prefixes for both the Shark installation as well as any other dependencies
you may need to install (e.g., Boost, ATLAS, etc.).


I get strange warnings when I compile Shark using a certain Microsoft compiler. What should I do?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Please just ignore them.


I get an error message C2589 concerning "std::min" when compiling using Microsoft Visual Studio
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

The solution is to add NOMINMAX to the list of compiler macro declarations.


For some reason, cmake does not find boost
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Please see the troubleshooting site on :ref:`label_for_findboost` for how to provide
hints to the Shark CMake configuration about the location of Boost.


How do I install the Shark documentation?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

There is a concise "README.txt" file in the ``doc/`` subdirectory intended for people who do not
have access to the documentation while building the documentation. Also, there is a tutorial on
the documentation system, including building the documentation, located
:doc:`here <../tutorials/for_developers/managing_the_documentation>`. Finally, there will be a
separate documentation package available for download on the
:doc:`download page <../downloads/downloads>`.


Compilation of programs
-----------------------


I get an error when compiling my/example programs stating that certain symbols are not found. The letters "boost" occur in the error message often. What can I do?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Shark relies on `Boost <http://www.boost.org>`_. You must have Boost installed and explicitly link
against the required Boost libraries. The library boost_serialization is always used, some programs
require boost_system, boost_filesystem, and/or boost_program_options. So if you are using gcc, adding
``-lboost_serialization -lboost_system -lboost_filesystem -lboost_program_options`` solves the problem.
Please see the :doc:`installation guide <../getting_started/installation>` as well as the
:doc:`installation troubleshooting page <../getting_started/troubleshooting>` for additional information
on how to build, find, and link to Boost. Also have a look at the auto
generated :ref:`CMake files for projects using Shark <label_for_cmake_example_project>`.


Functionality
-------------



What happened to the [...] library, the [...] add-on package, the function [...]?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

We are in the process of porting all relevant features to the new version of Shark.
If you miss a certain feature, post a feature inquiry to the mailing list or a feature
request to the Sourceforge feature request page. However we are sure that most 
of the functionality is there.



What are the differences between Shark and other libraries? Why should I use Shark?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Shark is a native C++ library designed for development and real-world
application of state-of-the-art machine learning and optimization
algorithms. The library has a history of more than 10 years of
successful applications. It is actively supported and still
growing. We are continuously extending and improving the algorithms in
various domains of machine learning and computational intelligence.

Flexibility and speed are the main design criteria. We think that its
flexibility and extensibility make Shark stand out from other libraries.

It is mostly self-contained and offers computational intelligence
techniques such as single- and multi-objective evolutionary algorithms
and neural networks as well as kernel-based machine learning methods
and classical optimization techniques in a coherent framework. This is
unique.

Shark is an object-oriented software library and to use it requires
knowledge in C++ programming. If a graphical user interface is
important for you, you may go for other machine learning software (or
feel free to contribute such a front-end for Shark).

Shark implements a lot of powerful algorithms not available in any
other machine learning library, of course in particular methods based
on the research of the developers.

Some highlights:

* The Shark SVM is the only SVM package implementing the fastest
  SMO-based learning algorithm for binary and multi-class support
  vector machines.
* Shark provides a variety of model-selection algorithms for SVMs, for example gradient-based optimization
  of the kernel-target alignment, which is not available in any other library.
* Shark provides a large collection of efficient gradient-based optimization techniques, for example the
  frequently applied iRprop+, a fast and robust method not available in other machine learning libraries.
* We do not know any software library for single-objective evolutionary algorithms that comes close to
  Shark in terms of variety and quality of algorithms for real-valued optimization. To our knowledge,
  Shark is also one the most comprehensive libraries for evolutionary multi-objective optimization. The efficient
  implementation of the hypervolume metric (S or Lebesgue measure) and of the powerful MO-CMA-ES are special
  features.




