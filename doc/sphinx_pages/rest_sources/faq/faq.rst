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
  installation. You can directly copy its output into your bug report.

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


Do I need root/administrator access to install the Shark library?
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

No, root/administrator access is not required. However, if you want to install the library to a
central location (such as ``/usr/lib/`` on Linux) you will of course need write access to that directory.
Otherwise, simply select local prefixes for both the Shark installation as well as any other dependencies
you may need to install (e.g., Boost, ATLAS, etc.).


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
.


Functionality
-------------

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




