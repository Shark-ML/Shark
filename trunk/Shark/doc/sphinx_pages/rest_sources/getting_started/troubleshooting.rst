.. highlight:: bash

Troubleshooting the installation procedure
==========================================

.. contents:: Contents:

This page covers problems that might occur during the Shark installation procedure. The usual,
out-of-the-box installation procedure is explained on the :doc:`installation page
<../getting_started/installation>`.

We now first list some hints for getting Shark to find Boost if it does not automatically do so.
After that, we give some hints for compiling Boost yourself if you need to. This is followed
by miscellaneous hints.


.. _label_for_findboost:

Getting Shark to find Boost
---------------------------


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



If you encounter problems with the latest SVN sources
-----------------------------------------------------


It is our policy that code in the SVN repository should always compile,
although there may be incomplete functionality at times. However, at rare
times it does happen that a current SVN snapshot does not compile on one
or more platforms. In such a case, you can examine the commit logs, inqure
with the mailing list, or try one or two commits further up or down.


.. todo::

    add link to build server for people to check status of SVN HEAD
