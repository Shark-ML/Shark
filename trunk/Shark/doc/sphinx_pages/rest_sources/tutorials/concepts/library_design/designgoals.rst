
Design Goals
============

The major design goals of the Shark library are speed, modularity, and
portability. There are a number of different aspects of these goals, and
they require a number of trade-offs.


Speed
-----

The speed of the library depends on two major factors. First of all, C++
translates into fast code. Second, the interfaces are designed such that
information can be propagated efficiently. Expensive copying of data is
avoided whereever possible.


Modularity
----------

Shark is highly modular, aiming for maximal reusability of code. This
modularity is formalized by a number of core interfaces, as well as
heavy use of templatization. For example, optimization algorithms
communicate with objective functions only through top level interfaces,
such that in principle every algorithm can be used for every objective
function. Furthermore, this works seamlessly with dense and sparse
vectors, and the user is free to rely on arbitrary data structures
describing the search space (such as strings, graphs, and so on).


Portability
-----------

In our experience portability has two main aspects, namely the choice of
the programming language, and the dependencies of the software.

Being a C++ library, Shark depends on the presence of a modern and
(largely) standard compliant C++ compiler. However, standard compliant
C++ is an extremely portable programming language with compilers
available for practically all major platforms. This allows us to compile
Shark and programs using the library on all major operating systems, but
even on embedded and mobile platforms.

A piece of software is only as portable as its dependencies. Thus, we
keep Shark as self-contained as possible. However, we rely on a few
more dependencies:

We use the Cmake build system to compile Shark and its accompanying
example and unit test programs. Cmake is quickly becoming a widespread
standard. It is widely available.

Shark relies on the boost libraries for a number of tasks, such as
efficient linear algebra operations on dense and sparse data structures
and serialization, see http://www.boost.org/.
These well-established libraries provide stable solutions to many
standard problems and are themselves extremely portable. Thus, the
dependency on boost is expected to even further increase the portability
of Shark.

Optionally Shark can be linked with support for reading and writing the
HDF5-based format of the machine learning data website mldata.org
(see http://www.hdfgroup.org/HDF5/ and http://www.mldata.org/).
The cmake build system tries to locate this library automatically and
disables the functionality if HDF5 is not found. Thus, this dependency
does not affect the portability of the rest of the Shark library.

Similarly, Shark uses openMP to parallelize certain loops, such as in
evolutionary computation, if possible. This feature is not available if
openMP is not found on the target system, and its absence does not
affect portability.

The same applies to ATLAS as an optimised implementation of the famous BLAS linear
algebra package. While Shark works completely out of the box without ATLAS, it
can achieve higher performance using this library. Especially on higher dimensional
problems it is advisable to use ATLAS.
