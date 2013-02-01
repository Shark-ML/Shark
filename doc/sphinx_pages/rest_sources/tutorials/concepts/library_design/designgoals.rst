
Design Goals
============

The major design goals of the Shark library are speed, modularity, and
portability. These goals are partially conflicting making trade-offs necessary.


Speed
-----

Shark is written in C++, which in general translates into fast
code. The interfaces are designed such that information can be
propagated efficiently. Expensive copying of data is avoided wherever
possible. Shark can be linked against ATLAS (Automatically Tuned
Linear Algebra Software) to guarantee fast matrix and vector
operations.


Modularity
----------

Shark is highly modular, aiming for maximal reusability of code. This
modularity is formalized by a number of core interfaces, as well as
heavy use of templatization. For example, optimization algorithms
communicate with objective functions only through top level interfaces,
such that in principle every algorithm can be used for every objective
function. Furthermore, this works seamlessly with dense and sparse
vectors, and the user is free to rely on arbitrary data structures
describing search space (such as strings, graphs, etc.).


Portability
-----------

The choice of the programming language and the dependencies on other
software are key aspects of portability.  Shark depends on the
presence of a modern and (largely) standard compliant C++
compiler. Standard compliant C++ is an extremely portable programming
language with compilers available for practically all major
platforms. This allows us to compile Shark and programs using the
library on all major operating systems, even on embedded and mobile
platforms.

A piece of software is only as portable as its dependencies. Thus, we
keep Shark as self-contained as possible. However, we rely on a few
dependencies:

We use the CMake build system to compile Shark and its accompanying
example and unit test programs. CMake is available for a large,
increasing number of platforms, see http://www.cmake.org/.

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
The CMake build system tries to locate this library automatically and
disables the functionality if HDF5 is not found. Thus, this dependency
does not affect the portability of the rest of the Shark library.

Similarly, Shark uses openMP to parallelize certain loops, such as in
evolutionary computation, if possible. This feature is not available if
openMP is not found on the target system, and its absence does not
affect portability.

The same applies to ATLAS as an optimized implementation of the famous
BLAS linear algebra package. While Shark works completely out of the
box without ATLAS, it can achieve higher performance using this
library. Especially on higher dimensional problems it is advisable to
use ATLAS.
