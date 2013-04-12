//
//  Copyright (c) 2000-2002
//  Joerg Walter, Mathias Koch
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

#ifndef _BOOST_UBLAS_CONFIG_
#define _BOOST_UBLAS_CONFIG_

#include <cassert>
#include <cstddef>
#include <algorithm>
#include <limits>

#include <boost/config.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/and.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/remove_reference.hpp>

// Comeau compiler - thanks to Kresimir Fresl
#if defined (__COMO__) && ! defined (BOOST_STRICT_CONFIG)

// Missing std::abs overloads for float types in <cmath> are in <cstdlib>
#if defined(__LIBCOMO__) && (__LIBCOMO_VERSION__ <= 31)
#include <cstdlib>
#endif

#endif


//  SGI MIPSpro C++ compiler
#if defined (__sgi) && ! defined (BOOST_STRICT_CONFIG)

// Missing std::abs overloads for float types in <cmath> are in <cstdlib>
// This test should be library version specific.
#include <cstdlib>

#endif

// Enable performance options in RELEASE mode
#if defined (NDEBUG) || defined (BOOST_UBLAS_NDEBUG)

// Do not check sizes!
#define BOOST_UBLAS_USE_FAST_SAME

// NO runtime error checks with BOOST_UBLAS_CHECK macro
#ifndef BOOST_UBLAS_CHECK_ENABLE
#define BOOST_UBLAS_CHECK_ENABLE 0
#endif

// Disable performance options in DEBUG mode
#else


// Enable runtime error checks with BOOST_UBLAS_CHECK macro. Check bounds etc
#ifndef BOOST_UBLAS_CHECK_ENABLE
#define BOOST_UBLAS_CHECK_ENABLE 1
#endif

#endif


/*
 * General Configuration
 */

// Choose evaluation method for dense vectors and matrices
#if !(defined(BOOST_UBLAS_USE_INDEXING) || defined(BOOST_UBLAS_USE_ITERATING))
#define BOOST_UBLAS_USE_INDEXING
#endif

// Include type declerations and functions
#include <shark/LinAlg/BLAS/ublas/fwd.hpp>
#include <shark/LinAlg/BLAS/ublas/detail/definitions.hpp>


#endif
