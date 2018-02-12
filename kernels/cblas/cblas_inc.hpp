/*!
 *
 *
 * \brief       -
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 *
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef REMORA_KERNELS_CBLAS_CBLAS_INC_HPP
#define REMORA_KERNELS_CBLAS_CBLAS_INC_HPP

#ifdef __APPLE__

#ifdef __ASSERTMACROS__ //is AssertMacros already included?
//AssertMacros automatically defines __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES as 1
//if not already included
#if __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES
#warning "AssertMacros.h already included by some file. Disabling macros as otherwise compilation will fail"

//incomplete list (probably the worst offenders that will fail compilation.
#ifdef check
     #undef check
#endif
#ifdef require
     #undef require
#endif
#ifdef verify
     #undef verify
#endif

#endif
#else
//noone included it yet, so we can just prevent these macros...
#define __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES 0
#endif

// included to make Accelerate work with boost on MacOS
#include <boost/intrusive/list.hpp>

// Accelerate framework support added by TG 19.06.2015
extern "C" {
#include <Accelerate/Accelerate.h>
}
#undef nil

#else

extern "C" {
#include <cblas.h>
}

#endif

//all atlas using functions need this anyway...
//so we prevent multiple includes in all atlas using functions
//which should decrease compile time a small bit
#include <complex>
#include "../../detail/traits.hpp"

namespace remora {namespace bindings {

template <typename Ord> struct storage_order {};
template<> struct storage_order<row_major> {
	enum ename { value = CblasRowMajor };
};
template<> struct storage_order<column_major> {
	enum ename { value = CblasColMajor };
};

template<class T>
struct allowed_cblas_type{
	typedef std::false_type type;
};

template<>
struct allowed_cblas_type<float>{
	typedef std::true_type type;
};
template<>
struct allowed_cblas_type<double>{
	typedef std::true_type type;
};
template<>
struct allowed_cblas_type<std::complex<float> >{
	typedef std::true_type type;
};
template<>
struct allowed_cblas_type<std::complex<double> >{
	typedef std::true_type type;
};

}}

#ifndef OPENBLAS_CONST
typedef void cblas_float_complex_type;
typedef void cblas_double_complex_type;
#else
typedef float cblas_float_complex_type;
typedef double cblas_double_complex_type;
#endif


#endif
