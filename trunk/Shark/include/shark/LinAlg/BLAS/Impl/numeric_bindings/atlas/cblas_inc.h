/*
 *
 * Copyright (c) Kresimir Fresl 2002
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * Author acknowledges the support of the Faculty of Civil Engineering,
 * University of Zagreb, Croatia.
 *
 */

//////////////////////////////////////////////////////////////////////////
//
// ATLAS (Automatically Tuned Linear Algebra Software)
//
// ''At present, it provides C and Fortran77 interfaces to a portably
// efficient BLAS implementation, as well as a few routines from LAPACK.''
//
// see: http://math-atlas.sourceforge.net/
//
//////////////////////////////////////////////////////////////////////////

#ifndef SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_CBLAS_INC_H
#define SHARK_LINALG_IMPL_NUMERIC_BINDINGS_ATLAS_CBLAS_INC_H

extern "C" {
#include <cblas.h>
#include <clapack.h>
}

//all atla susing functions need this anyway...
//so we prevent multiple includes in all atlas using functions
//which should decrease compile time a small bit
#include <shark/LinAlg/BLAS/traits/matrix_raw.hpp>
#include <shark/LinAlg/BLAS/traits/vector_raw.hpp>
#include <shark/Core/Exception.h>
#include <complex>

namespace shark {
namespace blas {
namespace bindings {

template <typename Ord> struct storage_order {};
template<> struct storage_order<row_major_tag> {
	enum ename { value = CblasRowMajor };
};
template<> struct storage_order<column_major_tag> {
	enum ename { value = CblasColMajor };
};


template <typename UpLo> struct uplo_triang {};
template<> struct uplo_triang<upper_tag> {
	enum ename { value = CblasUpper };
};
template<> struct uplo_triang<lower_tag> {
	enum ename { value = CblasLower };
};

}

}
}


#endif
