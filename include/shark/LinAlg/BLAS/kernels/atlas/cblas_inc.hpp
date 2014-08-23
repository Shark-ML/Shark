/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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

#ifndef SHARK_LINALG_BLAS_KERNELS_ATLAS_CBLAS_INC_HPP
#define SHARK_LINALG_BLAS_KERNELS_ATLAS_CBLAS_INC_HPP

extern "C" {
#include <cblas.h>
#include <clapack.h>
}

//all atlas using functions need this anyway...
//so we prevent multiple includes in all atlas using functions
//which should decrease compile time a small bit
#include <shark/Core/Exception.h>
#include <complex>
#include <boost/mpl/bool.hpp>
#include "../traits.hpp"

namespace shark {
namespace blas {
namespace bindings {

template <typename Ord> struct storage_order {};
template<> struct storage_order<row_major> {
	enum ename { value = CblasRowMajor };
};
template<> struct storage_order<column_major> {
	enum ename { value = CblasColMajor };
};

}

}
}


#endif
