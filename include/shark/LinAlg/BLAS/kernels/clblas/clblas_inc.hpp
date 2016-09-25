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

#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_CBLAS_INC_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_CBLAS_INC_HPP
#

extern "C" {
#include <clBLAS.h>
}
#include <boost/compute/container/vector.hpp>
#include "../../expression_types.hpp"

namespace shark {
namespace blas {
namespace clblas{

template <typename Ord> struct storage_order {};
template<> struct storage_order<row_major> {
	enum ename { value = clblasRowMajor };
};
template<> struct storage_order<column_major> {
	enum ename { value = clblasColumnMajor };
};

}}}

#endif
