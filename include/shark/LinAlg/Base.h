/*!
 * 
 *
 * \brief       Entry Point for all Basic Linear Algebra(BLAS) in shark
 * 
 * 
 *
 * \author      O.Krause, T.Glasmachers, T. Voss
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_LINALG_BASE_H
#define SHARK_LINALG_BASE_H

/**
* \brief Shark linear algebra definitions
*
* \par
* This file provides all basic definitions for linear algebra.
* If defines objects and views for vectors and matrices over
* several base types as well as a lot of usefull functions.
*/

//for debug error handling of linear algebra
#include <shark/Core/Shark.h>
#include <shark/LinAlg/BLAS/remora.hpp>
#include <shark/Core/Exception.h>
#include <shark/LinAlg/Metrics.h>
//this ensures, that Sequence is serializable
#include <boost/serialization/deque.hpp>
#include <deque>

namespace shark{

namespace blas{
using namespace remora;
}
#define SHARK_VECTOR_MATRIX_TYPEDEFS(basetype, prefix) \
	typedef blas::vector< basetype > prefix##Vector; \
	typedef blas::matrix< basetype, blas::row_major > prefix##Matrix; \
	typedef blas::compressed_vector< basetype > Compressed##prefix##Vector; \
	typedef blas::compressed_matrix< basetype > Compressed##prefix##Matrix;

	SHARK_VECTOR_MATRIX_TYPEDEFS(long double, BigReal);
	SHARK_VECTOR_MATRIX_TYPEDEFS(double, Real)
	SHARK_VECTOR_MATRIX_TYPEDEFS(float, Float)
	SHARK_VECTOR_MATRIX_TYPEDEFS(std::complex<double>, Complex)
	SHARK_VECTOR_MATRIX_TYPEDEFS(int, Int)
	SHARK_VECTOR_MATRIX_TYPEDEFS(unsigned int, UInt)
        SHARK_VECTOR_MATRIX_TYPEDEFS(bool, Bool);
#undef SHARK_VECTOR_MATRIX_TYPEDEFS

typedef blas::vector< double, blas::gpu_tag > RealGPUVector;
typedef blas::vector< float, blas::gpu_tag > FloatGPUVector;
typedef blas::matrix< double, blas::row_major, blas::gpu_tag > RealGPUMatrix;
typedef blas::matrix< float, blas::row_major, blas::gpu_tag > FloatGPUMatrix;

typedef blas::permutation_matrix PermutationMatrix;

///Type of Data sequences.
typedef std::deque<RealVector> Sequence;
}
#endif
