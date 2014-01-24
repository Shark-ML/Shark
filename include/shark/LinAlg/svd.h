/*!
 * 
 * \file        svd.h
 *
 * \brief       Algorithm for Singular Value Decomposition
 * 
 * 
 * 
 *
 * \author      O. Krause
 * \date        2011
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
#ifndef SHARK_LINALG_SVD_H
#define SHARK_LINALG_SVD_H

#include <shark/LinAlg/Base.h>

namespace shark{ namespace blas{

/**
 * \ingroup shark_globals
 * 
 * @{
 */


//! \par
//! Determines the singular value decomposition of a rectangular
//! matrix "amatA".
//!
template<class MatrixT,class MatrixU,class VectorT>
void svd
(
	const MatrixT& amatA,
	MatrixU& umatA,
	MatrixU& vmatA,
	VectorT& wvecA,
	unsigned maxIterations = 200,
	bool ignoreThreshold = true
);


/** @}*/
}}

#include "Impl/svd.inl"
#endif