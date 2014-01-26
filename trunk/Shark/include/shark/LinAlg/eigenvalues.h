/*!
 * 
 *
 * \brief       Algorithms for Eigenvalue decompositions
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
#ifndef SHARK_LINALG_EIGENVALUES_H
#define SHARK_LINALG_EIGENVALUES_H

#include <shark/LinAlg/Base.h>

namespace shark{ namespace blas{

/**
 * \ingroup shark_globals
 * 
 * @{
 */

//! Sorts the eigenvalues in vector "dvecA" in descending order and the corresponding
//! eigenvectors in matrix "vmatA".
template<class MatrixT,class VectorT>
void eigensort
(
	MatrixT& vmatA,
	VectorT& dvecA
);

//! Used as frontend calculating
//! the eigenvalues and the normalized eigenvectors of a symmetric matrix "amatA" using the Givens
//! and Householder reduction without corrupting "A" during application. Each time this frontend is
//! called additional memory is allocated for intermediate results.
template<class MatrixT,class MatrixU,class VectorT>
void eigensymm
(
	const MatrixT& A,
	MatrixU& G,
	VectorT& l
);


//! Calculates the eigenvalues and the normalized eigenvectors of a symmetric matrix "amatA" using the Givens
//! and Householder reduction. This method works without corrupting "amatA" during application by demanding
//! another Array 'odvecA' as an algorithmic buffer instead of using the last row of 'amatA' to store
//! intermediate algorithmic results.
template<class MatrixT,class MatrixU,class VectorT>
void eigensymm
(
	const MatrixT& amatA,
	MatrixU& vmatA,
	VectorT& dvecA,
	VectorT& odvecA
);

/** @}*/
}}

#include "Impl/eigensort.inl"
#include "Impl/eigensymm.inl"
#endif