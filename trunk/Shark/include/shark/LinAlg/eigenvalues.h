/*!
 *  \brief Algorithms for Eigenvalue decompositions
 *
 *
 *  \author  O. Krause
 *  \date    2011
 *
 *  \par Copyright (c) 1999-2001:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 *
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