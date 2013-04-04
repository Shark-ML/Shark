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

namespace shark{

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

//! Calculates the eigenvalues and the normalized eigenvectors of the
//! symmetric matrix "amatA" using the Jacobi method.
template<class MatrixT,class MatrixU,class VectorT>
void eigensymmJacobi
(
	MatrixT& amatA,
	MatrixU& vmatA,
	VectorT& dvecA
);

//! Calculates the eigenvalues and the normalized
//! eigenvectors of the symmetric matrix "amatA" using a modified
//! Jacobi method.
template<class MatrixT,class MatrixU,class VectorT>
void eigensymmJacobi2
(
	MatrixT& amatA,
	MatrixU& vmatA,
	VectorT& dvecA
);



//! Calculates the eigenvalues and the normalized
//! eigenvectors of a symmetric matrix "amatA" using the Givens
//! and Householder reduction, however, "hmatA" contains intermediate results after application.
template<class MatrixT,class MatrixU,class MatrixV,class VectorT>
void eigensymm_intermediate
(
	const MatrixT& amatA,
	MatrixU& hmatA,
	MatrixV& vmatA,
	VectorT& dvecA
);

//! Used as frontend alculating
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

//! Calculates the relative error of eigenvalue  no. "c".
template<class MatrixT,class MatrixU,class VectorT>
double eigenerr
(
	const MatrixT& amatA,
	const MatrixU& vmatA,
	const VectorT& dvecA,
	unsigned c
);

//! Determines the rank of the symmetric matrix "amatA".
template<class MatrixT,class MatrixU,class VectorT>
unsigned rank
(
	const MatrixT& amatA,
	const MatrixU& vmatA,
	const VectorT& dvecA
);

//! Calculates the determinant of the symmetric matrix "amatA".
template<class MatrixT,class MatrixU,class VectorT>
double detsymm
(
	MatrixT& amatA,
	MatrixU& vmatA,
	VectorT& dvecA
);

//! Calculates the log of the determinant of the symmetric matrix "amatA".
template<class MatrixT,class MatrixU,class VectorT>
double logdetsymm
(
	MatrixT& amatA,
	MatrixU& vmatA,
	VectorT& dvecA
);

//! Calculates the rank of the symmetric matrix "amatA", its eigenvalues and
//! eigenvectors.
template<class MatrixT,class MatrixU,class MatrixV,class VectorT>
unsigned rankDecomp
(
	MatrixT& amatA,
	MatrixU& vmatA,
	MatrixV& hmatA,
	VectorT& dvecA
);

/** @}*/
}

#include "Impl/eigensort.inl"
#include "Impl/eigensymmJacobi.inl"
#include "Impl/eigensymmJacobi2.inl"
#include "Impl/eigensymm.inl"
#include "Impl/eigenerr.inl"
#include "Impl/rank.inl"
#include "Impl/detsymm.inl"
#include "Impl/rankDecomp.inl"
#endif