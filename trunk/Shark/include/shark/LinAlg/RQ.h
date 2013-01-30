/*!
 *  \brief Algorithm for Triangular-Quadratic-Decomposition
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
#ifndef SHARK_LINALG_RQ_H
#define SHARK_LINALG_RQ_H

#include <shark/LinAlg/Base.h>
namespace shark{

/**
 * \ingroup shark_globals
 * 
 * @{
 */


//! \brief Calculates the RQ-Decomposition of A using pivoting
template<class MatrixT,class Mat>
std::size_t pivotingRQ
(
	blas::matrix_expression<MatrixT> const& matrixA,
	blas::matrix_container<Mat>& matrixR,
	blas::matrix_container<Mat>& matrixQ,
	blas::permutation_matrix<std::size_t> & permutation
);

template<class MatrixT,class MatrixU>
std::size_t pivotingRQHouseholder
(
	blas::matrix_expression<MatrixT> const& matrixA,
	blas::matrix_container<MatrixU>& matrixR,
	blas::matrix_container<MatrixU>& householderTransform,
	blas::permutation_matrix<std::size_t> & permutation
);


/** @}*/
}

#include "Impl/pivotingRQ.inl"
#endif
