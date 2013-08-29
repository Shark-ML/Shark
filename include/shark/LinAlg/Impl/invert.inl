//===========================================================================
/*!
 *  \file invert.inl
 *
 *  \brief Determines the inverse matrix of a full-rank input matrix
 *   	   by using boost-ublas' LU-factorization.
 *
 *  \author  O. Krause
 *  \date    2011
 *
 *  \par Copyright (c) 1998-2000:
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
 *  <BR><HR>
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
 */
//===========================================================================
#ifndef SHARK_LINALG_IMPL_INVERT_INL
#define SHARK_LINALG_IMPL_INVERT_INL
#include <shark/LinAlg/BLAS/ublas/triangular.hpp>
#include <shark/LinAlg/BLAS/ublas/lu.hpp>

#ifdef SHARK_USE_ATLAS
#include <shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/potri.h>
#endif

//===========================================================================
/*!
 *  \brief Returns the inverse of the input matrix. The matrix must have full rank.
 *  \param  input The input matrix.
 *  \return   The inverse matrix.
 */
template<class MatrixT>
shark::RealMatrix shark::blas::invert(const MatrixT& input)
{

	MatrixT inverse(input.size2(), input.size1());

 	typedef permutation_matrix<std::size_t> pmatrix;
 	// create a working copy of the input
 	shark::RealMatrix A(input);
 	// create a permutation matrix for the LU-factorization
 	pmatrix pm(input.size1());

 	// perform LU-factorization
 	int res = lu_factorize(A,pm);
	if( res != 0){
		throw SHARKEXCEPTION("[invert] matrix not invertable");
	}
 	// create identity matrix of "inverse"
 	inverse.assign(identity_matrix<double>(A.size1()));

 	// backsubstitute to get the inverse
 	lu_substitute(A, pm, inverse);

 	return inverse;
}
template<class MatrixT,class MatrixU>
void shark::blas::invertSymmPositiveDefinite(MatrixT &I, const MatrixU& ArrSymm)
{

#ifdef SHARK_USE_ATLAS
	unsigned m = ArrSymm.size1();
	choleskyDecomposition(ArrSymm, I);
	bindings::potri(CblasLower,I);
	for(std::size_t i = 0; i != m; ++i){
		for(std::size_t j = 0; j < i; ++j){
			I(j,i) = I(i,j);
		}
	}
#else
	MatrixT CholArraySymm;

	choleskyDecomposition(ArrSymm, CholArraySymm);
	unsigned m = CholArraySymm.size1();
	I.resize(m,m);
	I.clear();

	MatrixT CholArraySymmInv(m,m);
	CholArraySymmInv.clear();

	for(size_t j = 0; j < m; j++)
		CholArraySymmInv(j ,j) = 1/CholArraySymm(j,j);

	for(size_t j = 0; j < m; j++)
	{
		for(size_t i = j+1; i < m; i++)
		{
			double s = 0;
			for(size_t k = 0; k < i; k++)
			{
				s += CholArraySymm(i, k) * CholArraySymmInv(k, j);
			}
			CholArraySymmInv(i ,j) = -1/CholArraySymm(i, i)*s;
		}
	}
	//no atlas, no fast prod anyway...
	//fast_prod(trans(CholArraySymmInv), CholArraySymmInv,I,1.0);
	axpy_prod(trans(CholArraySymmInv), CholArraySymmInv,I,false);
#endif
}
 
 #endif
