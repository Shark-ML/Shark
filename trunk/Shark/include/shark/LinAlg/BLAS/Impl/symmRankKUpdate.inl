//===========================================================================
/*!
 *  \brief Implementation of the rank k update to a symmetric matrix C<-alpha * A * A^T + beta* C
 *
 *  \author  O.Krause
 *  \date    2011
 *
 *  \par Copyright (c) 1998-2011:
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
#ifndef SHARK_LINALG_IMPL_SYMM_RANK_K_UPDATE_INL
#define SHARK_LINALG_IMPL_SYMM_RANK_K_UPDATE_INL

#include <shark/LinAlg/BLAS/Tools.h>
#include "numeric_bindings/syrk.h"

template<class MatA,class MatC>
void shark::blas::symmRankKUpdate(
	matrix_expression<MatA> const & matA,
	matrix_expression<MatC>& matC,
bool beta,double alpha){
	if(!beta)
		zero(matC);
	bindings::syrk<false>(alpha,matA,1.0,matC);
	
	//reconstruct symmetric elements
	for(std::size_t i = 0; i != matC().size1(); ++i){
		for(std::size_t j = 0; j < i; ++j){
			matC()(i,j) = matC()(j,i);
		}
	}
}

//undocumented versions to prevent ublas annoyingness
namespace shark{ namespace blas{
template<class MatA,class MatC>
void symmRankKUpdate(
	matrix_expression<MatA> const & matA,
	matrix_range<MatC> matC,
bool beta = false,double alpha = 1.0){
	symmRankKUpdate(matA,static_cast<matrix_expression<matrix_range<MatC> >& >(matC),beta,alpha);
}
}}
#endif
