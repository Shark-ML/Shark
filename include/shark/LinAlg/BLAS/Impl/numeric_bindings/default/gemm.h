/*!
 *  \author O. Krause
 *  \date 2010
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

#ifndef SHARK_LINALG_IMPL_NUMERIC_BINDINGS_DEFAULT_GEMM_HPP
#define SHARK_LINALG_IMPL_NUMERIC_BINDINGS_DEFAULT_GEMM_HPP

namespace shark { namespace blas { namespace bindings {

// C <- alpha * op (A) * op (B) + beta * C
// op (A) == A || A^T || A^H
template <typename T, typename MatA, typename MatB, typename MatC>
void gemm(
T alpha, matrix_expression<MatA> const &matA, 
matrix_expression<MatB> const &matB,
T beta, 
matrix_expression<MatC>& matC
) {
	if(alpha != 1.0){
		beta /= alpha;
	}
	if ( beta != 1.0){
		matC()*=beta;
	}
	if(traits::isRowMajor(matB)){
		axpy_prod(matA(),matB(),matC(),false);
	}
	else{
		if(traits::isRowMajor(matA)){
			noalias(matC())+=prod(matA,matB);	
		}else{
			if(matA().size2()> 100){
				matrix<typename MatB::value_type, row_major> temp=matA;
				noalias(matC())+=prod(temp,matB);	
			}
			else
			{
				axpy_prod(matA(),matB(),matC(),false);
			}
		}
	}
	if(alpha != 1.0){
		matC() *= alpha;
	}
}

}}}

#endif
