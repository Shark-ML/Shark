/*!
 *  \author O. Krause
 *  \date 2012
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
#ifndef SHARK_LINALG_IMPL_NUMERIC_BINDINGS_DEFAULT_GEMV_H
#define SHARK_LINALG_IMPL_NUMERIC_BINDINGS_DEFAULT_GEMV_H

namespace shark {namespace blas {namespace bindings {

// y <- alpha * op (A) * x + beta * y
// op (A) == A || A^T || A^H
template <typename T, typename MatA, typename VectorB, typename VectorC>
void gemv(
T alpha, matrix_expression<MatA> const &matA, 
vector_expression<VectorB> const &vecB,
T beta, vector_expression<VectorC> &vecC
) {
	if(alpha != 1.0){
		beta /= alpha;
	}
	if ( beta != 1.0){
		vecC()*=beta;
	}

	if(!traits::isRowMajor(matA)){
		//there is no reasonable fast operation for A^Tx in uBLAS. yes, that is indeed scary
		for(std::size_t i = 0; i != matA().size2(); ++i){
			vecC()+=vecB()(i)*column(matA(),i);
		}
	}
	else{
		vecC()+=prod(matA,vecB);
	}
	
	if(alpha != 1.0){
		vecC() *= alpha;
	}
}

}}}
#endif
