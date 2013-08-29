/**
*
*  \brief Optimized operations for Linear Algebra
*
*  \author O.Krause
*  \date 2011
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
#ifndef SHARK_LINALG_IMPL_SUMMATRIX_H
#define SHARK_LINALG_IMPL_SUMMATRIX_H

//included by ../fastOperations.h

namespace shark {namespace blas{
namespace detail{
template<class MatA,class VecB>
void sumRowsImpl(MatA const& matA, VecB& vecB, column_major_tag){
	vecB.resize(matA.size2());
	zero(vecB);
	for(std::size_t i = 0; i != matA.size2(); ++i){ 
		vecB(i)+=sum(column(matA,i));
	}
}

template<class MatA,class VecB>
void sumRowsImpl(MatA const& matA, VecB& vecB, row_major_tag){
	vecB.resize(matA.size2());
	zero(vecB);
	for(std::size_t i = 0; i != matA.size1(); ++i){ 
		noalias(vecB) += row(matA,i);
	}
}

}//end detail

//dispatcher
template<class MatA, class VecB>
void sumColumns(matrix_expression<MatA> const& A, vector_container<VecB>& b){
	typedef matrix_unary2<MatA, scalar_identity<typename MatA::value_type> > TransA;
	detail::sumRowsImpl(trans(A),b(),typename TransA::orientation_category());
}

//dispatcher
template<class MatA, class VecB>
void sumRows(matrix_expression<MatA> const& A, vector_container<VecB>& b){
	detail::sumRowsImpl(A(),b(),typename MatA::orientation_category());
}

}}
#endif
