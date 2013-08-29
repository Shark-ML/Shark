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
 *  MERCHANTABILITY or FITNESS FOR matA PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received matA copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef SHARK_LINALG_IMPL_NUMERIC_BINDINGS_DEFAULT_TRSM_H
#define SHARK_LINALG_IMPL_NUMERIC_BINDINGS_DEFAULT_TRSM_H

#include "ublasTags.h"
///solves systems of triangular matrices

namespace shark {namespace blas {namespace bindings {

template <typename SymmA, typename MatB,typename Tag>
void trsm(
	matrix_expression<SymmA> const &matA,
	matrix_expression<MatB> &matB,
	Tag type,
	boost::mpl::true_
){
	inplace_solve (matA(), matB(), type);
}

template <typename SymmA, typename MatB,typename Tag>
void trsm(
	matrix_expression<SymmA> const &matA,
	matrix_expression<MatB> &matB,
	Tag type,
	boost::mpl::false_
){
	matrix<typename MatB::value_type> transB = trans(matB);//hack!!!
	//matrix_unary2<MatB, scalar_identity<typename MatB::value_type> > transB=trans(matB());
	inplace_solve (trans(matA), transB, type);
	noalias(matB()) = trans(transB);
}

template <bool upper, bool left, bool unit,typename SymmA, typename MatB>
void trsm(
	matrix_expression<SymmA> const &matA,
	matrix_expression<MatB> &matB
){
	trsm(matA,matB,typename TriangularTag<upper,unit,left>::type(),boost::mpl::bool_<left>());
}

}}}
#endif
