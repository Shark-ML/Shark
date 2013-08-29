//===========================================================================
/*!
 *  \brief Implementation of the fast_prod a dynamic numeric binding to ATLAS gemv
 *
 *  \author  O.Krause
 *  \date    2012
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
#ifndef SHARK_LINALG_IMPL_FAST_PROD_VECTOR_INL
#define SHARK_LINALG_IMPL_FAST_PROD_VECTOR_INL

#include <boost/mpl/or.hpp>
#include <shark/LinAlg/BLAS/Tools.h>
#include <shark/LinAlg/BLAS/traits/matrix_raw.hpp>
#include "numeric_bindings/gemv.h"

namespace shark{ namespace blas{ namespace detail{

//is called based on the sparse/not sparse basis dispatcher when the arguments are sparse
template<class MatA,class VecB,class VecC>
void fast_prod_impl(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	vector_expression<VecC>& vecC,
	bool beta,double alpha,boost::mpl::true_
){
	if(!beta){
		shark::blas::zero(vecC);
		axpy_prod(matA(),vecB(),vecC(),true);
	}else{
		axpy_prod(matA(),vecB(),vecC(),false);
	}
	if(alpha != 1.0){
		vecC() *= alpha;
	}
}
//is called based on the sparse/not sparse basis dispatcher when the arguments are dense
template<class MatA,class VecB,class VecC>
void fast_prod_impl(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	vector_expression<VecC>& vecC,
	bool beta,double alpha,boost::mpl::false_
){
	if(!beta)
		shark::blas::zero(vecC);
	
	bindings::gemv(alpha, matA, vecB, 1.0, vecC);
}

}}}


//dispatches sparse/non-sparse
template<class MatA,class VecB,class VecC>
void shark::blas::fast_prod(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	vector_expression<VecC>& vecC,
	bool beta,double alpha)
{
	SIZE_CHECK(matA().size2()==vecB().size());
	SIZE_CHECK(matA().size1()==vecC().size());
	//we need to distpach here since we want to call a special prod if A or b are sparse
	//IsDense evaluates to true_ if matA and vecB are dense
	typedef typename boost::mpl::or_<
		traits::IsSparse<MatA>,
		traits::IsSparse<VecB>
	>::type IsSparse;
	
	detail::fast_prod_impl(matA(),vecB(),vecC(),beta,alpha,IsSparse());
}
//dispatcher for subranges
template<class MatA,class VecB,class VecC>
void shark::blas::fast_prod(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	vector_range<VecC> vecC,
	bool beta,double alpha)
{
	//just call the other version, by explicitly converting to the vector_expression
	fast_prod(matA(),vecB(),
	static_cast<vector_expression<vector_range<VecC> >& >(vecC),
	beta,alpha);
}
//dispatcher for matrix_rows
template<class MatA,class VecB,class MatC>
void shark::blas::fast_prod(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	matrix_row<MatC> vecC,
	bool beta,double alpha)
{
	//just call the other version, by explicitly converting to the vector_expression
	fast_prod(matA(),vecB(),
	static_cast<vector_expression<matrix_row<MatC> >& >(vecC),
	beta,alpha);
}
//dispatcher for matrix_columns
template<class MatA,class VecB,class MatC>
void shark::blas::fast_prod(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	matrix_column<MatC> vecC,
	bool beta,double alpha)
{
	//just call the other version, by explicitly converting to the vector_expression
	fast_prod(matA(),vecB(),
	static_cast<vector_expression<matrix_column<MatC> >& >(vecC),
	beta,alpha);
}

///\brief Fast matrix/vector product.
///
///Computes c= alpha* b^TA +beta*c.
template<class MatA,class VecB,class VecC>
void shark::blas::fast_prod(
	vector_expression<VecB> const & vecB,
	matrix_expression<MatA> const & matA,
	vector_expression<VecC>& vecC,
bool beta,double alpha){
	fast_prod(trans(matA),vecB,vecC,beta,alpha);
}

///\brief Fast matrix/vector product.
///
///Computes c= alpha* b^TA +beta*c.
template<class MatA,class VecB,class VecC>
void shark::blas::fast_prod(
	vector_expression<VecB> const & vecB,
	matrix_expression<MatA> const & matA,
	vector_range<VecC>& vecC,
bool beta,double alpha){
	fast_prod(vecB(),matA(),
	static_cast<vector_expression<vector_range<VecC> >& >(vecC),
	beta,alpha);
}

///\brief Fast matrix/vector product.
///
///Computes c= alpha* b^TA +beta*c.
template<class MatA,class VecB,class VecC>
void shark::blas::fast_prod(
	vector_expression<VecB> const & vecB,
	matrix_expression<MatA> const & matA,
	matrix_row<VecC>& vecC,
bool beta,double alpha){
	fast_prod(vecB(),matA(),
	static_cast<vector_expression<matrix_row<VecC> >& >(vecC),
	beta,alpha);
}

///\brief Fast matrix/vector product.
///
///Computes c= alpha* b^TA +beta*c.
template<class MatA,class VecB,class VecC>
void shark::blas::fast_prod(
	vector_expression<VecB> const & vecB,
	matrix_expression<MatA> const & matA,
	matrix_column<VecC>& vecC,
bool beta,double alpha){
	fast_prod(vecB(),matA(),
	static_cast<vector_expression<matrix_column<VecC> >& >(vecC),
	beta,alpha);
}

#endif
