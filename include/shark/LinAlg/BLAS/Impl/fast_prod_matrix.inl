//===========================================================================
/*!
 *  \brief Implementation of the fast_prod a dynamic numeric binding to ATLAS gemm
 *
 *  \author  O.Krause
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
#ifndef SHARK_LINALG_BLAS_IMPL_FAST_PROD_MATRIX_INL
#define SHARK_LINALG_BLAS_IMPL_FAST_PROD_MATRIX_INL

#include <boost/mpl/or.hpp>
#include <shark/LinAlg/BLAS/Tools.h>
//#include <boost/numeric/ublas/operation_sparse.hpp>//sparse prod
#include <shark/LinAlg/BLAS/traits/matrix_raw.hpp>

#include "numeric_bindings/gemm.h"

namespace shark{ namespace blas{ namespace detail{

template<class MatA,class MatB,class MatC>
void fast_prod_dense(
	matrix_expression<MatA> const & matA,
	matrix_expression<MatB> const & matB,
	matrix_expression<MatC>& matC,
	bool beta,double alpha
){
	if(!beta)
		shark::blas::zero(matC);
	//at this point, c is not transposed(because we would have transposed it twice if it was)
	//now we need to ensure that all matrices are either row or column major. We do this by using
	//equivalent expressions. e.g. if a matrix is row major(and not transposed) transposing it is equivalent 
	//to making it column major.
	bool sameOrderA = traits::sameOrientation(matA,matC);
	bool sameOrderB = traits::sameOrientation(matB,matC);
	bool transA = traits::isTransposed(matA);
	bool transB = traits::isTransposed(matB);
	if(transA == sameOrderA){
		if(transB == sameOrderB){
			bindings::gemm(alpha, trans(matA), trans(matB), 1.0, matC);
		}
		else{
			bindings::gemm(alpha, trans(matA), matB, 1.0, matC);
		}
	}
	else{
		if(transB == sameOrderB){
			bindings::gemm(alpha, matA, trans(matB), 1.0, matC);
		}
		else{
			bindings::gemm(alpha, matA, matB, 1.0, matC);
		}
	}
}

//sparse implementation
//if A or B are sparse, we choose the sparse prod
template<class MatA,class MatB,class MatC>
void fast_prod_detail(
	matrix_expression<MatA> const & matA,
	matrix_expression<MatB> const & matB,
	matrix_expression<MatC>& matC,
	bool beta,double alpha,
	boost::mpl::true_
){
	if ( !beta ){
		shark::blas::zero(matC);
	}
	else if(alpha != 1.0){
		matC()/=alpha;
	}
	
	if(traits::isRowMajor(matA) && traits::isColumnMajor(matB)){
		matC()+=prod(matA,matB);
	}
	else
		axpy_prod(matA(),matB(),matC(),false);

	if(alpha != 1.0){
		matC() *= alpha;
	}
}

//nothing sparse here, use dense routines
template<class MatA,class MatB,class MatC>
void fast_prod_detail(
	matrix_expression<MatA> const & matA,
	matrix_expression<MatB> const & matB,
	matrix_expression<MatC>& matC,
	bool beta,double alpha,
	boost::mpl::false_
){
//	//if c is transposed, we add another layer of template madness and transpose the whole expression.
//	if(traits::isTransposed(matC)){
//		matrix_unary2<MatC, scalar_identity<typename MatC::value_type> > transC=trans(matC);
//		fast_prod_dense(trans(matB),trans(matA),transC,beta,alpha);
//	}
//	else{
		fast_prod_dense(matA,matB,matC,beta,alpha);
//	}
}

}}}

//dispatcher
template<class MatA,class MatB,class MatC>
void shark::blas::fast_prod(
	matrix_expression<MatA> const & matA,
	matrix_expression<MatB> const & matB,
	matrix_expression<MatC>& matC,
	bool beta,double alpha
){
	SIZE_CHECK(matB().size2()==matC().size2());
	SIZE_CHECK(matA().size1()==matC().size1());
	SIZE_CHECK(matA().size2()==matB().size1());
	
	//all matrices should be a real object not some lazy expression without storage behind it.
	//if this crashes and your input was sparse, and not a compressed vector: 
	//this is unsupported by shark. sorry.
	BOOST_STATIC_ASSERT(!traits::IsUnknownStorage<MatA>::value);
	BOOST_STATIC_ASSERT(!traits::IsUnknownStorage<MatB>::value);
	BOOST_STATIC_ASSERT(!traits::IsUnknownStorage<MatC>::value);
	//we need to dispach here since we want to call a special prod if A or B are sparse
	//IsSparse evaluates to true_ if matA or matB are sparse
	typedef typename boost::mpl::or_<
		traits::IsSparse<MatA>,
		traits::IsSparse<MatB>
	>::type IsSparse;
	
	detail::fast_prod_detail(matA(),matB(),matC(),beta,alpha,IsSparse() );
}
template<class MatA,class MatB,class MatC>
void shark::blas::fast_prod(
	matrix_expression<MatA> const & matA,
	matrix_expression<MatB> const & matB,
	matrix_range<MatC> matC,
bool beta,double alpha){
	typedef matrix_expression<matrix_range<MatC> > Base;
	fast_prod(matA,matB,static_cast< Base &>(matC),beta,alpha);
}
#endif
