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


/*
BE AWARE THAT THE METHODS USED HERE ARE WORK IN PROGRESS AND
ARE NOT GUARANTEED TO WORK PROPERLY!
*/
#ifndef SHARK_LINALG_BLAS_IMPL_BLOCKMATRIXVECTOROPERATION_INL
#define SHARK_LINALG_BLAS_IMPL_BLOCKMATRIXVECTOROPERATION_INL

#include <shark/LinAlg/BLAS/ublas.h>
#include <shark/LinAlg/BLAS/traits/matrix_raw.hpp>
#include <shark/LinAlg/BLAS/traits/vector_raw.hpp>
namespace shark {
namespace blas{
namespace detail{
	
//first the all-dense-version

///\brief General matrix-vector multiplication like operation.
///
///The usual scalar multiplication x*y is here replaced by a general kernel, which enables
///the computation of more complex operations than the matrix-vector product. However
///this is not as optimized as the fast_prod for the matrix-vector product, at least when ATLAS is used!
///this is the version for column major matrices. here we compute 4 columns at the same time. This saves
///memory lookup of a factor of 4.
template<class MatA,class VecB,class VecC,class ComputeKernel>
void generalMatrixVectorOperation(
	MatA const & matA,
	VecB const & vecB,
	VecC& vecC,
	ComputeKernel kernel,
	column_major_tag,
	dense_proxy_tag,
	dense_proxy_tag
){
	typedef typename traits::PointerType<MatA const>::type PointerA;
	typedef typename VecB::value_type ValueB;
	
	std::size_t n = matA.size1();
	std::size_t k = matA.size2();
	SIZE_CHECK( k == vecB.size());
	SIZE_CHECK( n == vecC.size());
	
	std::size_t maxColumn=(k/4)*4;
	std::size_t stride=traits::matrix_stride2(matA);
	
	//we need a case distinction here: if the columns are packed we can directly access 
	// the memory behind it Else we need to do indexed access
	if(traits::hasDenseLeadingDimension(matA)){
		//calculate 4 columns at the same time
		std::size_t column=0;
		while(column!= maxColumn){
			PointerA A0 = traits::matrix_storage(matA)+column*stride;
			PointerA A1 = A0+stride;
			PointerA A2 = A0+2*stride;
			PointerA A3 = A0+3*stride;
			ValueB b0=vecB(column);
			ValueB b1=vecB(column+1);
			ValueB b2=vecB(column+2);
			ValueB b3=vecB(column+3);
			for(std::size_t i = 0; i != n; ++i){
				vecC[i]+=kernel(A0[i],b0);
				vecC[i]+=kernel(A1[i],b1);
				vecC[i]+=kernel(A2[i],b2);
				vecC[i]+=kernel(A3[i],b3);
			}
			column+=4;
		}
		//calculate the rest
		for(;column!=k; ++column){
			PointerA A = traits::matrix_storage(matA)+column*stride;
			ValueB b=vecB(column);
			for(std::size_t i = 0; i != n; ++i){
				vecC[i]+=kernel(A[i],b);
			}
		}
	}
	else
	{
		//no pointer access
		std::size_t column=0;
		while(column!= maxColumn){
			ValueB b0=vecB(column);
			ValueB b1=vecB(column+1);
			ValueB b2=vecB(column+2);
			ValueB b3=vecB(column+3);
			for(std::size_t i = 0; i != n; ++i){
				vecC[i]+=kernel(matA(i,column),b0);
				vecC[i]+=kernel(matA(i,column+1),b1);
				vecC[i]+=kernel(matA(i,column+2),b2);
				vecC[i]+=kernel(matA(i,column+3),b3);
			}
			column+=4;
		}
		//calculate the rest
		for(;column!=k; ++column){
			ValueB b=vecB(column);
			for(std::size_t i = 0; i != n; ++i){
				vecC[i]+=kernel(matA(i,column),b);
			}
		}
	}
}

///\brief General matrix-vector multiplication like operation.
///
///The usual scalar multiplication x*y is here replaced by a general kernel, which enables
///the computation of more complex operations than the matrix-vector product. However
///this is not as optimized as the fast_prod for the matrix-vector product, at least when ATLAS is used!
///this is the version for row major matrices. here we compute 4 rows at the same time. This saves
///memory lookup of a factor of 4.
template<class MatA,class VecB,class VecC,class ComputeKernel>
void generalMatrixVectorOperation(
	MatA const & matA,
	VecB const & vecB,
	VecC& vecC,
	ComputeKernel kernel,
	row_major_tag,
	dense_proxy_tag,
	dense_proxy_tag
){
	typedef typename traits::PointerType<MatA const>::type PointerA;
	typedef typename traits::PointerType<VecB const>::type PointerB;
	typedef typename VecC::value_type ValueC;
	
	std::size_t n = matA.size1();
	std::size_t k = matA.size2();
	SIZE_CHECK( k == vecB.size());
	SIZE_CHECK( n == vecC.size());
	
	std::size_t maxRow=(n/4)*4;
	std::size_t stride=traits::matrix_stride1(matA);
	
	//we need a case distinction here: if the rows are dense, we can directly access 
	//the memory behind it Else we need to do indexed access
	if(traits::hasDenseLeadingDimension(matA)){
		//calculate 4 rows at the same time until the remaining number of columns is lower than 4
		std::size_t row=0;
		std::size_t strideB=traits::vector_stride(vecB);
		PointerB B=traits::vector_storage(vecB);
		while(row!= maxRow){
			PointerA A0 = traits::matrix_storage(matA)+row*stride;
			PointerA A1 = A0+stride;
			PointerA A2 = A0+2*stride;
			PointerA A3 = A0+3*stride;
			ValueC c0=0;
			ValueC c1=0;
			ValueC c2=0;
			ValueC c3=0;
			for(std::size_t i = 0; i != k; ++i){
				c0+=kernel(A0[i],B[strideB*i]);
				c1+=kernel(A1[i],B[strideB*i]);
				c2+=kernel(A2[i],B[strideB*i]);
				c3+=kernel(A3[i],B[strideB*i]);
			}
			vecC(row)+=c0;
			vecC(row+1)+=c1;
			vecC(row+2)+=c2;
			vecC(row+3)+=c3;
			row+=4;
		}
		//calculate the rest
		for(;row!=n; ++row){
			PointerA A = traits::matrix_storage(matA)+row*k;
			ValueC c0=0;
			for(std::size_t i = 0; i != k; ++i){
				c0+=kernel(A[i],vecB(i));
			}
			vecC(row)+=c0;
		}
	}
	else
	{
		//no pointer access
		std::size_t row=0;
		while(row!= maxRow){
			ValueC c0=0;
			ValueC c1=0;
			ValueC c2=0;
			ValueC c3=0;
			for(std::size_t i = 0; i != k; ++i){
				c0+=kernel(matA(row,i),vecB(i));
				c1+=kernel(matA(row+1,i),vecB(i));
				c2+=kernel(matA(row+2,i),vecB(i));
				c3+=kernel(matA(row+3,i),vecB(i));
			}
			vecC(row)+=c0;
			vecC(row+1)+=c1;
			vecC(row+2)+=c2;
			vecC(row+3)+=c3;
			row+=4;
		}
		//calculate the rest
		for(;row!=n; ++row){
			ValueC c0=0;
			for(std::size_t i = 0; i != k; ++i){
				c0+=kernel(matA(row,i),vecB(i));
			}
			vecC(row)+=c0;
		}
	}
}
}//end namespace detail
}
}
#endif
