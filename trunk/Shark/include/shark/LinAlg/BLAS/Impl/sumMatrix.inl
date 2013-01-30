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

namespace shark {
namespace detail{
template<std::size_t L2Size,class MatA,class VecB>
void sumRows(MatA const& matA, VecB& vecB, blas::column_major_tag){
	//typedef typename traits::PointerType<MatA const>::type PointerA;
	typedef typename MatA::value_type ValueA;
	typedef typename VecB::value_type ValueB;
	
	std::size_t n = matA.size1();
	std::size_t k = matA.size2();
	vecB.resize(k);
	
	std::size_t maxColumn=(k/4)*4;
	//std::size_t stride=traits::matrix_stride2(matA);
	
	//if the matrix is too big, we use the default algorithm
	//it must be possible to store the full matrix in L2 cache
	if(n*k*sizeof(ValueA) > L2Size){
		maxColumn = 0;
	}
	
//	//we need a case distinction here: if the columns are packed we can directly access 
//	// the memory behind it, else we need to do indexed access
//	std::size_t column=0;
//	if(traits::hasDenseLeadingDimension(matA)){
//		//calculate 4 columns at the same time
//		while(column!= maxColumn){
//			PointerA A0 = traits::matrix_storage(matA)+column*stride;
//			PointerA A1 = A0+stride;
//			PointerA A2 = A1+stride;
//			PointerA A3 = A2+stride;
//			ValueB b0=0,b1=0,b2=0,b3=0;
//			for(std::size_t i = 0; i != n; ++i){
//				b0+=A0[i];
//				b1+=A1[i];
//				b2+=A2[i];
//				b3+=A3[i];
//			}
//			vecB(column)+=b0;
//			vecB(column+1)+=b1;
//			vecB(column+2)+=b2;
//			vecB(column+3)+=b3;
//			column+=4;
//		}
//	}
//	else{
		//no pointer access
		std::size_t column=0;
		while(column!= maxColumn){
			ValueB b0=0,b1=0,b2=0,b3=0;
			for(std::size_t i = 0; i != n; ++i){
				b0+=matA(i,column);
				b1+=matA(i,column+1);
				b2+=matA(i,column+2);
				b3+=matA(i,column+3);
			}
			vecB(column)+=b0;
			vecB(column+1)+=b1;
			vecB(column+2)+=b2;
			vecB(column+3)+=b3;
			column+=4;
			
		}
//	}
	//calculate the rest
	while(column!=k){
		vecB(column)+=sum(blas::column(matA,column));
		++column;
	}
}
//default case: 
template<std::size_t L2Size,class MatA,class VecB,class Orientation>
//void sumRows(MatA const& matA, VecB& vecB, blas::row_major_tag){
void sumRows(MatA const& matA, VecB& vecB, Orientation){
	//typedef typename traits::PointerType<MatA const>::type PointerA;
	typedef typename MatA::value_type ValueA;
	
	std::size_t n = matA.size1();
	std::size_t k = matA.size2();
	vecB.resize(k);
	
	std::size_t maxRow=(n/4)*4;
//	std::size_t stride=traits::matrix_stride1(matA);
	
	//if the matrix is too big, we use the default algorithm
	//it must be possible to store the full matrix in L2 cache
	if(n*k*sizeof(ValueA) > L2Size){
		maxRow = 0;
	}
	
//	//we need a case distinction here: if the rows are dense, we can directly access 
//	//the memory behind it Else we need to do indexed access
//	std::size_t row=0;
//	if(traits::hasDenseLeadingDimension(matA)){
//		//calculate 4 rows at the same time until the remaining number of columns is lower than 4
//		while(row != maxRow){
//			PointerA A0 = traits::matrix_storage(matA)+row*stride;
//			PointerA A1 = A0+stride;
//			PointerA A2 = A1+stride;
//			PointerA A3 = A2+stride;
//			for(std::size_t i = 0; i != k; ++i){
//				vecB(i)+=A0[i]+A1[i]+A2[i]+A3[i];
//			}
//			row+=4;
//		}
//	}
//	else{
		//no pointer access
		std::size_t row=0;
		while(row!= maxRow){
			for(std::size_t i = 0; i != k; ++i){
				vecB(i)+=matA(row,i) + matA(row+1,i);
				vecB(i)+=matA(row+2,i) + matA(row+3,i);
			}
			row+=4;
		}
//	}
	//calculate the rest
	while(row != n){
		noalias(vecB)+=blas::row(matA,row);
		++row;
	}
}

}//end detail

//dispatcher
template<class MatA, class VecB>
void sumColumns(blas::matrix_expression<MatA> const& A, blas::vector_container<VecB>& b){
	typedef blas::matrix_unary2<MatA, blas::scalar_identity<typename MatA::value_type> > TransA;
	detail::sumRows<65536*64>(trans(A),b(),typename TransA::orientation_category());
}

//dispatcher
template<class MatA, class VecB>
void sumRows(blas::matrix_expression<MatA> const& A, blas::vector_container<VecB>& b){
	detail::sumRows<65536*64>(A(),b(),typename MatA::orientation_category());
}
}
#endif
