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
BE AWARE THAT THE METHODS USED HERE ARE WORK IN PROGFRESS AND
ARE NOT GUARANTEED TO WORK PROPERLY!
*/
#ifndef SHARK_LINALG_BLAS_BLOCK_PRODUCTS_H
#define SHARK_LINALG_BLAS_BLOCK_PRODUCTS_H

#include "Impl/BlockMatrixVectorOperation.inl"
namespace shark {
namespace detail{

///\brief Optimized implementation  of a matrix-matrix operation for a small block matrix (blockStorage) and a bigger Panel matrix (matA).
///
///A Panel matrix is a matrix with (in this case) much more rows than columns.
///This function is the main building block for fast bigger matrix matrix operations. 
///In the case of matrix multiplication, it multiplies a block matrix with a panel matrix (currently only a row vector).
///as A*trans(blockStorage). It is assumed that all matrices are row major and packed. Also blockStorage should fit into L1 cache.
///\par
///The compute kernel performs an operation for two elements a_i b_j as kernel(a_i,b_j).
///The kernel must be very small so that it can be inlined. 
///some kernels which are possible:
///kernel(a_i,b_j) = a_i*b_j(result in matrix matrix multiplication)
///kernel(a_i,b_j) = (a_i-b_j)^2 (results in kernel gram matrix for the squared euclidean kernel)
template<class MatA,class MatB,class MatC,class ComputeKernel>
void generalPanelBlockOperation(
	MatA const & matA,
	MatB const & blockStorage,
	blas::matrix_expression<MatC>& matC,
	ComputeKernel kernel
){
	//ensure, that the Matrix is row_major
	BOOST_STATIC_ASSERT(
	(boost::is_same<
		typename MatB::orientation_category,
		blas::row_major_tag
	>::value));
	
	SIZE_CHECK(blockStorage.size1() == matC().size2());
	SIZE_CHECK(blockStorage.size2() == matA.size2());
	std::size_t k = blockStorage.size2();
	
	//version 2x2
	std::size_t maxK= (k/4)*4;
	std::size_t maxRows=(matA.size1()/2)*2;
	std::size_t maxColumns=(matC().size2()/2)*2;
	
	//vc++ has problems optimizing the direct accessors
	double const* A=&(matA(0,0));
	double const* B=&blockStorage(0,0);
	
	std::size_t strideA=matA.size2();
	std::size_t strideB=blockStorage.size2();
	for(std::size_t row = 0; row != maxRows; row+=2){
		double const* Arow=A+row*strideA;
		double const* ArowNext=Arow+strideA;
		for(std::size_t columnC = 0; columnC != maxColumns; columnC+=2){
			double const* Brow=B+columnC*strideB;
			double const* BrowNext=Brow+strideB;
			
			typename MatC::value_type sum11=0;
			typename MatC::value_type sum12=0;
			typename MatC::value_type sum21=0;
			typename MatC::value_type sum22=0;
			
			for(std::size_t j = 0; j != maxK;j+=4){
				//first peeling
				sum11+=kernel(Arow[j],Brow[j]);
				sum11+=kernel(Arow[j+1],Brow[j+1]);
				sum12+=kernel(Arow[j],BrowNext[j]);
				sum12+=kernel(Arow[j+1],BrowNext[j+1]);
				
				sum11+=kernel(Arow[j+2],Brow[j+2]);
				sum11+=kernel(Arow[j+3],Brow[j+3]);
				sum12+=kernel(Arow[j+2],BrowNext[j+2]);
				sum12+=kernel(Arow[j+3],BrowNext[j+3]);
				
				sum21+=kernel(ArowNext[j],Brow[j]);
				sum21+=kernel(ArowNext[j+1],Brow[j+1]);
				sum22+=kernel(ArowNext[j],BrowNext[j]);
				sum22+=kernel(ArowNext[j+1],BrowNext[j+1]);
				
				sum21+=kernel(ArowNext[j+2],Brow[j+2]);
				sum21+=kernel(ArowNext[j+3],Brow[j+3]);
				sum22+=kernel(ArowNext[j+2],BrowNext[j+2]);
				sum22+=kernel(ArowNext[j+3],BrowNext[j+3]);
			}
			//the number of k is not a multiple of 4 and so we must peel the last layer separately
			if(maxK!=k){
				for(std::size_t j = maxK; j != k;j++){
					sum11+=kernel(Arow[j],Brow[j]);
					sum12+=kernel(Arow[j],BrowNext[j]);
					sum21+=kernel(ArowNext[j],Brow[j]);
					sum22+=kernel(ArowNext[j],BrowNext[j]);
				}
			}
			matC()(row,columnC)+=sum11;
			matC()(row,columnC+1)+=sum12;
			matC()(row+1,columnC)+=sum21;
			matC()(row+1,columnC+1)+=sum22;
		}
		//the number of columns is uneven and so we must peel the last layer separately
		if(maxColumns!=matC().size2()){
			std::size_t col=matC().size2()-1;
			typename MatC::value_type sum11=0;
			typename MatC::value_type sum21=0;
			
			for(std::size_t j = 0; j != maxK;j+=4){
				sum11+=kernel(matA(row,j),blockStorage(col,j));
				sum11+=kernel(matA(row,j+1),blockStorage(col,j+1));
				sum11+=kernel(matA(row,j+2),blockStorage(col,j+2));
				sum11+=kernel(matA(row,j+3),blockStorage(col,j+3));
				
				sum21+=kernel(matA(row+1,j),blockStorage(col,j));
				sum21+=kernel(matA(row+1,j+1),blockStorage(col,j+1));
				sum21+=kernel(matA(row+1,j+2),blockStorage(col,j+2));
				sum21+=kernel(matA(row+1,j+3),blockStorage(col,j+3));
				
			}
			if(maxK != k){
				for(std::size_t j = maxK; j != k;j++){
					sum11+=kernel(matA(row,j),blockStorage(col,j));
					sum21+=kernel(matA(row+1,j),blockStorage(col,j));
				}
			}
			matC()(row,col)+=sum11;
			matC()(row+1,col)+=sum21;
		}
	}
	if(maxRows == matA.size1())
		return;
	//the number of rows is uneven, so we also have to peal this layer and repeat everything...
	std::size_t row=matA.size1()-1;
	for(std::size_t columnC = 0; columnC != maxColumns; columnC+=2){
		typename MatC::value_type sum11=0;
		typename MatC::value_type sum12=0;
		
		for(std::size_t j = 0; j != maxK;j+=4){
			sum11+=kernel(matA(row,j),blockStorage(columnC,j));
			sum11+=kernel(matA(row,j+1),blockStorage(columnC,j+1));
			sum11+=kernel(matA(row,j+2),blockStorage(columnC,j+2));
			sum11+=kernel(matA(row,j+3),blockStorage(columnC,j+3));
			
			sum12+=kernel(matA(row,j),blockStorage(columnC+1,j));
			sum12+=kernel(matA(row,j+1),blockStorage(columnC+1,j+1));
			sum12+=kernel(matA(row,j+2),blockStorage(columnC+1,j+2));
			sum12+=kernel(matA(row,j+3),blockStorage(columnC+1,j+3));
		}
		if(maxK != k){
			for(std::size_t j = maxK; j != k;j++){
				sum11+=kernel(matA(row,j),blockStorage(columnC,j));
				sum12+=kernel(matA(row,j),blockStorage(columnC+1,j));
			}
			std::size_t lastK=k-1;
			sum11+=kernel(matA(row,lastK),blockStorage(columnC,lastK));
			sum12+=kernel(matA(row,lastK),blockStorage(columnC+1,lastK));
		}
		matC()(row,columnC)+=sum11;
		matC()(row,columnC+1)+=sum12;
	}
	if(maxColumns!=matC().size2()){
		std::size_t col=matC().size2()-1;
		typename MatC::value_type sum11=0;
		
		for(std::size_t j = 0; j != maxK;j+=4){
			sum11+=kernel(matA(row,j),blockStorage(col,j));
			sum11+=kernel(matA(row,j+1),blockStorage(col,j+1));
			sum11+=kernel(matA(row,j+2),blockStorage(col,j+2));
			sum11+=kernel(matA(row,j+3),blockStorage(col,j+3));
		}
		if(maxK != k){
			for(std::size_t j = maxK; j != k;j++){
				sum11+=kernel(matA(row,j),blockStorage(col,j));
			}
		}
		matC()(row,col)+=sum11;
	}
}
//~ template<class MatA,class MatB,class MatC,class ComputeKernel>
//~ void generalPanelBlockOperation(
	//~ MatA const & matA,
	//~ MatB const & blockStorage,
	//~ blas::matrix_expression<MatC>& matC,
	//~ ComputeKernel kernel
//~ ){
	//~ //ensure, that the Matrix is row_major
	//~ BOOST_STATIC_ASSERT(
	//~ (boost::is_same<
		//~ typename MatB::orientation_category,
		//~ blas::row_major_tag
	//~ >::value));
	
	//~ SIZE_CHECK(blockStorage.size1() == matC().size2());
	//~ SIZE_CHECK(blockStorage.size2() == matA.size2());
	//~ std::size_t k = blockStorage.size2();
	
	//~ //version 2x2
	//~ std::size_t maxK= (k/2)*2;
	//~ std::size_t maxRows=(matA.size1()/2)*2;
	//~ std::size_t maxColumns=(matC().size2()/2)*2;
	
	//~ //vc++ has problems optimizing the direct accessors
	//~ double const* A=&(matA(0,0));
	//~ double const* B=&blockStorage(0,0);
	
	//~ std::size_t strideA=matA.size2();
	//~ std::size_t strideB=blockStorage.size2();
	//~ for(std::size_t row = 0; row != maxRows; row+=2){
		//~ double const* Arow=A+row*strideA;
		//~ double const* ArowNext=Arow+strideA;
		//~ for(std::size_t columnC = 0; columnC != maxColumns; columnC+=2){
			//~ double const* Brow=B+columnC*strideB;
			//~ double const* BrowNext=Brow+strideB;
			
			//~ typename MatC::value_type sum11=0;
			//~ typename MatC::value_type sum12=0;
			//~ typename MatC::value_type sum21=0;
			//~ typename MatC::value_type sum22=0;
			
			//~ for(std::size_t j = 0; j != maxK;j+=2){
				//~ //sum11+=kernel(matA(row,j),blockStorage(columnC,j));
				//~ //sum11+=kernel(matA(row,j+1),blockStorage(columnC,j+1));
				
				//~ //sum12+=kernel(matA(row,j),blockStorage(columnC+1,j));
				//~ //sum12+=kernel(matA(row,j+1),blockStorage(columnC+1,j+1));
				
				//~ //sum21+=kernel(matA(row+1,j),blockStorage(columnC,j));
				//~ //sum21+=kernel(matA(row+1,j+1),blockStorage(columnC,j+1));
				
				//~ //sum22+=kernel(matA(row+1,j),blockStorage(columnC+1,j));
				//~ //sum22+=kernel(matA(row+1,j+1),blockStorage(columnC+1,j+1));
				
				//~ sum11+=kernel(Arow[j],Brow[j]);
				//~ sum11+=kernel(Arow[j+1],Brow[j+1]);
				//~ sum12+=kernel(Arow[j],BrowNext[j]);
				//~ sum12+=kernel(Arow[j+1],BrowNext[j+1]);
				
				//~ sum21+=kernel(ArowNext[j],Brow[j]);
				//~ sum21+=kernel(ArowNext[j+1],Brow[j+1]);
				//~ sum22+=kernel(ArowNext[j],BrowNext[j]);
				//~ sum22+=kernel(ArowNext[j+1],BrowNext[j+1]);
				
			//~ }
			//~ //the number of k's is uneven and so we must peel the last layer separately
			//~ if(maxK!=k){
				//~ std::size_t lastK=k-1;
				//~ sum11+=kernel(matA(row,lastK),blockStorage(columnC,lastK));
				//~ sum12+=kernel(matA(row,lastK),blockStorage(columnC+1,lastK));
				//~ sum21+=kernel(matA(row+1,lastK),blockStorage(columnC,lastK));
				//~ sum22+=kernel(matA(row+1,lastK),blockStorage(columnC+1,lastK));
			//~ }
			//~ matC()(row,columnC)+=sum11;
			//~ matC()(row,columnC+1)+=sum12;
			//~ matC()(row+1,columnC)+=sum21;
			//~ matC()(row+1,columnC+1)+=sum22;
		//~ }
		//~ //the number of columns is uneven and so we must peel the last layer separately
		//~ if(maxColumns!=matC().size2()){
			//~ std::size_t col=matC().size2()-1;
			//~ typename MatC::value_type sum11=0;
			//~ typename MatC::value_type sum21=0;
			
			//~ for(std::size_t j = 0; j != maxK;j+=2){
				//~ sum11+=kernel(matA(row,j),blockStorage(col,j));
				//~ sum11+=kernel(matA(row,j+1),blockStorage(col,j+1));
				
				//~ sum21+=kernel(matA(row+1,j),blockStorage(col,j));
				//~ sum21+=kernel(matA(row+1,j+1),blockStorage(col,j+1));
				
			//~ }
			//~ if(maxK != k){
				//~ std::size_t lastK=k-1;
				//~ sum11+=kernel(matA(row,lastK),blockStorage(col,lastK));
				//~ sum21+=kernel(matA(row+1,lastK),blockStorage(col,lastK));
			//~ }
			//~ matC()(row,col)+=sum11;
			//~ matC()(row+1,col)+=sum21;
		//~ }
	//~ }
	//~ if(maxRows == matA.size1())
		//~ return;
	//~ //the number of rows is uneven, so we also have to peal this layer and repeat everything...
	//~ std::size_t row=matA.size1()-1;
	//~ for(std::size_t columnC = 0; columnC != maxColumns; columnC+=2){
		//~ typename MatC::value_type sum11=0;
		//~ typename MatC::value_type sum12=0;
		
		//~ for(std::size_t j = 0; j != maxK;j+=2){
			//~ sum11+=kernel(matA(row,j),blockStorage(columnC,j));
			//~ sum11+=kernel(matA(row,j+1),blockStorage(columnC,j+1));
			
			//~ sum12+=kernel(matA(row,j),blockStorage(columnC+1,j));
			//~ sum12+=kernel(matA(row,j+1),blockStorage(columnC+1,j+1));
		//~ }
		//~ if(maxK != k){
			//~ std::size_t lastK=k-1;
			//~ sum11+=kernel(matA(row,lastK),blockStorage(columnC,lastK));
			//~ sum12+=kernel(matA(row,lastK),blockStorage(columnC+1,lastK));
		//~ }
		//~ matC()(row,columnC)+=sum11;
		//~ matC()(row,columnC+1)+=sum12;
	//~ }
	//~ if(maxColumns!=matC().size2()){
		//~ std::size_t col=matC().size2()-1;
		//~ typename MatC::value_type sum11=0;
		
		//~ for(std::size_t j = 0; j != maxK;j+=2){
			//~ sum11+=kernel(matA(row,j),blockStorage(col,j));
			//~ sum11+=kernel(matA(row,j+1),blockStorage(col,j+1));
		//~ }
		//~ if(maxK != k){
			//~ std::size_t lastK=k-1;
			//~ sum11+=kernel(matA(row,lastK),blockStorage(col,lastK));
		//~ }
		//~ matC()(row,col)+=sum11;
	//~ }
//~ }
///\brief Optimized implementation  of a vector matrix operations for a small block matrix (blockStorage) and a vector
///
///This function is the main building block for fast bigger matrix matrix operations. 
///In the case of matrix multiplication, it multiplies a block matrix with a panel matrix (currently only a row vector).
///as A*trans(blockStorage). It is assumed that all matrices are column major and packed. Also blockStorage should fit into L1 cache.
///\par
///The compute kernel performs an operation for two elements a_i b_j as kernel(a_i,b_j).
///The kernel must be very small so that it can be inlined. 
///some kernels which are possible:
///kernel(a_i,b_j) = a_i*b_j(result in matrix matrix multiplication)
///kernel(a_i,b_j) = (a_i-b_j)^2 (results in kernel gram matrix for the squared euclidean kernel)
template<class MatA,class MatB,class MatC,class ComputeKernel>
void generalBlockPanelOperation(
	MatA const & blockStorage,
	MatB const & matB,
	blas::matrix_expression<MatC>& matC,
	ComputeKernel kernel
){
	//still unoptimized, but that's not so relevant, since propably unused
	//ensure, that the Matrix is column_major
	BOOST_STATIC_ASSERT(
	(boost::is_same<
		typename MatA::orientation_category,
		blas::column_major_tag
	>::value));
	
	SIZE_CHECK(blockStorage.size2() == matC().size1());
	SIZE_CHECK(blockStorage.size1() == matB.size1());
	std::size_t k = blockStorage.size1();
	
	
	for(std::size_t columnC = 0; columnC != matC().size2(); ++columnC){
		for(std::size_t rowC = 0; rowC != matC().size1(); ++rowC){
			typename MatC::value_type sum=0;
			for(std::size_t j = 0; j != k; ++j){
				sum+=kernel(blockStorage(j,rowC),matB(j,columnC));
			}
			matC()(rowC,columnC)+=sum;
		}
	}
}

//now handle submatrices
template<class MatA,class MatB,class MatC,class ComputeKernel>
inline void generalPanelBlockOperation(
	blas::matrix_expression<MatA> const & matA,
	blas::matrix_expression<MatB> const & blockStorage,
	blas::matrix_range<MatC> matC,
	ComputeKernel kernel
){
	typedef blas::matrix_expression<blas::matrix_range<MatC> > super;
	generalPanelBlockOperation(matA(),blockStorage(),static_cast<super&>(matC),kernel);
}
template<class MatA,class MatB,class MatC,class ComputeKernel>
inline void generalBlockPanelOperation(
	blas::matrix_expression<MatA> const & blockStorage,
	blas::matrix_expression<MatB> const & matB,
	blas::matrix_range<MatC> matC,
	ComputeKernel kernel
){
	typedef blas::matrix_expression<blas::matrix_range<MatC> > super;
	generalBlockPanelOperation(blockStorage(),matB(),static_cast<super&>(matC),kernel);
}


template<class MatA,class MatB,class MatC,class ComputeKernel>
void generalPanelPanelOperation(
	MatA const & matA,
	MatB const & matB,
	MatC& matC,
	ComputeKernel kernel,
	blas::row_major_tag
){
	typedef typename MatC::value_type value_type;
	typedef blas::matrix<value_type,blas::row_major> StorageMatrix;
	
	std::size_t k = matA.size2();
	std::size_t m = matB.size2();
	std::size_t n = matA.size1();
	SIZE_CHECK( k == matB.size1());
	SIZE_CHECK( matC.size1() == matA.size1());
	SIZE_CHECK( matC.size2() == matB.size2());
	
	//test whether A is a matrix from real storage, if it is not, copy it and call the function again with the copy
	if(!traits::hasStorage(matA)){
		StorageMatrix packedA = matA;
		generalPanelPanelOperation(packedA, matB, matC, kernel, blas::row_major_tag());
		return;
	}
	
	StorageMatrix blockStorage(k, k);
	for(std::size_t beginBlock = 0; beginBlock < m; beginBlock+=k){
		std::size_t currentK=std::min(m - beginBlock,k);
		std::size_t endBlock=beginBlock + currentK;
		
		//pack a Block of matrix B for computation
		blockStorage.resize(currentK, k);
		noalias(blockStorage)=trans(subrange(matB, 0, k, beginBlock, endBlock));
		//evaluate the block operation (GEPB_OPT1 in the algorithm)
		for(std::size_t rowC = 0; rowC < n; ++rowC){
			detail::generalPanelBlockOperation(
				subrange(matA, rowC ,rowC + 1, 0, k),
				blockStorage,
				subrange(matC, rowC, rowC + 1, beginBlock, endBlock),
				kernel
			);
		}
	}
}
///\brief performs matrix-matrix multiplication like operations on 2 Panel matrices
///
///algorithm as described in Gotos paper "Anatomy of High-Performance Matrix Multiplication"
///this is the version for column-major matrices. Goto described it in Fig10 of his algorithm.
///we generalize it here for arbitrary operations instead of the multiplication. It is also not
///really optimized to the bitter end, but at least factor 2 faster, than straight forward code.
template<class MatA,class MatB,class MatC,class ComputeKernel>
void generalPanelPanelOperation(
	MatA const & matA,
	MatB const & matB,
	MatC& matC,
	ComputeKernel kernel,
	blas::column_major_tag
){
	typedef typename MatC::value_type value_type;
	typedef blas::matrix<value_type,blas::column_major> StorageMatrix;
	
	//test whether B is a matrix from real storage, if it is not, copy it and call the function again with the copy
	if(!traits::hasStorage(matB)){
		StorageMatrix packedB=matB;
		generalPanelPanelOperation(matA, packedB,matC,kernel,blas::column_major_tag());
		return;
	}
	
	std::size_t n = matA.size1();
	std::size_t m = matB.size2();
	std::size_t k = matA.size2();
	SIZE_CHECK( k == matB.size1());
	SIZE_CHECK( matC.size1() == matA.size1());
	SIZE_CHECK( matC.size2() == matB.size2());
	
	
	StorageMatrix blockStorage(k,k);
	
	for(std::size_t beginBlock = 0; beginBlock < n; beginBlock += k){
		std::size_t currentK = std::min(n - beginBlock, k);
		std::size_t endBlock = beginBlock + currentK;
		
		//pack a Block of matrix B for computation. transpose the part of B to fulfill the requirements
		//of generalRegisterPanelBlockOperation. This will pay off since  blockStorage is reused often
		blockStorage.resize(k,currentK);
		noalias(blockStorage)=trans( subrange(matA, beginBlock, endBlock, 0, k));
		//evaluate the block operation (GEPB_OPT1 in the algorithm)
		for(std::size_t columnC= 0; columnC!= m; ++columnC){
			detail::generalBlockPanelOperation(
				blockStorage,
				subrange(matB,0, k, columnC, columnC + 1 ),
				subrange(matC, beginBlock, endBlock,columnC, columnC + 1),
				kernel
			);
		}
	}
}

///\brief Performs matrix-matrix multiplication like operations on 2 matrices.
///
///algorithm as described in Gotos paper "Anatomy of High-Performance Matrix Multiplication"
///this is the outer layer of the described algorithm. it splits the matrices into panels and than performs
///Panel-Panel operations repeatedly.This is the version for row_major matrices, which is a bit more optimized
///than column major.
template<class MatA,class MatB,class MatC,class ComputeKernel>
void generalMatrixMatrixOperation(
	MatA const & matA,
	MatB const & matB,
	MatC& matC,
	ComputeKernel kernel,
	blas::row_major_tag
){
	
	typedef typename MatC::value_type value_type;
	typedef blas::matrix<value_type,blas::row_major> StorageMatrix;
	
	std::size_t n = matA.size1();
	std::size_t m = matB.size2();
	std::size_t k = matA.size2();
	SIZE_CHECK( k == matB.size1());
	SIZE_CHECK( matC.size1() == matA.size1());
	SIZE_CHECK( matC.size2() == matB.size2());
	
	//small matrices are calculated in one go.
	//todo: check whether matrices are not packed and than check strides big stride->copy
	if(n < 64 && m < 64){
		StorageMatrix transA=trans(matA);
		generalPanelBlockOperation(transA, matB,matC,kernel);
		return;
	}
	
	//very simple choice at the moment, but should be sufficient for not too big matrices.
	std::size_t const blockSize=std::min(k,std::size_t(256));
	
	for(std::size_t i = 0; i < k; i += blockSize){
		//copy the Panel of A into temporary storage, than get all subpanels of B and then call gepp for every combination
		std::size_t currentSize=std::min(k-i,blockSize);
		StorageMatrix Atemp=subrange(matA,0,n, i ,i+currentSize);

		generalPanelPanelOperation(
			Atemp,
			subrange(matB,i , i+currentSize, 0, m),
			matC,
			kernel,
			blas::row_major_tag()
		);
	}
}

///\brief Performs matrix-matrix multiplication like operations on 2 matrices.
///
///algorithm as described in Gotos paper "Anatomy of High-Performance Matrix Multiplication"
///this is the outer layer of the described algorithm. it splits the matrices into panels and than performs
///Panel-Panel operations repeatedly. This is the version for column major matrices
template<class MatA,class MatB,class MatC,class ComputeKernel>
void generalMatrixMatrixOperation(
	MatA const & matA,
	MatB const & matB,
	MatC& matC,
	ComputeKernel kernel,
	blas::column_major_tag
){
	
	typedef typename MatC::value_type value_type;
	typedef blas::matrix<value_type,blas::column_major> StorageMatrix;
	
	std::size_t n = matA.size1();
	std::size_t m = matB.size2();
	std::size_t k = matA.size2();
	SIZE_CHECK( k == matB.size1());
	SIZE_CHECK( matC.size1() == matA.size1());
	SIZE_CHECK( matC.size2() == matB.size2());
	
	//small matrices are calculated in one go.
	//todo: check whether matrices are not packed and than check strides big stride->copy
	if(n < 64 && m < 64){
		//these matrices are psurely to small for the overhead of copying...
		generalBlockPanelOperation(matA, trans(matB),matC,kernel,blas::column_major_tag());
		return;
	}
	
	
	//very simple choice at the moment, but should be sufficient for not too big matrices.
	std::size_t const blockSize=std::min(k,512ul);
	
	for(std::size_t i = 0; i < k; i += blockSize){
		//copy the Panel of A into temporary storage, than get all subpanels of B and then call gepp for every combination
		std::size_t currentSize=std::min(k-i,blockSize);
		StorageMatrix Atemp=subrange(matA,0,n, i ,i+currentSize);

		generalPanelPanelOperation(
			subrange(Atemp,0,n,0,currentSize),
			subrange(matB,i , i+currentSize, 0, m),
			matC,
			kernel,
			blas::column_major_tag()
		);
	}
}
}


///\brief General building block for matrix-matrix-multiplication like operations using matrices with much more columns than rows(often called gepp in BLAS).
///
///Performs matrix-matrix like operations. the type of operation is governed by the kernel parameter.
///If the kernel is: kernel(x,y)=x*y, the algorithm can be described as matrix-matrix multiplication as follows:
///The Algorithm computes C+=A*B for Panel-Matrices A and B which have the properties:
///If A and B are row_major, A should have more rows than columns - the more, the better, 
///B must have more columns than rows. The result is a sum of outer products.
///Also the matrices should be so small that A, a square matrix D with size A.size2()  and the result of A*D fit's
///into L2 cache.
///C must be initialized with the proper sizes.
///\par 
///Algorithm implemented as described in Gotos paper "Anatomy of High-Performance Matrix Multiplication"
///we generalize it here for arbitrary operations instead of the multiplication, since we use ATLAS for fast matrix products.
///this is only a fallback or used for operations which do not fit into the BLAS framework.
template<class MatA,class MatB,class MatC,class ComputeKernel>
void generalPanelPanelOperation(
	blas::matrix_expression<MatA> const & matA,
	blas::matrix_expression<MatB> const & matB,
	blas::matrix_container<MatC>& matC,
	ComputeKernel kernel
){
	//all Matrices need to be either row major or column major, no mixing is allowed!
	BOOST_STATIC_ASSERT(
		(boost::is_same<
			typename MatB::orientation_category,
			typename MatC::orientation_category
		>::value));
	BOOST_STATIC_ASSERT(
		(boost::is_same<
			typename MatA::orientation_category,
			typename MatC::orientation_category
		>::value));
	//check correctness of arguments
	SIZE_CHECK(matC().size2()==matB().size2());
	SIZE_CHECK(matC().size1()==matA().size1());
	SIZE_CHECK(matA().size2()==matB().size1());
	detail::generalPanelPanelOperation(
		matA(),matB(),matC(),
		kernel,
		typename MatA::orientation_category());
}
///\brief General building block for matrix-matrix-multiplication like operations using generla matrices(often called gemmin BLAS).
///
///Performs matrix-matrix like operations. the type of operation is governed by the kernel parameter.
///If the kernel is: kernel(x,y)=x*y, the algorithm can be described as matrix-matrix multiplication as follows:
///The Algorithm computes C+=A*B for Panel-Matrices A and B which have the properties:
///C must be initialized with the proper sizes.
///\par 
///Algorithm implemented as described in Gotos paper "Anatomy of High-Performance Matrix Multiplication"
///we generalize it here for arbitrary operations instead of the multiplication, since we use ATLAS for fast matrix products.
///this is only a fallback or used for operations which do not fit into the BLAS framework.
template<class MatA,class MatB,class MatC,class ComputeKernel>
void generalMatrixMatrixOperation(
	blas::matrix_expression<MatA> const & matA,
	blas::matrix_expression<MatB> const & matB,
	blas::matrix_container<MatC>& matC,
	ComputeKernel kernel
){
	//all Matrices need to be either row major or column major, no mixing is allowed!
	BOOST_STATIC_ASSERT(
		(boost::is_same<
			typename MatB::orientation_category,
			typename MatC::orientation_category
		>::value));
	BOOST_STATIC_ASSERT(
		(boost::is_same<
			typename MatA::orientation_category,
			typename MatC::orientation_category
		>::value));
	//check correctness of arguments
	SIZE_CHECK(matC().size2()==matB().size2());
	SIZE_CHECK(matC().size1()==matA().size1());
	SIZE_CHECK(matA().size2()==matB().size1());
	detail::generalMatrixMatrixOperation(
		matA(),matB(),matC(),
		kernel,
		typename MatA::orientation_category());
}
}
#endif
