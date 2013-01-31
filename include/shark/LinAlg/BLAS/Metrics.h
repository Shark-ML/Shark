/**
*
*  \brief Helper functions to calculate several norms and distances.
*
*  \author O.Krause M.Thuma
*  \date 2010-2011
*
*  \par Copyright (c) 1998-2007:
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
#ifndef SHARK_LINALG_BLAS_METRICS_H
#define SHARK_LINALG_BLAS_METRICS_H

#include <shark/LinAlg/BLAS/fastOperations.h>
#include <shark/LinAlg/BLAS/VectorTransformations.h>
#include <shark/LinAlg/BLAS/MatrixVectorOperation.h>
#include <shark/LinAlg/BLAS/Tools.h>
namespace shark{
	
///////////////////////////////////////NORMS////////////////////////////////////////

/**
* \brief Calculates squared norm_2.
*/
template<class VectorT>
typename VectorT::value_type normSqr(blas::vector_expression<VectorT> const& vector) {
	return sum(sqr(vector));
}

/**
* \brief Normalized squared norm_2 (diagonal Mahalanobis).
*
* Contrary to some conventions, dimension-wise weights are considered instead of std. deviations:
* \f$ n^2(v) = \sum_i w_i v_i^2 \f$
* nb: the weights themselves are not squared, but multiplied onto the squared components
*/
template<class VectorT, class WeightT>
typename VectorT::value_type diagonalMahalanobisNormSqr(
	blas::vector_expression<VectorT> const& vector, 
	blas::vector_expression<WeightT> const& weights
) {
	SIZE_CHECK( vector().size() == weights().size() );
	return inner_prod(weights(),sqr(vector()));
}

/**
* \brief Normalized norm_2 (diagonal Mahalanobis).
*
* Contrary to some conventions, dimension-wise weights are considered instead of std. deviations:
* \f$ n^2(v) = \sqrt{\sum_i w_i v_i^2} \f$
* nb: the weights themselves are not squared, but multiplied onto the squared components
*/
template<class VectorT, class WeightT>
typename VectorT::value_type diagonalMahalanobisNorm(
	blas::vector_expression<VectorT> const& vector, 
	blas::vector_expression<WeightT> const& weights
) {
	SIZE_CHECK( vector().size() == weights().size() );
	return std::sqrt(diagonalMahalanobisNormSqr(vector,weights));
}

////////////////////////////////////////DISTANCES/////////////////////////////////////////////////

namespace detail{
	/**
	* \brief Normalized Euclidian squared distance (squared diagonal Mahalanobis) 
	* between two vectors, optimized for two Compressed arguments.
	*
	* Contrary to some conventions, dimension-wise weights are considered instead of std. deviations:
	* \f$ d^2(v) = \sum_i w_i (x_i-z_i)^2 \f$
	* NOTE: The weights themselves are not squared, but multiplied onto the squared components.
	*/
	template<class VectorT, class VectorU, class WeightT>
	typename VectorT::value_type diagonalMahalanobisDistanceSqr(
		VectorT const& op1,
		VectorU const& op2,
		WeightT const& weights,
		boost::mpl::true_, //first argument compressed 
		boost::mpl::true_ //second argument compressed
	){
		typename VectorT::value_type sum=0;
		typename VectorT::const_iterator iter1=op1.begin();
		typename VectorU::const_iterator iter2=op2.begin();
		typename VectorT::const_iterator end1=op1.end();
		typename VectorU::const_iterator end2=op2.end();
		size_t index1=iter1.index();
		size_t index2=iter2.index();
		while(true)
		{
			if(index1<index2){
				sum += weights(index1) * sqr(*iter1);
				++iter1;
				if(iter1==end1) break;
				index1=iter1.index();
			}
			else if(index1==index2){
				sum += weights(index1) * sqr(*iter1-*iter2);
				++iter1;
				++iter2;
				if(iter1==end1) break;
				if(iter2==end2) break;
				index1=iter1.index();
				index2=iter2.index();
			}
			else {
				sum += weights(index2) * sqr(*iter2);
				++iter2;
				if(iter2==end2) break;
				index2=iter2.index();
			}
		}
		while(iter1 != end1)
		{
			index1=iter1.index();
			sum += weights(index1) * sqr(*iter1);
			++iter1;
		}
		while(iter2 != end2)
		{
			index2=iter2.index();
			sum += weights(index2) * sqr(*iter2);
			++iter2;
		}
		return sum;
	}
	
	/**
	* \brief Normalized Euclidian squared distance (squared diagonal Mahalanobis) 
	* between two vectors, optimized for one dense and one sparse argument
	*/
	template<class VectorT, class VectorU, class WeightT>
	typename VectorT::value_type diagonalMahalanobisDistanceSqr(
		VectorT const& op1,
		VectorU const& op2,
		WeightT const& weights,
		boost::mpl::true_, //first argument compressed 
		boost::mpl::false_ //second argument not compressed, we assume dense
	){
		typename VectorT::value_type sum=0;
		
		typename VectorT::const_iterator iter=op1.begin();
		typename VectorT::const_iterator end=op1.end();
		
		std::size_t index = iter.index();
		std::size_t pos = 0;
		while(pos != op2.size()){
			if(pos == index){
				sum += weights(index) * sqr(*iter-op2(pos));
				++pos;
				++iter;
				if(iter == end)
					index = op2.size();
				else
					index = iter.index();
			}
			else while(pos != index){
				sum += weights(pos) * sqr(op2(pos));
				++pos;
			}
		}
		return sum;
	}
	template<class VectorT, class VectorU, class WeightT>
	typename VectorT::value_type diagonalMahalanobisDistanceSqr(
		VectorT const& op1,
		VectorU const& op2,
		WeightT const& weights,
		boost::mpl::false_ arg1tag, //first argument not compressed, we assume dense 
		boost::mpl::true_ arg2tag   //second argument compressed
	){
		return diagonalMahalanobisDistanceSqr(op2,op1,weights,arg2tag,arg1tag);
	}
	
	template<class VectorT, class VectorU, class WeightT>
	typename VectorT::value_type diagonalMahalanobisDistanceSqr(
		VectorT const& op1,
		VectorU const& op2,
		WeightT const& weights,
		boost::mpl::false_,
		boost::mpl::false_
	){
		return diagonalMahalanobisNormSqr(op1-op2,weights);
	}
	
	
	struct SquaredScalarDistance{
		template<class T>
		T operator()(T x, T y){
			T diff=x-y;
			return diff*diff;
		}
	};
	///\brief efficient implementation for multiple vectors
	template<class MatrixT,class VectorU, class Result>
	void distanceSqrBlockVector(
		MatrixT const& operands,
		VectorU const& op2,
		Result& result,
		boost::mpl::false_,
		boost::mpl::false_
	){
		shark::zero(result);
		shark::generalMatrixVectorOperation(operands,op2,result,SquaredScalarDistance());
	}
	
	//default implementation used, when one of the arguments is not dense
	template<class MatrixT,class VectorU, class Result, class Flag>
	void distanceSqrBlockVector(
		MatrixT const& operands,
		VectorU const& op2,
		Result& result,
		boost::mpl::true_,
		Flag
	){
		for(std::size_t i = 0; i != operands.size1(); ++i){
			result(i) = sum(sqr(row(operands,i)-op2));
		}
	}
	//default implementation used, when none of the arguments are dense
	template<class MatrixT,class VectorU, class Result>
	void distanceSqrBlockVector(
		MatrixT const& operands,
		VectorU const& op2,
		Result& result,
		boost::mpl::false_,
		boost::mpl::true_
	){
		for(std::size_t i = 0; i != operands.size1(); ++i){
			result(i) = sum(sqr(row(operands,i)-op2));
		}
	}
	
	
	///\brief implementation for two input blocks where at least one matrix has only a few rows
	template<class MatrixX,class MatrixY, class Result>
	void distanceSqrBlockBlockRowWise(
		MatrixX const& X,
		MatrixY const& Y,
		Result& distances
	){
		std::size_t sizeX=X.size1();
		std::size_t sizeY=Y.size1();
		if(sizeX  < sizeY){//iterate over the rows of the block with less rows
			for(std::size_t i = 0; i != sizeX; ++i){
				blas::matrix_row<Result> distanceRow = row(distances,i);
				distanceSqrBlockVector(
					Y,row(X,i),distanceRow,
					typename traits::IsCompressed<MatrixY>::type(),
					typename traits::IsCompressed<MatrixX>::type()
				);
			}
		}else{
			for(std::size_t i = 0; i != sizeY; ++i){
				blas::matrix_column<Result> distanceCol = column(distances,i);
				distanceSqrBlockVector(
					X,row(Y,i),distanceCol,
					typename traits::IsCompressed<MatrixX>::type(),
					typename traits::IsCompressed<MatrixY>::type()
				);
			}
		}
	}
	
	///\brief implementation for two dense input blocks
	template<class MatrixX,class MatrixY, class Result>
	void distanceSqrBlockBlock(
		MatrixX const& X,
		MatrixY const& Y,
		Result& distances,
		boost::mpl::false_,//both arguments are dense
		boost::mpl::false_
	){
		typedef typename Result::value_type value_type;
		std::size_t sizeX=X.size1();
		std::size_t sizeY=Y.size1();
		if(sizeX < 10 || sizeY<10){
			distanceSqrBlockBlockRowWise(X,Y,distances);
			return;
		}
		//fast blockwise iteration
		//uses: (a-b)^2 = a^2 -2ab +b^2
		fast_prod(X,trans(Y),distances);
		distances*=-2;
		//first a^2+b^2 
		blas::vector<value_type> ySqr(sizeY);
		for(std::size_t i = 0; i != sizeY; ++i){
			ySqr(i) = normSqr(row(Y,i));
		}
		//initialize d_ij=x_i^2+y_i^2
		for(std::size_t i = 0; i != sizeX; ++i){
			value_type xSqr = normSqr(row(X,i));
			noalias(row(distances,i)) += repeat(xSqr,sizeY) + ySqr;
		}
	}
	//\brief default implementation used, when one of the arguments is not dense
	template<class MatrixX,class MatrixY,class Result>
	void distanceSqrBlockBlock(
		MatrixX const& X,
		MatrixY const& Y,
		Result& distances,
		boost::mpl::true_,
		boost::mpl::true_
	){
		distanceSqrBlockBlockRowWise(X,Y,distances);
	}
}

/**
* \ingroup shark_globals
* 
* @{
*/

/** 
* \brief Normalized Euclidian squared distance (squared diagonal Mahalanobis) 
* between two vectors.
*
* NOTE: The weights themselves are not squared, but multiplied onto the squared components.
*/
template<class VectorT, class VectorU, class WeightT>
typename VectorT::value_type diagonalMahalanobisDistanceSqr(
	blas::vector_expression<VectorT> const& op1,
	blas::vector_expression<VectorU> const& op2, 
	blas::vector_expression<WeightT> const& weights
){
	SIZE_CHECK(op1().size()==op2().size());
	SIZE_CHECK(op1().size()==weights().size());
	//dispatch given the types of the argument
	return detail::diagonalMahalanobisDistanceSqr(
		op1(), op2(), weights(),
		typename traits::IsCompressed<VectorT>::type(),
		typename traits::IsCompressed<VectorU>::type()
	);
}

/**
* \brief Squared distance between two vectors.
*/
template<class VectorT,class VectorU>
typename VectorT::value_type distanceSqr(
	blas::vector_expression<VectorT> const& op1,
	blas::vector_expression<VectorU> const& op2
){
	SIZE_CHECK(op1().size()==op2().size());
	typedef typename VectorT::value_type value_type;
	blas::scalar_vector< value_type > one(op1().size(),static_cast<value_type>(1.0));
	return diagonalMahalanobisDistanceSqr(op1,op2,one);
}

/**
* \brief Squared distance between a vector and a set of vectors
*
* The squared distance between the vector and every row-vector of the matrix is calculated.
* This can be implemented much more efficiently.
*/
template<class MatrixT,class VectorU>
blas::vector<typename MatrixT::value_type> distanceSqr(
	blas::matrix_expression<MatrixT> const& operands,
	blas::vector_expression<VectorU> const& op2
){
	SIZE_CHECK(operands().size2()==op2().size());
	blas::vector<typename MatrixT::value_type> distances(operands().size1());
	detail::distanceSqrBlockVector(
		operands(),op2(),distances,
		typename traits::IsCompressed<MatrixT>::type(),
		typename traits::IsCompressed<VectorU>::type()
	);
	return distances;
}

/**
* \brief Squared distance between a vector and a set of vectors
*
* The squared distance between the vector and every row-vector of the matrix is calculated.
* This can be implemented much more efficiently.
*/
template<class MatrixT,class VectorU>
blas::vector<typename MatrixT::value_type> distanceSqr(
	blas::vector_expression<VectorU> const& op1,
	blas::matrix_expression<MatrixT> const& operands
){
	SIZE_CHECK(operands().size2()==op1().size());
	blas::vector<typename MatrixT::value_type> distances(operands().size1());
	detail::distanceSqrBlockVector(
		operands(),op1(),distances,
		typename traits::IsCompressed<MatrixT>::type(),
		typename traits::IsCompressed<VectorU>::type()
	);
	return distances;
}

/**
* \brief Squared distance between the vectors of two sets of vectors
*
* The squared distance between every row-vector of the first matrix x
* and every row-vector of the second matrix y is calculated.
* This can be implemented much more efficiently. 
* The results are returned as a matrix, where the element in the i-th 
* row and the j-th column is distanceSqr(x_i,y_j).
*/
template<class MatrixT,class MatrixU>
blas::matrix<typename MatrixT::value_type> distanceSqr(
	blas::matrix_expression<MatrixT> const& X,
	blas::matrix_expression<MatrixU> const& Y
){
	typedef blas::matrix<typename MatrixT::value_type> Matrix;
	SIZE_CHECK(X().size2()==Y().size2());
	std::size_t sizeX=X().size1();
	std::size_t sizeY=Y().size1();
	Matrix distances(sizeX, sizeY);
	detail::distanceSqrBlockBlock(
		X(),Y(),distances,
		typename traits::IsCompressed<MatrixT>::type(),
		typename traits::IsCompressed<MatrixU>::type()
	);
	return distances;
	
}


/**
* \brief Calculates distance between two vectors.
*/
template<class VectorT,class VectorU>
typename VectorT::value_type distance(
	blas::vector_expression<VectorT> const& op1,
	blas::vector_expression<VectorU> const& op2
){
	SIZE_CHECK(op1().size()==op2().size());
	return std::sqrt(distanceSqr(op1,op2));
}

/**
* \brief Normalized euclidian distance (diagonal Mahalanobis) between two vectors.
*
* Contrary to some conventions, dimension-wise weights are considered instead of std. deviations:
* \f$ d(v) = \left( \sum_i w_i (x_i-z_i)^2 \right)^{1/2} \f$
* nb: the weights themselves are not squared, but multiplied onto the squared components
*/
template<class VectorT, class VectorU, class WeightT>
typename VectorT::value_type diagonalMahalanobisDistance(
	blas::vector_expression<VectorT> const& op1,
	blas::vector_expression<VectorU> const& op2, 
	blas::vector_expression<WeightT> const& weights
){
	SIZE_CHECK(op1().size()==op2().size());
	SIZE_CHECK(op1().size()==weights().size());
	return std::sqrt(diagonalMahalanobisDistanceSqr(op1(), op2(), weights));
}
/** @}*/
}

#endif