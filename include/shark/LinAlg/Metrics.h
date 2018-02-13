/*!
 * 
 *
 * \brief       Helper functions to calculate several norms and distances.
 * 
 * 
 *
 * \author      O.Krause M.Thuma
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_LINALG_METRICS_H
#define SHARK_LINALG_METRICS_H

#include <shark/LinAlg/BLAS/remora.hpp>
#include <shark/Core/Math.h>
namespace remora{
	
///////////////////////////////////////NORMS////////////////////////////////////////

/**
* \brief Normalized squared norm_2 (diagonal Mahalanobis).
*
* Contrary to some conventions, dimension-wise weights are considered instead of std. deviations:
* \f$ n^2(v) = \sum_i w_i v_i^2 \f$
* nb: the weights themselves are not squared, but multiplied onto the squared components
*/
template<class VectorT, class WeightT, class Device>
typename VectorT::value_type diagonalMahalanobisNormSqr(
	vector_expression<VectorT, Device> const& vector, 
	vector_expression<WeightT, Device> const& weights
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
template<class VectorT, class WeightT, class Device>
typename VectorT::value_type diagonalMahalanobisNorm(
	vector_expression<VectorT, Device> const& vector, 
	vector_expression<WeightT, Device> const& weights
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
		sparse_tag, 
		sparse_tag
	){
		using shark::sqr;
		typename VectorT::value_type sum=0;
		typename VectorT::const_iterator iter1=op1.begin();
		typename VectorU::const_iterator iter2=op2.begin();
		typename VectorT::const_iterator end1=op1.end();
		typename VectorU::const_iterator end2=op2.end();
		//be aware of empty vectors!
		while(iter1 != end1 && iter2 != end2)
		{
			std::size_t index1=iter1.index();
			std::size_t index2=iter2.index();
			if(index1==index2){
				sum += weights(index1) * sqr(*iter1-*iter2);
				++iter1;
				++iter2;
			}
			else if(index1<index2){
				sum += weights(index1) * sqr(*iter1);
				++iter1;
			}
			else {
				sum += weights(index2) * sqr(*iter2);
				++iter2;
			}
		}
		while(iter1 != end1)
		{
			std::size_t index1=iter1.index();
			sum += weights(index1) * sqr(*iter1);
			++iter1;
		}
		while(iter2 != end2)
		{
			std::size_t index2=iter2.index();
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
		sparse_tag, 
		dense_tag
	){
		std::size_t pos = 0;
		typename VectorT::value_type sum=0;
		auto end=op1.end();
		for(auto iter=op1.begin();iter != end;++iter,++pos){
			for(;pos != iter.index();++pos){
				sum += weights(pos) * op2(pos) * op2(pos);
			}
			double diff =  *iter-op2(pos);
			sum += weights(pos) * diff * diff;
		}
		for(;pos != op2.size();++pos){
			sum += weights(pos) * op2(pos) * op2(pos);
		}
		return sum;
	}
	template<class VectorT, class VectorU, class WeightT>
	typename VectorT::value_type diagonalMahalanobisDistanceSqr(
		VectorT const& op1,
		VectorU const& op2,
		WeightT const& weights,
		dense_tag arg1tag,
		sparse_tag arg2tag
	){
		return diagonalMahalanobisDistanceSqr(op2,op1,weights,arg2tag,arg1tag);
	}
	
	template<class VectorT, class VectorU, class WeightT>
	typename VectorT::value_type diagonalMahalanobisDistanceSqr(
		VectorT const& op1,
		VectorU const& op2,
		WeightT const& weights,
		dense_tag,
		dense_tag
	){
		return inner_prod(op1-op2,(op1-op2)*weights);
	}
	
	
	template<class MatrixT,class VectorU, class Result>
	void distanceSqrBlockVector(
		MatrixT const& operands,
		VectorU const& op2,
		Result& result
	){
		typedef typename Result::value_type value_type;
		scalar_vector< value_type, cpu_tag > one(op2.size(),static_cast<value_type>(1.0));
		for(std::size_t i = 0; i != operands.size1(); ++i){
			result(i) = diagonalMahalanobisDistanceSqr(
				row(operands,i),op2,one,
				typename MatrixT::evaluation_category::tag(),
				typename VectorU::evaluation_category::tag()
			);
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
				auto distanceRow = row(distances,i);
				distanceSqrBlockVector(
					Y,row(X,i),distanceRow
				);
			}
		}else{
			for(std::size_t i = 0; i != sizeY; ++i){
				auto distanceCol = column(distances,i);
				distanceSqrBlockVector(
					X,row(Y,i),distanceCol
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
		dense_tag,
		dense_tag
	){
		typedef typename Result::value_type value_type;
		std::size_t sizeX=X.size1();
		std::size_t sizeY=Y.size1();
		ensure_size(distances,X.size1(),Y.size1());
		if(sizeX < 10 || sizeY<10){
			distanceSqrBlockBlockRowWise(X,Y,distances);
			return;
		}
		//fast blockwise iteration
		//uses: (a-b)^2 = a^2 -2ab +b^2
		noalias(distances) = -2*prod(X,trans(Y));
		//first a^2+b^2 
		vector<value_type> ySqr(sizeY);
		for(std::size_t i = 0; i != sizeY; ++i){
			ySqr(i) = norm_sqr(row(Y,i));
		}
		//initialize d_ij=x_i^2+y_i^2
		for(std::size_t i = 0; i != sizeX; ++i){
			value_type xSqr = norm_sqr(row(X,i));
			noalias(row(distances,i)) += repeat(xSqr,sizeY) + ySqr;
		}
	}
	//\brief default implementation used, when one of the arguments is not dense
	template<class MatrixX,class MatrixY,class Result>
	void distanceSqrBlockBlock(
		MatrixX const& X,
		MatrixY const& Y,
		Result& distances,
		sparse_tag,
		sparse_tag
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
template<class VectorT, class VectorU, class WeightT, class Device>
typename VectorT::value_type diagonalMahalanobisDistanceSqr(
	vector_expression<VectorT, Device> const& op1,
	vector_expression<VectorU, Device> const& op2, 
	vector_expression<WeightT, Device> const& weights
){
	SIZE_CHECK(op1().size()==op2().size());
	SIZE_CHECK(op1().size()==weights().size());
	//dispatch given the types of the argument
	return detail::diagonalMahalanobisDistanceSqr(
		op1(), op2(), weights(),
		typename VectorT::evaluation_category::tag(),
		typename VectorU::evaluation_category::tag()
	);
}

/**
* \brief Squared distance between two vectors.
*/
template<class VectorT,class VectorU, class Device>
typename VectorT::value_type distanceSqr(
	vector_expression<VectorT, Device> const& op1,
	vector_expression<VectorU, Device> const& op2
){
	SIZE_CHECK(op1().size()==op2().size());
	typedef typename VectorT::value_type value_type;
	scalar_vector< value_type, cpu_tag > one(op1().size(),static_cast<value_type>(1.0));
	return diagonalMahalanobisDistanceSqr(op1,op2,one);
}

/**
* \brief Squared distance between a vector and a set of vectors and stores the result in the vector of distances
*
* The squared distance between the vector and every row-vector of the matrix is calculated.
* This can be implemented much more efficiently.
*/
template<class MatrixT,class VectorU, class VectorR, class Device>
void distanceSqr(
	matrix_expression<MatrixT, Device> const& operands,
	vector_expression<VectorU, Device> const& op2,
	vector_expression<VectorR, Device>& distances
){
	SIZE_CHECK(operands().size2()==op2().size());
	ensure_size(distances,operands().size1());
	detail::distanceSqrBlockVector(
		operands(),op2(),distances()
	);
}

/**
* \brief Squared distance between a vector and a set of vectors
*
* The squared distance between the vector and every row-vector of the matrix is calculated.
* This can be implemented much more efficiently.
*/
template<class MatrixT,class VectorU, class Device>
vector<typename MatrixT::value_type> distanceSqr(
	matrix_expression<MatrixT, Device> const& operands,
	vector_expression<VectorU, Device> const& op2
){
	SIZE_CHECK(operands().size2()==op2().size());
	vector<typename MatrixT::value_type> distances(operands().size1());
	distanceSqr(operands,op2,distances);
	return distances;
}

/**
* \brief Squared distance between a vector and a set of vectors
*
* The squared distance between the vector and every row-vector of the matrix is calculated.
* This can be implemented much more efficiently.
*/
template<class MatrixT,class VectorU, class Device>
vector<typename MatrixT::value_type> distanceSqr(
	vector_expression<VectorU, Device> const& op1,
	matrix_expression<MatrixT, Device> const& operands
){
	SIZE_CHECK(operands().size2()==op1().size());
	vector<typename MatrixT::value_type> distances(operands().size1());
	distanceSqr(operands,op1,distances);
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
template<class MatrixT,class MatrixU, class Device>
matrix<typename MatrixT::value_type> distanceSqr(
	matrix_expression<MatrixT, Device> const& X,
	matrix_expression<MatrixU, Device> const& Y
){
	typedef matrix<typename MatrixT::value_type> Matrix;
	SIZE_CHECK(X().size2()==Y().size2());
	std::size_t sizeX=X().size1();
	std::size_t sizeY=Y().size1();
	Matrix distances(sizeX, sizeY);
	detail::distanceSqrBlockBlock(
		X(),Y(),distances,
		typename MatrixT::evaluation_category::tag(),
		typename MatrixU::evaluation_category::tag()
	);
	return distances;
	
}


/**
* \brief Calculates distance between two vectors.
*/
template<class VectorT,class VectorU, class Device>
typename VectorT::value_type distance(
	vector_expression<VectorT, Device> const& op1,
	vector_expression<VectorU, Device> const& op2
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
template<class VectorT, class VectorU, class WeightT, class Device>
typename VectorT::value_type diagonalMahalanobisDistance(
	vector_expression<VectorT, Device> const& op1,
	vector_expression<VectorU, Device> const& op2, 
	vector_expression<WeightT, Device> const& weights
){
	SIZE_CHECK(op1().size()==op2().size());
	SIZE_CHECK(op1().size()==weights().size());
	return std::sqrt(diagonalMahalanobisDistanceSqr(op1(), op2(), weights));
}
/** @}*/
}

#endif
