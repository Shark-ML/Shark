/*!
 * 
 *
 * \brief       Some operations for creating rotation matrices
 * 
 * 
 * 
 *
 * \author      O. Krause
 * \date        2011
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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

#ifndef SHARK_LINALG_ROTATIONS_H
#define SHARK_LINALG_ROTATIONS_H

#include <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>
namespace shark{ namespace blas{
/**
 * \ingroup shark_globals
 * 
 * @{
 */

//! \brief Generates a Householder reflection from a vector to use with applyHouseholderLeft/Right
//!
//! Given a Vector x=(x0,x1,...,xn), finds a reflection with the property
//! (c, 0,0,...0) = (I-beta v v^t)x
//! and v = (x0-c,x1,x2,...,xn)
template<class X, class R>
typename X::value_type createHouseholderReflection(
	vector_expression<X> const& x, 
	vector_expression<R>& reflection
){
	SIZE_CHECK(x().size() != 0);
	SIZE_CHECK(x().size() == reflection().size());
	
	//special case for last iteration of QR etc
	//by convention diagonal elements are > 0
	if(x().size() == 1){
		reflection()(0) = 1;
		return 2;
	}
		
	
	typedef typename X::value_type Value;
	
	double norm = norm_2(x);
	if (x()(0) >= 0.0)
		norm *= -1.0;
    
	noalias(reflection()) = x;
	reflection()(0) -= norm;
	reflection() /= (x()(0) - norm);
	//if pivot is close to 0, this is one->numericaly stable
	//compared to the usual formula
	Value beta = (norm - x()(0)) / norm;
	return beta;
}
//\brief rotates a matrix using a householder reflection 
//
//calculates A*(1-beta*xx^T)
template<class Mat, class R, class T>
void applyHouseholderOnTheRight(
	matrix_expression<Mat> & matrix,
	vector_expression<R> const& reflection, 
	T beta
){
	SIZE_CHECK(matrix().size2() == reflection().size());
	SIZE_CHECK(reflection().size() != 0 );
	
	//special case for last iteration of QR etc
	if(reflection().size() == 1){
		matrix() *= 1-beta;
		return;
	}
	
	SIZE_CHECK(matrix().size2() == reflection().size());
	vector<T> temp(matrix().size1());
	
	//Ax
	axpy_prod(matrix,reflection,temp);
	
	//A -=beta*(Ax)x^T
    noalias(matrix()) -= beta * outer_prod(temp,reflection);
}


/// \brief rotates a matrix using a householder reflection 
///
/// calculates (1-beta*xx^T)*A
template<class Mat, class R, class T>
void applyHouseholderOnTheLeft(
	matrix_expression<Mat> & matrix,
	vector_expression<R> const& reflection, 
	T const& beta
){

	SIZE_CHECK(matrix().size1() == reflection().size());
	SIZE_CHECK(reflection().size() != 0 );
	
	//special case for last iteration of QR etc
	if(reflection().size() == 1){
		matrix()*=1-beta;
		return;
	}

	vector<T> temp(matrix().size2());
	
	//x^T A
	axpy_prod(trans(matrix),reflection,temp);
	
	//A -=beta*x(x^T A)
    noalias(matrix()) -= beta * outer_prod(reflection,temp);
}

/// \brief rotates a matrix using a householder reflection 
///
/// calculates (1-beta*xx^T)*A
template<class Mat, class R, class T>
void applyHouseholderOnTheLeft(
	temporary_proxy<Mat> matrix,
	vector_expression<R> const& reflection, 
	T const& beta
){
	applyHouseholderOnTheLeft(static_cast<Mat&>(matrix),reflection,beta);
}

/// \brief Initializes a matrix such that it forms a random rotation matrix.
///
/// The matrix needs to be quadratic and have the proper size
/// (e.g. call matrix::resize before).
///
/// One common way to  do this is using Gram-Schmidt-Orthogonalisation 
/// on a matrix which is initialized with gaussian numbers. However, this is
/// highly unstable for big matrices. 
///
/// This algorithm is implemented from one of the algorithms presented in
/// Francesco Mezzadri "How to generate random matrices from the classical compact groups"
/// http://arxiv.org/abs/math-ph/0609050v2
///
/// He gives two algorithms: the first one uses QR decomposition on the random gaussian
/// matrix and ensures that the signs of Q are correct by multiplying every column of Q
/// with the sign of the diagonal of R. 
///
/// We use nother algorithm implemented in the paper which works similarly, but 
/// reversed. We apply Householder rotations H_N H_{N-1}..H_1 where
/// H_1 is generated from a random vector on the n-dimensional unit sphere.
/// this requires less operations and is thus preferable. Also only half the
/// random numbers need to be generated
template< class MatrixT, typename RngType >
void randomRotationMatrix(matrix_container<MatrixT>& matrixC,RngType& rng){
	MatrixT& matrix = matrixC();
	SIZE_CHECK(matrix.size1() == matrix.size2());
	SIZE_CHECK(matrix.size1() > 0);
	Normal< RngType > normal( rng, 0, 1 );
	size_t size = matrix.size1();
	diag(matrix) = repeat(1.0,size);
	//the first element is just -1 or 1
	matrix(size-1,size-1) = normal();
	matrix(size-1,size-1) /= std::abs(matrix(size-1,size-1));
	
	RealVector v(size);
	for(std::size_t i = 2; i != size;++i){
		//create the random vector on the unit-sphere for the i-dimensional subproblem
		for(std::size_t j=0;j != i; ++j){
			v(j) = normal();
		}
		subrange(v,0,i) /=norm_2(subrange(v,0,i));//project on sphere
		
		double v0 = v(0);
		v(0) += v0/std::abs(v0);
		
		//compute new norm of v
		//~ double normvSqr = 1.0+(1+v0)*(1+v0)-v0*v0;
		double normvSqr = norm_sqr(subrange(v,0,i));
		
		//apply the householder rotation to the i-th submatrix
		applyHouseholderOnTheLeft(subrange(matrix,size-i,size,size-i,size),subrange(v,0,i),2.0/normvSqr);
		
	}
}

/// \brief Initializes a matrix such that it forms a random rotation
///
///matrix.  The matrix needs to be quadratic and have the proper size
///(e.g. call matrix::resize before) uses the global RNG.
template<class MatrixT>
void randomRotationMatrix(matrix_container<MatrixT>& matrixC){
	randomRotationMatrix( matrixC, Rng::globalRng );
}

//! Creates a random rotation matrix with a certain size using the random number generator rng.
template<typename RngType>
RealMatrix randomRotationMatrix(size_t size,RngType& rng){
	RealMatrix mat(size,size);
	randomRotationMatrix(mat,rng);
	return mat;
}

//! Creates a random rotation matrix with a certain size using the global random number gneerator.
inline RealMatrix randomRotationMatrix(size_t size){
	return randomRotationMatrix( size, shark::Rng::globalRng );
}

/** @}*/
}}
#endif
