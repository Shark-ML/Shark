/*!
 *  \brief Some operations for creating rotation matrices
 *
 *
 *  \author  O. Krause
 *  \date    2011
 *
 *  \par Copyright (c) 1999-2001:
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
 *
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
	
/*! Transforms a quadratic matrix such that it forms an orthonormal Basis
 *  using Gram-Schmidt Orthonormalisation.
 */
template<class MatrixT>
void orthoNormalize(blas::matrix_container<MatrixT>& matrixC){
	MatrixT& matrix = matrixC();
	SIZE_CHECK(matrix.size1() == matrix.size2());

	size_t size = matrix.size1();
	RealVector diff(size);
	for(size_t i=0; i != size;++i){
		diff.clear();
		for(size_t j=0;j != i; ++j){
			row(matrix,i)-=row(matrix,j) * inner_prod(row(matrix,j),row(matrix,i));
		}
		row(matrix,i)/=norm_2(row(matrix,i));
	}
}

/// \brief Initializes a matrix such that it forms a random rotation matrix.
///
/// The matrix needs to be quadratic and have the proper size
/// (e.g. call matrix::resize before).
template< class MatrixT, typename RngType >
void randomRotationMatrix(blas::matrix_container<MatrixT>& matrixC,RngType& rng){
	MatrixT& matrix = matrixC();
	SIZE_CHECK(matrix.size1() == matrix.size2());

	size_t size = matrix.size1();
	Normal< RngType > normal( rng, 0, 1 );
	for(std::size_t i=0; i != size;++i){
		for(std::size_t j=0;j != size; ++j){
			matrix(i,j) = normal();
		}
	}
	orthoNormalize(matrix);

}

/// \brief Initializes a matrix such that it forms a random rotation
///
///matrix.  The matrix needs to be quadratic and have the proper size
///(e.g. call matrix::resize before) uses the global RNG.
template<class MatrixT>
void randomRotationMatrix(blas::matrix_container<MatrixT>& matrixC){
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


//! \brief Generates a Householder reflection from a vector to use with applyHouseholderLeft/Right
//!
//! Given a Vector x=(x0,x1,...,xn), finds a reflection with the property
//! (c, 0,0,...0) = (I-beta v v^t)x
//! and v = (x0-c,x1,x2,...,xn)
template<class X, class R>
typename X::value_type createHouseholderReflection(
	blas::vector_expression<X> const& x, 
	blas::vector_expression<R>& reflection
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
	blas::matrix_expression<Mat> & matrix,
	blas::vector_expression<R> const& reflection, 
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
	blas::vector<T> temp(matrix().size1());
	
	//Ax
	axpy_prod(matrix,reflection,temp);
	
	//A -=beta*(Ax)x^T
    noalias(matrix()) -= beta * outer_prod(temp,reflection);
}
//\brief rotates a matrix using a householder reflection 
//
//calculates (1-beta*xx^T)*A
template<class Mat, class R, class T>
void applyHouseholderOnTheLeft(
	blas::matrix_expression<Mat> & matrix,
	blas::vector_expression<R> const& reflection, 
	T const& beta
){

	SIZE_CHECK(matrix().size1() == reflection().size());
	SIZE_CHECK(reflection().size() != 0 );
	
	//special case for last iteration of QR etc
	if(reflection().size() == 1){
		matrix()*=1-beta;
		return;
	}

	blas::vector<T> temp(matrix().size2());
	
	//x^T A
	axpy_prod(trans(matrix),reflection,temp);
	
	//A -=beta*x(x^T A)
    noalias(matrix()) -= beta * outer_prod(reflection,temp);
}
/** @}*/
}}
#endif
