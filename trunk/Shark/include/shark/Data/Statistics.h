//===========================================================================
/*!
 * 
 *
 * \brief       some functions for vector valued statistics like mean, variance and covariance
 * 
 * 
 *
 * \author      O.Krause, C. Igel
 * \date        2010-2013
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
//===========================================================================
#ifndef SHARK_DATA_STATISTICS_H
#define SHARK_DATA_STATISTICS_H

#include <shark/Data/Dataset.h>

/**
* \ingroup shark_globals
* 
* @{
*/

namespace shark{
	
//! Calculates the mean and variance values of the input data
template<class Vec1T,class Vec2T,class Vec3T>
void meanvar
(
	const Data<Vec1T>& data,
	blas::vector_container<Vec2T>& mean,
	blas::vector_container<Vec3T>& variance
);
//! Calculates the mean, variance and covariance values of the input data
template<class Vec1T,class Vec2T,class MatT>
void meanvar
(
	const Data<Vec1T>& data,
	blas::vector_container<Vec2T>& mean,
	blas::matrix_container<MatT>& variance
);

//! Calculates the mean vector of the input vectors.
template<class VectorType>
VectorType mean(Data<VectorType> const& data);

template<class VectorType>
VectorType mean(UnlabeledData<VectorType> const& data){
	return mean(static_cast<Data<VectorType> const&>(data));
}

//! Calculates the variance vector of the input vectors
template<class VectorType>
VectorType variance(const Data<VectorType>& data);

//! Calculates the covariance matrix of the data vectors
template<class VectorType>
typename VectorMatrixTraits<VectorType>::DenseMatrixType covariance(const Data<VectorType>& data);
	
}
/** @}*/
#include "Impl/Statistics.inl"

#endif
