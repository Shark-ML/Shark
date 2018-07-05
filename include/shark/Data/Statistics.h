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
template<class Vec1T,class Vec2T,class Vec3T, class Device>
void meanvar
(
	Data<Vec1T> const& data,
	blas::vector_container<Vec2T, Device>& mean,
	blas::vector_container<Vec3T, Device>& variance
);
//! Calculates the mean, variance and covariance values of the input data
template<class Vec1T,class Vec2T,class MatT, class Device>
void meanvar
(
	Data<Vec1T> const& data,
	blas::vector_container<Vec2T, Device>& mean,
	blas::matrix_container<MatT, Device>& variance
);

//! Calculates the mean vector of the input vectors.
template<class VectorType>
VectorType mean(Data<VectorType> const& data);

//! Calculates the variance vector of the input vectors
template<class VectorType>
VectorType variance(Data<VectorType> const& data);

//! Calculates the covariance matrix of the data vectors
template<class VectorType>
blas::matrix<typename VectorType::value_type> covariance(Data<VectorType> const& data);
	
}
/** @}*/
#include "Impl/Statistics.inl"

#endif
