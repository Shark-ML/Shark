//===========================================================================
/*!
 *  \file VectorStatistics.h
 *
 *  some functions for vector valued statistics like mean, variance and covariance
 *
 *  \author O.Krause, C. Igel
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
