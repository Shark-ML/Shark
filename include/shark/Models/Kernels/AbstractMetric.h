//===========================================================================
/*!
 * 
 *
 * \brief       abstract super class of all metrics
 * 
 * 
 *
 * \author      O. Krause
 * \date        2015
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
//===========================================================================

#ifndef SHARK_MODELS_KERNELS_ABSTRACTMETRIC_H
#define SHARK_MODELS_KERNELS_ABSTRACTMETRIC_H

#include <shark/Data/BatchInterface.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Core/ISerializable.h>
#include <shark/Core/INameable.h>
#include <shark/Core/Traits/ProxyReferenceTraits.h>
namespace shark {
	
	
template<class InputTypeT>
class AbstractMetric: public INameable, public IParameterizable, public ISerializable{
public:
	/// \brief  Input type of the Kernel.
	typedef InputTypeT InputType;
	/// \brief batch input type of the kernel
	typedef typename Batch<InputTypeT>::type BatchInputType;
	/// \brief Const references to InputType
	typedef typename ConstProxyReference<InputType const>::type ConstInputReference;
	/// \brief Const references to BatchInputType
	typedef typename ConstProxyReference<BatchInputType const>::type ConstBatchInputReference;

	AbstractMetric() { }
	virtual ~AbstractMetric() { }
		
	/// \brief From ISerializable, reads a metric from an archive.
	virtual void read( InArchive & archive ){
		RealVector p;
		archive & p;
		setParameterVector(p);
	}

	/// \brief From ISerializable, writes a metric to an archive.
	///
	/// The default implementation just saves the parameters.
	virtual void write( OutArchive & archive ) const{
		RealVector p = parameterVector();
		archive & p;
	}
	
	/// Computes the squared distance in the kernel induced feature space.
	virtual double featureDistanceSqr(ConstInputReference x1, ConstInputReference x2) const=0;
	
	virtual RealMatrix featureDistanceSqr(
		ConstBatchInputReference batchX1,
		ConstBatchInputReference batchX2
	) const = 0;
	

	/// \brief Computes the distance in the kernel induced feature space.
	double featureDistance(ConstInputReference x1, ConstInputReference x2) const {
		return std::sqrt(featureDistanceSqr(x1, x2));
	}
};
}
#endif