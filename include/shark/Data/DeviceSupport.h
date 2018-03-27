//===========================================================================
/*!
 * 
 *
 * \brief       Supporting routines for transfering dataset between devices, e.g. cpu to opencl

 *
 * \author      O. Krause
 * \date        2018
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

#ifndef SHARK_DATA_DEVICE_SUPPORT_H
#define SHARK_DATA_DEVICE_SUPPORT_H

#include <shark/Data/Dataset.h>
#include <shark/LinAlg/BLAS/device_copy.hpp>

namespace shark {

/**
 * \addtogroup shark_globals
 * @{
 */

///\brief Transfers a dataset from CPU to the GPU/OpenCL device
template<class Type, class T>
Data<blas::vector<Type, blas::gpu_tag> > toGPU(Data<blas::vector<T, blas::cpu_tag> > const& data){
	Data<blas::vector<Type, blas::gpu_tag> > data_gpu(data.numberOfBatches());
	for(std::size_t i = 0; i != data.numberOfBatches(); ++i){
		data_gpu.batch(i) = blas::copy_to_gpu(data.batch(i));
	}
	data_gpu.shape() = data.shape();
	return data_gpu;
}

///\brief Transfers a dataset from CPU to the GPU/OpenCL device
///
/// class labels are converted to one-hot encoding with a given Type
template<class Type>
Data<blas::vector<Type, blas::gpu_tag> > toGPU(Data<unsigned int > const& data){
	Data<blas::vector<Type, blas::gpu_tag> > data_gpu(data.numberOfBatches());
	std::size_t numClasses = numberOfClasses(data);
	for(std::size_t i = 0; i != data.numberOfBatches(); ++i){
		auto const& labels = data.batch(i);
		blas::matrix<Type> batch(labels.size(),numClasses, 0.0);
		for(std::size_t j = 0; j != labels.size(); ++j){
			batch(j,labels(j)) = Type(1);
		}
		data_gpu.batch(i) = blas::copy_to_gpu(batch);
	}
	data_gpu.shape() = data.shape();
	return data_gpu;
}

///\brief Transfers a labeled dataset from CPU to the GPU/OpenCL device
template<class Type, class I, class L>
LabeledData<blas::vector<Type, blas::gpu_tag>, blas::vector<Type, blas::gpu_tag> > toGPU(LabeledData<I,L> const& data){
	typedef LabeledData<blas::vector<Type, blas::gpu_tag>, blas::vector<Type, blas::gpu_tag> > DatasetType;
	return DatasetType(toGPU<Type>(data.inputs()),toGPU<Type>(data.labels()));
}


/** @*/
}

#endif
