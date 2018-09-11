//===========================================================================
/*!
 * 
 *
 * \brief  Convolution of an image with a filter
 * 
 * 
 *
 * \author      O.Krause
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

#ifndef SHARK_CORE_IMAGE_HIP_CONVOLUTION_CUDA_H
#define SHARK_CORE_IMAGE_HIP_CONVOLUTION_CUDA_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
#include <shark/Core/Images/Enums.h>
#include <shark/Core/Images/Hip/CudnnBackend.h>


namespace shark{
namespace image{

template<class M1, class M2>
void convolution(
	blas::matrix_expression<M1, blas::hip_tag> const& images, 
	blas::matrix_expression<M1, blas::hip_tag> const& filters, 
	blas::matrix_expression<M2, blas::hip_tag>& results,
	Shape const& shapeImage,
	Shape const& shapeFilters,
	Shape const& shapeResults,
	std::size_t paddingHeight,
	std::size_t paddingWidth,
	ImageFormat imageFormat,
	ImageFormat filterFormat,
	bool flipFilters
){
	SHARK_ASSERT(imageFormat == ImageFormat::NCHW);
	SHARK_ASSERT(filterFormat == ImageFormat::NCHW);
	typedef typename M1::value_type T;
	
	auto& device = results().queue();
	auto handle = hip::getCudnn(device).handle();
	cudaStream_t stream = blas::hip::get_stream(device).handle();
	device.set_device();
	hip::checkCudnn(cudnnSetStream(handle, stream));
	
	//Create convolution descriptors
	hip::CudnnTensorDescriptor<T> imageDesc(images().size1(), shapeImage[0], shapeImage[1], shapeImage[2]);
	hip::CudnnFilterDescriptor<T> filterDesc(filters().size1(), shapeFilters[0], shapeFilters[1], shapeFilters[2]);
	hip::CudnnTensorDescriptor<T> resultDesc(images().size1(), shapeResults[0], shapeResults[1], shapeResults[2]);
	hip::CudnnConvolutionDescriptor<T> convDesc(paddingHeight/2, paddingWidth/2, flipFilters);
	
	//acquire workspace and choose algorithm
	auto& workspace = hip::getCudnnWorkspace(device);
	cudnnConvolutionFwdAlgo_t algo;
	hip::checkCudnn(cudnnGetConvolutionForwardAlgorithm(
		handle,
		imageDesc.handle(), filterDesc.handle(), convDesc.handle(), resultDesc.handle(),
		CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
		workspace.size(), &algo
	));
	
	
	//run algorithm
	T alpha = T(1.0);
	T beta = T(1.0);
	hip::checkCudnn(cudnnConvolutionForward(
		handle,
		&alpha, imageDesc.handle(), images().raw_storage().values,
		filterDesc.handle(), filters().raw_storage().values,
		convDesc.handle(), algo, workspace.get(), workspace.size(), 
		&beta, resultDesc.handle(), results().raw_storage().values
	));
}

template<class M1, class M2>
void convolutionBackwardInputs(
	blas::matrix_expression<M1, blas::hip_tag> const& deltas, 
	blas::matrix_expression<M1, blas::hip_tag> const& filters, 
	blas::matrix_expression<M2, blas::hip_tag>& derivatives,
	Shape const& shapeDelta,
	Shape const& shapeFilters,
	Shape const& shapeDerivatives,
	std::size_t paddingHeight,
	std::size_t paddingWidth,
	bool flipFilters
){
	typedef typename M1::value_type T;
	auto& device = derivatives().queue();
	auto handle = hip::getCudnn(device).handle();
	cudaStream_t stream = blas::hip::get_stream(device).handle();
	device.set_device();
	hip::checkCudnn(cudnnSetStream(handle, stream));
	
	//Create convolution descriptors
	hip::CudnnTensorDescriptor<T> deltaDesc(deltas().size1(), shapeDelta[0], shapeDelta[1], shapeDelta[2]);
	hip::CudnnFilterDescriptor<T> filterDesc(filters().size1(), shapeFilters[0], shapeFilters[1], shapeFilters[2]);
	hip::CudnnTensorDescriptor<T> derivativeDesc(deltas().size1(), shapeDerivatives[0], shapeDerivatives[1], shapeDerivatives[2]);
	hip::CudnnConvolutionDescriptor<T> convDesc(paddingHeight/2, paddingWidth/2, flipFilters);
	
	//acquire workspace and choose algorithm
	auto& workspace = hip::getCudnnWorkspace(device);
	cudnnConvolutionBwdDataAlgo_t algo;
	hip::checkCudnn(cudnnGetConvolutionBackwardDataAlgorithm(
		handle, filterDesc.handle(), deltaDesc.handle(), convDesc.handle(), derivativeDesc.handle(),
		CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
		workspace.size(), &algo
	));
	
	//run algorithm
	T alpha = T(1.0);
	T beta = T(1.0);
	hip::checkCudnn(cudnnConvolutionBackwardData(
		handle,
		&alpha, filterDesc.handle(), filters().raw_storage().values,
		deltaDesc.handle(), deltas().raw_storage().values,
		convDesc.handle(), algo, workspace.get(), workspace.size(), 
		&beta, derivativeDesc.handle(), derivatives().raw_storage().values
	));
}

template<class M1, class M2>
void convolutionBackwardFilters(
	blas::matrix_expression<M1, blas::hip_tag> const& images,
	blas::matrix_expression<M1, blas::hip_tag> const& deltas,
	blas::matrix_expression<M2, blas::hip_tag>& filterDerivative,
	Shape const& shapeImage,
	Shape const& shapeDelta,
	Shape const& shapeFilters,
	std::size_t paddingHeight,
	std::size_t paddingWidth,
	bool flipFilters
){
	typedef typename M1::value_type T;
	auto& device = filterDerivative().queue();
	auto handle = hip::getCudnn(device).handle();
	cudaStream_t stream = blas::hip::get_stream(device).handle();
	device.set_device();
	hip::checkCudnn(cudnnSetStream(handle, stream));
	
	//Create convolution descriptors
	hip::CudnnTensorDescriptor<T> imageDesc(images().size1(), shapeImage[0], shapeImage[1], shapeImage[2]);
	hip::CudnnTensorDescriptor<T> deltaDesc(images().size1(), shapeDelta[0], shapeDelta[1], shapeDelta[2]);
	hip::CudnnFilterDescriptor<T> derivativeDesc(images().size1(), shapeFilters[0], shapeFilters[1], shapeFilters[2]);
	hip::CudnnConvolutionDescriptor<T> convDesc(paddingHeight/2, paddingWidth/2, flipFilters);
	
	//acquire workspace and choose algorithm
	auto& workspace = hip::getCudnnWorkspace(device);
	cudnnConvolutionBwdFilterAlgo_t algo;
	hip::checkCudnn(cudnnGetConvolutionBackwardFilterAlgorithm(
		handle, imageDesc.handle(), deltaDesc.handle(), convDesc.handle(), derivativeDesc.handle(),
		CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
		workspace.size(), &algo
	));
	
	//run algorithm
	T alpha = T(1.0);
	T beta = T(1.0);
	hip::checkCudnn(cudnnConvolutionBackwardFilter(
		handle,
		&alpha, imageDesc.handle(), images().raw_storage().values,
		deltaDesc.handle(), deltas().raw_storage().values,
		convDesc.handle(), algo, workspace.get(), workspace.size(), 
		&beta, derivativeDesc.handle(), filterDerivative().raw_storage().values
	));
}}}

#endif