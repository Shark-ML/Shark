//===========================================================================
/*!
 * 
 *
 * \brief       Handling of the cuda cudnn as hip backend
 *
 * \author      O. Krause
 * \date        2018
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

#ifndef SHARK_CORE_IMAGE_HIP_CUDNN_BACKEND_H
#define SHARK_CORE_IMAGE_HIP_CUDNN_BACKEND_H
#include <cudnn.h>
#include <shark/LinAlg/Base.h>
#include <shark/Core/Images/Enums.h>
#include <shark/Core/Exception.h>

namespace shark{
namespace image{
namespace hip{
	
class CudnnErrorCategory: public std::error_category{
public:
	const char* name() const noexcept{
		return "cudnn";
	}
	std::string message( int condition ) const{
		switch(condition){
			case CUDNN_STATUS_SUCCESS:
				return "Success";
			case CUDNN_STATUS_NOT_INITIALIZED:
				return "cudnn library not initialized";
			case CUDNN_STATUS_ALLOC_FAILED:
				return "Resource allocation failed";
			case CUDNN_STATUS_BAD_PARAM: 
				return "An incorrect value or parameter was passed to the function.";
			case CUDNN_STATUS_ARCH_MISMATCH:
				return "Compute Capability must be at least 3.0";
			case CUDNN_STATUS_MAPPING_ERROR:
				return "Access to GPU memory space failed";
			case CUDNN_STATUS_EXECUTION_FAILED:
				return "GPU program failed to execute";
			case CUDNN_STATUS_INTERNAL_ERROR:
				return "An internal cudnn operation failed";
			case CUDNN_STATUS_NOT_SUPPORTED:
				return "Function not implemented";
			case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
				return "Runtime library required by RNN calls cannot be found";
			case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
				return "Pipeline not yet empty";
			case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
				return "Numerical overflow occurred during the GPU kernel execution.";
			default:
				return "Unknown error code: "+std::to_string(condition);
		}
	}
	static CudnnErrorCategory& category(){
		static CudnnErrorCategory cat;
		return cat;
	}
};
class CudnnException:public std::system_error{
public:
	CudnnException(cudnnStatus_t code): std::system_error(code, CudnnErrorCategory::category()){}
};

inline void checkCudnn(cudnnStatus_t code){
	if(code != CUDNN_STATUS_SUCCESS)
		throw CudnnException(code);
}

template<class T>
struct CudnnDataType;

template<>
struct CudnnDataType<float>{
	static const cudnnDataType_t value = CUDNN_DATA_FLOAT ;
};

template<>
struct CudnnDataType<double>{
	static const cudnnDataType_t value = CUDNN_DATA_DOUBLE;
};


template<class T>
class CudnnTensorDescriptor{
public:
	CudnnTensorDescriptor(
		std::size_t n, std::size_t c, std::size_t h, std::size_t w
	){
		checkCudnn(cudnnCreateTensorDescriptor(&m_handle));
		checkCudnn(cudnnSetTensor4dDescriptor(
			m_handle, CUDNN_TENSOR_NCHW, CudnnDataType<T>::value,
			int(n), int(c), int(h), int(w)
		));
	}
	CudnnTensorDescriptor(CudnnTensorDescriptor const&) = delete;
	CudnnTensorDescriptor(CudnnTensorDescriptor&& other){
		m_handle = other.m_handle;
		other.m_handle = nullptr;
	}
	
	CudnnTensorDescriptor& operator=(CudnnTensorDescriptor const&) = delete;
	CudnnTensorDescriptor& operator=(CudnnTensorDescriptor&& other){
		checkCudnn(cudnnDestroyTensorDescriptor(m_handle));
		m_handle = other.m_handle;
		other.m_handle = nullptr;
		return *this;
	}
	~CudnnTensorDescriptor(){
		checkCudnn(cudnnDestroyTensorDescriptor(m_handle));
	}
	cudnnTensorDescriptor_t handle()const{
		return m_handle;
	}
private:
	cudnnTensorDescriptor_t m_handle;
};

template<class T>
class CudnnConvolutionDescriptor{
public:
	CudnnConvolutionDescriptor(
		std::size_t paddingHeight, std::size_t paddingWidth, bool flipFilters
	){
		checkCudnn(cudnnCreateConvolutionDescriptor(&m_handle));
		checkCudnn(cudnnSetConvolution2dDescriptor(
			m_handle, paddingHeight, paddingWidth,
			1, 1, 1, 1,
			flipFilters? CUDNN_CONVOLUTION :  CUDNN_CROSS_CORRELATION,
			CudnnDataType<T>::value
		));
	}
	CudnnConvolutionDescriptor(CudnnConvolutionDescriptor const&) = delete;
	CudnnConvolutionDescriptor(CudnnConvolutionDescriptor&& other){
		m_handle = other.m_handle;
		other.m_handle = nullptr;
	}
	
	CudnnConvolutionDescriptor& operator=(CudnnConvolutionDescriptor const&) = delete;
	CudnnConvolutionDescriptor& operator=(CudnnConvolutionDescriptor&& other){
		checkCudnn(cudnnDestroyConvolutionDescriptor(m_handle));
		m_handle = other.m_handle;
		other.m_handle = nullptr;
		return *this;
	}
	~CudnnConvolutionDescriptor(){
		checkCudnn(cudnnDestroyConvolutionDescriptor(m_handle));
	}
	cudnnConvolutionDescriptor_t handle()const{
		return m_handle;
	}
private:
	cudnnConvolutionDescriptor_t m_handle;
};

template<class T>
class CudnnFilterDescriptor{
public:
	CudnnFilterDescriptor(
		std::size_t numFilters, std::size_t numChannels, std::size_t h, std::size_t w
	){
		checkCudnn(cudnnCreateFilterDescriptor(&m_handle));
		checkCudnn(cudnnSetFilter4dDescriptor(
			m_handle, CudnnDataType<T>::value, CUDNN_TENSOR_NCHW,
			int(numFilters), int(numChannels), int(h), int(w)
		));
	}
	CudnnFilterDescriptor(CudnnFilterDescriptor const&) = delete;
	CudnnFilterDescriptor(CudnnFilterDescriptor&& other){
		m_handle = other.m_handle;
		other.m_handle = nullptr;
	}
	
	CudnnFilterDescriptor& operator=(CudnnFilterDescriptor const&) = delete;
	CudnnFilterDescriptor& operator=(CudnnFilterDescriptor&& other){
		checkCudnn(cudnnDestroyFilterDescriptor(m_handle));
		m_handle = other.m_handle;
		other.m_handle = nullptr;
		return *this;
	}
	~CudnnFilterDescriptor(){
		checkCudnn(cudnnDestroyFilterDescriptor(m_handle));
	}
	cudnnFilterDescriptor_t handle()const{
		return m_handle;
	}
private:
	cudnnFilterDescriptor_t m_handle;
};

class CudnnDevice{
public:
	CudnnDevice(blas::hip::device& device):m_device(&device){
		m_device->set_device();
		checkCudnn(cudnnCreate(&m_handle));
	}
	CudnnDevice(CudnnDevice const&) = delete;
	CudnnDevice(CudnnDevice&& other){
		m_device = other.m_device;
		m_handle = other.m_handle;
		other.m_handle = nullptr;
	}
	
	CudnnDevice& operator=(CudnnDevice const&) = delete;
	CudnnDevice& operator=(CudnnDevice&& other){
		if(m_handle){
			m_device->set_device();
			checkCudnn(cudnnDestroy(m_handle));
		}
		m_device = other.m_device;
		m_handle = other.m_handle;
		other.m_handle = 0;
		return *this;
	}
	~CudnnDevice(){
		if(m_handle){
			m_device->set_device();
			checkCudnn(cudnnDestroy(m_handle));
		}
	}
	
	cudnnHandle_t handle()const{
		return m_handle;
	}
	
private:
	blas::hip::device* m_device;
 	cudnnHandle_t m_handle;
};

inline CudnnDevice& getCudnn(blas::hip::device& device){
	thread_local static std::vector<std::pair<int, CudnnDevice> > cudnnDevices;
	for(std::size_t i = 0; i != cudnnDevices.size(); ++i){
		if(cudnnDevices[i].first == device.device_id())
			return cudnnDevices[i].second;
	}
	cudnnDevices.emplace_back(device.device_id(), CudnnDevice(device));
	return cudnnDevices.back().second;
}

inline blas::hip::buffer<char>& getCudnnWorkspace(blas::hip::device& device){
	thread_local static std::vector<std::pair<int, blas::hip::buffer<char> > > workspaces;
	for(std::size_t i = 0; i != workspaces.size(); ++i){
		if(workspaces[i].first == device.device_id())
			return workspaces[i].second;
	}
	
	std::size_t maxWorkspaceSize = 100 * 1024 * 1024;
	
	workspaces.emplace_back(device.device_id(), blas::hip::buffer<char>(maxWorkspaceSize, device));
	return workspaces.back().second;
}

}}}
#endif