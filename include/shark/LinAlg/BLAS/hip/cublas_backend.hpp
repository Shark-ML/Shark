//===========================================================================
/*!
 * 
 *
 * \brief       Handling of the cuda blas as hip backend
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

#ifndef REMORA_HIP_CUBLAS_DEVICE_HPP
#define REMORA_HIP_CUBLAS_DEVICE_HPP
#include "device.hpp"

#include <cublas_v2.h>

namespace remora{ namespace hip{
	
class cublas_error_category: public std::error_category{
public:
	const char* name() const noexcept{
		return "cuBLAS";
	}
	std::string message( int condition ) const{
		switch(condition){
			case CUBLAS_STATUS_SUCCESS:
				return "Success";
			case CUBLAS_STATUS_NOT_INITIALIZED:
				return "cublas library not initialized";
			case CUBLAS_STATUS_ALLOC_FAILED:
				return "Resource allocation failed";
			case CUBLAS_STATUS_INVALID_VALUE: 
				return "Unsupported numerical value was passed to function";
			case CUBLAS_STATUS_MAPPING_ERROR:
				return "Access to GPU memory space failed";
			case CUBLAS_STATUS_EXECUTION_FAILED:
				return "GPU program failed to execute";
			case CUBLAS_STATUS_INTERNAL_ERROR:
				return "An internal cublas operation failed";
			case CUBLAS_STATUS_NOT_SUPPORTED:
				return "Function not implemented";
			case CUBLAS_STATUS_ARCH_MISMATCH:
				return "Arch mismatch";
			default:
				return "Unknown error code: "+std::to_string(condition);
		}
	}
	static cublas_error_category& category(){
		static cublas_error_category cat;
		return cat;
	}
};
class cublas_exception:public std::system_error{
public:
	cublas_exception(cublasStatus_t code): std::system_error(code, cublas_error_category::category()){}
};

inline void check_cublas(cublasStatus_t code){
	if(code != CUBLAS_STATUS_SUCCESS)
		throw cublas_exception(code);
}
	
	
class blas_device{
public:
	blas_device(device& device):m_device(&device){
		m_device->set_device();
		check_cublas(cublasCreate(&m_handle));
	}
	blas_device(blas_device const&) = delete;
	blas_device(blas_device&& other){
		m_device = other.m_device;
		m_handle = other.m_handle;
		other.m_handle = nullptr;
	}
	
	blas_device& operator=(blas_device const&) = delete;
	blas_device& operator=(blas_device&& other){
		if(m_handle){
			m_device->set_device();
			check_cublas(cublasDestroy(m_handle));
		}
		m_device = other.m_device;
		m_handle = other.m_handle;
		other.m_handle = 0;
		return *this;
	}
	~blas_device(){
		if(m_handle){
			m_device->set_device();
			check_cublas(cublasDestroy(m_handle));
		}
	}
	

	
	////////////GEMM////////////
	
	void gemm(
		bool transA, bool transB,
		std::size_t m, std::size_t n, std::size_t k,
		float alpha, 
		float const* A, std::size_t ldA,
		float const* B, std::size_t ldB,
		float beta,
		float* C, std::size_t ldC,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasSgemm(
			m_handle,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			transB? CUBLAS_OP_T : CUBLAS_OP_N,
			int(m), int(n), int(k),
			&alpha,
			A, int(ldA),
			B, int(ldB),
			&beta,
			C, int(ldC)
		));
	}
	void gemm(
		bool transA, bool transB,
		std::size_t m, std::size_t n, std::size_t k,
		double alpha, 
		double const* A, std::size_t ldA,
		double const* B, std::size_t ldB,
		double beta,
		double* C, std::size_t ldC,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasDgemm(
			m_handle,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			transB? CUBLAS_OP_T : CUBLAS_OP_N,
			int(m), int(n), int(k),
			&alpha,
			A, int(ldA),
			B, int(ldB),
			&beta,
			C, int(ldC)
		));
	}
	
	////////////GEMV////////////	
	
	void gemv(
		bool transA,
		std::size_t m, std::size_t n,
		float alpha, 
		float const* A, std::size_t ldA,
		float const* x, std::size_t stridex,
		float beta,
		float* v, std::size_t stridev,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasSgemv(
			m_handle,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			int(m), int(n),
			&alpha,
			A, int(ldA),
			x, int(stridex),
			&beta,
			v, int(stridev)
		));
	}
	void gemv(
		bool transA,
		std::size_t m, std::size_t n,
		double alpha, 
		double const* A, std::size_t ldA,
		double const* x, std::size_t stridex,
		double beta,
		double* v, std::size_t stridev,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasDgemv(
			m_handle,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			int(m), int(n),
			&alpha,
			A, int(ldA),
			x, int(stridex),
			&beta,
			v, int(stridev)
		));
	}
	
	////////////TRMM////////////
	
	void trmm(
		bool leftA, bool upperA, bool transA, bool unitA,
		std::size_t m, std::size_t n,
		float alpha, 
		float const* A, std::size_t ldA,
		float* B, std::size_t ldB,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasStrmm(
			m_handle,
			leftA? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			unitA? CUBLAS_DIAG_UNIT: CUBLAS_DIAG_NON_UNIT,
			int(m), int(n),
			&alpha,
			A, int(ldA),
			B, int(ldB),
			B, int(ldB)
		));
	}
	void trmm(
		bool leftA, bool upperA, bool transA, bool unitA,
		std::size_t m, std::size_t n,
		double alpha, 
		double const* A, std::size_t ldA,
		double* B, std::size_t ldB,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasDtrmm(
			m_handle,
			leftA? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			unitA? CUBLAS_DIAG_UNIT: CUBLAS_DIAG_NON_UNIT,
			int(m), int(n),
			&alpha,
			A, int(ldA),
			B, int(ldB),
			B, int(ldB)
		));
	}
	
	////////////TRMV////////////
	
	void trmv(
		bool upperA, bool transA, bool unitA,
		std::size_t n,
		float const* A, std::size_t ldA,
		float* v, std::size_t stridev,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasStrmv(
			m_handle,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			unitA? CUBLAS_DIAG_UNIT: CUBLAS_DIAG_NON_UNIT,
			int(n),
			A, int(ldA),
			v, int(stridev)
		));
	}
	void trmv(
		bool upperA, bool transA, bool unitA,
		std::size_t n,
		double const* A, std::size_t ldA,
		double* v, std::size_t stridev,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasDtrmv(
			m_handle,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			unitA? CUBLAS_DIAG_UNIT: CUBLAS_DIAG_NON_UNIT,
			int(n),
			A, int(ldA),
			v, int(stridev)
		));
	}
	
	////////////TRSM////////////
	
	void trsm(
		bool leftA, bool upperA, bool transA, bool unitA,
		std::size_t m, std::size_t n,
		float alpha, 
		float const* A, std::size_t ldA,
		float* B, std::size_t ldB,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasStrsm(
			m_handle,
			leftA? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			unitA? CUBLAS_DIAG_UNIT: CUBLAS_DIAG_NON_UNIT,
			int(m), int(n),
			&alpha,
			A, int(ldA),
			B, int(ldB)
		));
	}
	void trsm(
		bool leftA, bool upperA, bool transA, bool unitA,
		std::size_t m, std::size_t n,
		double alpha, 
		double const* A, std::size_t ldA,
		double* B, std::size_t ldB,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasDtrsm(
			m_handle,
			leftA? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			unitA? CUBLAS_DIAG_UNIT: CUBLAS_DIAG_NON_UNIT,
			int(m), int(n),
			&alpha,
			A, int(ldA),
			B, int(ldB)
		));
	}
	
	////////////TRSV////////////
	
	void trsv(
		bool upperA, bool transA, bool unitA,
		std::size_t n,
		float const* A, std::size_t ldA,
		float* v, std::size_t stridev,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasStrsv(
			m_handle,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			unitA? CUBLAS_DIAG_UNIT: CUBLAS_DIAG_NON_UNIT,
			int(n),
			A, int(ldA),
			v, int(stridev)
		));
	}
	void trsv(
		bool upperA, bool transA, bool unitA,
		std::size_t n,
		double const* A, std::size_t ldA,
		double* v, std::size_t stridev,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasDtrsv(
			m_handle,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			unitA? CUBLAS_DIAG_UNIT: CUBLAS_DIAG_NON_UNIT,
			int(n),
			A, int(ldA),
			v, int(stridev)
		));
	}
	
	
	
	////////////SYRK////////////
	
	void syrk(
		bool upperA, bool transA,
		std::size_t n, std::size_t k,
		float alpha,
		float const* A, std::size_t ldA,
		float beta,
		float* C, std::size_t ldC,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasSsyrk(
			m_handle,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			int(n), int(k),
			&alpha,
			A, int(ldA),
			&beta,
			C, int(ldC)
		));
	}
	
	void syrk(
		bool upperA, bool transA,
		std::size_t n, std::size_t k,
		double alpha,
		double const* A, std::size_t ldA,
		double beta,
		double* C, std::size_t ldC,
		stream const& s
	){
		prepare(s);
		check_cublas(cublasDsyrk(
			m_handle,
			upperA? CUBLAS_FILL_MODE_UPPER: CUBLAS_FILL_MODE_LOWER,
			transA? CUBLAS_OP_T : CUBLAS_OP_N,
			int(n), int(k),
			&alpha,
			A, int(ldA),
			&beta,
			C, int(ldC)
		));
	}
	
private:
	void prepare(stream const& s){
		m_device->set_device();
		check_cublas(cublasSetStream(m_handle, (cudaStream_t) s.handle()));
	}
	device* m_device;
 	cublasHandle_t m_handle;
};

inline blas_device& get_blas(device& device){
	thread_local static std::vector<std::pair<int, blas_device> > blas_devices;
	for(std::size_t i = 0; i != blas_devices.size(); ++i){
		if(blas_devices[i].first == device.device_id())
			return blas_devices[i].second;
	}
	blas_devices.emplace_back(device.device_id(), blas_device(device));
	return blas_devices.back().second;
}

}}
#endif