/*!
 * 
 *
 * \brief       Generation of random variates on hip devices
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
#ifndef REMORA_KERNELS_HIP_RANDOM_HPP
#define REMORA_KERNELS_HIP_RANDOM_HPP

#include "../../proxy_expressions.hpp"
#include "../../detail/traits.hpp"
#include <random>
#include <cstdint>

namespace remora{ namespace hip{

__device__ uint RotL_32_device(uint x, uint N){
    return (x << (N & 31)) | (x >> ((32-N) & 31));
}

__device__ uint4 rng4_32_device(uint4 X, uint4 k){    
	enum r123_enum_threefry32x4 {
		R_32x4_0_0=10, R_32x4_0_1=26,
		R_32x4_1_0=11, R_32x4_1_1=21,
		R_32x4_2_0=13, R_32x4_2_1=27,
		R_32x4_3_0=23, R_32x4_3_1= 5,
		R_32x4_4_0= 6, R_32x4_4_1=20,
		R_32x4_5_0=17, R_32x4_5_1=11,
		R_32x4_6_0=25, R_32x4_6_1=10,
		R_32x4_7_0=18, R_32x4_7_1=20
	};
	
	uint k_hi =  0x1BD11BDA;
	k_hi ^= k.x;
	k_hi ^= k.y;
	k_hi ^= k.z; 
	k_hi ^= k.w;

	X.x += k.x;
	X.y += k.y;
	X.z += k.z;
	X.w += k.w;
	X.x += X.y; X.y = RotL_32_device(X.y,R_32x4_0_0); X.y ^= X.x;
	X.z += X.w; X.w = RotL_32_device(X.w,R_32x4_0_1); X.w ^= X.z;
	X.x += X.w; X.w = RotL_32_device(X.w,R_32x4_1_0); X.w ^= X.x;
	X.z += X.y; X.y = RotL_32_device(X.y,R_32x4_1_1); X.y ^= X.z;
	X.x += X.y; X.y = RotL_32_device(X.y,R_32x4_2_0); X.y ^= X.x;
	X.z += X.w; X.w = RotL_32_device(X.w,R_32x4_2_1); X.w ^= X.z;
	X.x += X.w; X.w = RotL_32_device(X.w,R_32x4_3_0); X.w ^= X.x;
	X.z += X.y; X.y = RotL_32_device(X.y,R_32x4_3_1); X.y ^= X.z;
	X.x += k.y;
	X.y += k.z;
	X.z += k.w;
	X.w += k_hi;
	X.w += 1;

	X.x += X.y; X.y = RotL_32_device(X.y,R_32x4_4_0); X.y ^= X.x;
	X.z += X.w; X.w = RotL_32_device(X.w,R_32x4_4_1); X.w ^= X.z;
	X.x += X.w; X.w = RotL_32_device(X.w,R_32x4_5_0); X.w ^= X.x;
	X.z += X.y; X.y = RotL_32_device(X.y,R_32x4_5_1); X.y ^= X.z;
	X.x += X.y; X.y = RotL_32_device(X.y,R_32x4_6_0); X.y ^= X.x;
	X.z += X.w; X.w = RotL_32_device(X.w,R_32x4_6_1); X.w ^= X.z;
	X.x += X.w; X.w = RotL_32_device(X.w,R_32x4_7_0); X.w ^= X.x;
	X.z += X.y; X.y = RotL_32_device(X.y,R_32x4_7_1); X.y ^= X.z;
	X.x += k.z;
	X.y += k.w;
	X.z += k_hi;
	X.w += k.x;
	X.w += 2;

	X.x += X.y; X.y = RotL_32_device(X.y,R_32x4_0_0); X.y ^= X.x;
	X.z += X.w; X.w = RotL_32_device(X.w,R_32x4_0_1); X.w ^= X.z;
	X.x += X.w; X.w = RotL_32_device(X.w,R_32x4_1_0); X.w ^= X.x;
	X.z += X.y; X.y = RotL_32_device(X.y,R_32x4_1_1); X.y ^= X.z;
	X.x += X.y; X.y = RotL_32_device(X.y,R_32x4_2_0); X.y ^= X.x;
	X.z += X.w; X.w = RotL_32_device(X.w,R_32x4_2_1); X.w ^= X.z;
	X.x += X.w; X.w = RotL_32_device(X.w,R_32x4_3_0); X.w ^= X.x;
	X.z += X.y; X.y = RotL_32_device(X.y,R_32x4_3_1); X.y ^= X.z;
	X.x += k.w;
	X.y += k_hi;
	X.w += k.x;
	X.w += k.y;
	X.w += 3;
	return X;
}
//source for internal routines to generate random numbers
__device__ float4 uniform4_32_device(float low, float high, uint4 x){
	float4 z = make_float4( float(x.x) / 0xffffffff, float(x.y) / 0xffffffff, float(x.z) / 0xffffffff, float(x.w) / 0xffffffff);
	float4 result;
	result.x = nextafterf(low + z.x * (high - low), low);
	result.y = nextafterf(low + z.y * (high - low), low);
	result.z = nextafterf(low + z.z * (high - low), low);
	result.w = nextafterf(low + z.w * (high - low), low);
	return result;
}
__device__ float4 normal4_32_device(float mean, float stddev, uint4 x){
	float4 u = uniform4_32_device(0,1,x);
	float2 r = make_float2(sqrtf(-2 * logf(u.x)),sqrtf(-2 * logf(u.y)));
	float2 phi_sin;
	float2 phi_cos;
	sincosf(2*(float)M_PI * u.z, &phi_sin.x, &phi_cos.x);
	sincosf(2*(float)M_PI * u.w, &phi_sin.y, &phi_cos.y);
	float4 result;
	result.x = mean + stddev * r.x * phi_sin.x;
	result.y = mean + stddev * r.x * phi_cos.x;
	result.z = mean + stddev * r.y * phi_sin.y;
	result.w = mean + stddev * r.y * phi_cos.y;
	return result;
}
__device__ uint4 discrete4_32_device(uint4 key, uint4 ctr, uint N, uint4 inc) {
	uint max_valid = (0xffffffff / N) * N;
	int num_valid = 0;
	uint res[4];
	//~ while(num_valid < 4){
		uint4 x = rng4_32_device(ctr, key);
		//~ if(x.x < max_valid){
		    res[num_valid] = x.x % N;
		    ++num_valid;
		//~ }
		//~ if(num_valid < 4 && x.y < max_valid){
		    res[num_valid] = x.y % N;
		    ++num_valid;
		//~ }
		//~ if(num_valid < 4 && x.z < max_valid){
		    res[num_valid] = x.z % N;
		    ++num_valid;
		//~ }
		//~ if(num_valid < 4 && x.w < max_valid){
		    res[num_valid] = x.w % N;
		    ++num_valid;
		//~ }
		//~ ctr.x += inc.x;
		//~ ctr.y += inc.y;
		//~ ctr.z += inc.z;
		//~ ctr.w += inc.w;
	//~ }
	return make_uint4(res[0],res[1],res[2],res[3]);
}
//kernel routines
template<class MatA>
__global__ void generate_uniform32_kernel(hipLaunchParm lp, MatA A, uint4 key, float low, float high, size_t size2) {
	std::size_t id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
	std::size_t id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y) * 4;
	uint4 ctr = make_uint4(id_x & 0x0000ffff, id_x >> 32, id_y & 0x0000ffff, id_y >> 32);
	
	uint4 x = rng4_32_device(ctr, key);
	float4 u = uniform4_32_device(low, high, x);
	
	A(id_x, id_y) = u.x;
	if(id_y + 1 < size2) A(id_x, id_y + 1) = u.y;
	if(id_y + 2 < size2) A(id_x, id_y + 2) = u.z;
	if(id_y + 3 < size2) A(id_x, id_y + 3) = u.w;
}
template<class MatA>
__global__ void generate_normal32_kernel(hipLaunchParm lp, MatA A, uint4 key, float mean, float stddev, size_t size2) {
	std::size_t id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
	std::size_t id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y) * 4;
	uint4 ctr = make_uint4(id_x & 0x0000ffff, id_x >> 32, id_y & 0x0000ffff, id_y >> 32);
	
	uint4 x = rng4_32_device(ctr, key);
	float4 u = normal4_32_device(mean, stddev, x);
	
	A(id_x, id_y) = u.x;
	if(id_y + 1 < size2) A(id_x, id_y + 1) = u.y;
	if(id_y + 2 < size2) A(id_x, id_y + 2) = u.z;
	if(id_y + 3 < size2) A(id_x, id_y + 3)= u.w;
}
template<class MatA>
__global__ void generate_discrete32_kernel(hipLaunchParm lp, MatA A, uint4 key, int low, int high, size_t size2) {
	typedef typename std::remove_reference<typename MatA::result_type>::type value_type;
	std::size_t id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
	std::size_t id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y) * 4;
	uint4 ctr = make_uint4(id_x & 0x0000ffff, id_x >> 32, id_y & 0x0000ffff, id_y >> 32);

	uint4 inc = make_uint4(0, id_y & 0x0000ffff, 0, id_y >> 32);
	uint4 z = discrete4_32_device(key, ctr, high - low + 1,inc); 
	
	int valid_required = ::min((size_t)4, size2 - id_y);
	A(id_x, id_y) = low + value_type(z.x);
	if(1 < valid_required) A(id_x, id_y + 1) = low + value_type(z.y);
	if(2 < valid_required) A(id_x, id_y + 2) = low + value_type(z.z);
	if(3 < valid_required) A(id_x, id_y + 3) = low + value_type(z.w);
	//~ A(id_x, id_y) = low;
	//~ if(1 < valid_required) A(id_x, id_y + 1) = high;
	//~ if(2 < valid_required) A(id_x, id_y + 2) = high - low + 1;
	//~ if(3 < valid_required) A(id_x, id_y + 3) =valid_required;
}

enum class RandomKernels{
	Uniform,
	Normal,
	Discrete
};
	
template<class Rng, class E, class T>
void run_random_kernel(RandomKernels kernel, Rng& rng, 
	E& expression, T arg1, T arg2
){
	static_assert(sizeof(T) == 4, "only 32 bit random number types are currently supported");
	
	//seed key from rng (this is the only possible entropy source for the rng!)
	std::uniform_int_distribution<uint32_t> dist(0,0xffffffff);
	static const std::size_t vectorSize = 4; 
	uint4 key = make_uint4(dist(rng),dist(rng),dist(rng),dist(rng));
	auto& device = expression().queue();
	std::size_t blockSize2 = std::min<std::size_t>(16, device.warp_size());
	std::size_t blockSize1 = std::min<std::size_t>(16, device.warp_size() / blockSize2);
	if(expression().size1() < blockSize1){
		blockSize2 = expression().queue().warp_size();
		blockSize1 = 1;
	}
	std::size_t numBlocks1 = (expression().size1()  + blockSize1 - 1) / blockSize1;
	std::size_t numBlocks2 = (expression().size2() + vectorSize * blockSize2 - 1) / (vectorSize * blockSize2);//each thread computes 4 elements at a time
	auto stream = get_stream(device).handle();
	if(kernel == RandomKernels::Uniform){
		hipLaunchKernel(
			hip::generate_uniform32_kernel, 
			dim3(numBlocks1, numBlocks2), dim3(blockSize1, blockSize2), 0, stream,
			expression().elements(), key, arg1, arg2, expression().size2()
		);
	}
	if(kernel == RandomKernels::Normal){
		hipLaunchKernel(
			hip::generate_normal32_kernel, 
			dim3(numBlocks1, numBlocks2), dim3(blockSize1, blockSize2), 0, stream,
			expression().elements(), key, arg1, arg2, expression().size2()
		);
	}
	if(kernel == RandomKernels::Discrete){
		hipLaunchKernel(
			hip::generate_discrete32_kernel, 
			dim3(numBlocks1, numBlocks2), dim3(blockSize1, blockSize2), 0, stream,
			expression().elements(), key, arg1, arg2, expression().size2()
		);
	}
}
}
namespace bindings{
template<class V, class Rng>
void generate_normal(
	vector_expression<V, hip_tag>& v,
	Rng& rng,
	typename V::value_type mean,
	typename V::value_type variance
) {
	auto mat_proxy = to_matrix(v,1, v().size());
	hip::run_random_kernel(hip::RandomKernels::Normal, rng, mat_proxy, mean, std::sqrt(variance));
}

template<class M, class Rng>
void generate_normal(
	matrix_expression<M, hip_tag>& m,
	Rng& rng,
	typename M::value_type mean,
	typename M::value_type variance
) {
	hip::run_random_kernel(hip::RandomKernels::Normal, rng, m, mean, std::sqrt(variance));
}

template<class V, class Rng>
void generate_uniform(
	vector_expression<V, hip_tag>& v,
	Rng& rng,
	typename V::value_type low,
	typename V::value_type high
) {
	auto mat_proxy = to_matrix(v,1, v().size());
	hip::run_random_kernel(hip::RandomKernels::Uniform, rng, mat_proxy, low, high);
}

template<class M, class Rng>
void generate_uniform(
	matrix_expression<M, hip_tag>& m,
	Rng& rng,
	typename M::value_type low,
	typename M::value_type high
) {
	hip::run_random_kernel(hip::RandomKernels::Uniform, rng, m, low, high);
}

template<class V, class Rng>
void generate_discrete(
	vector_expression<V, hip_tag>& v,
	Rng& rng,
	int low,
	int high
) {
	auto mat_proxy = to_matrix(v,1, v().size());
	hip::run_random_kernel(hip::RandomKernels::Discrete, rng, mat_proxy, low, high);
}

template<class M, class Rng>
void generate_discrete(
	matrix_expression<M, hip_tag>& m,
	Rng& rng,
	int low,
	int high
) {
	hip::run_random_kernel(hip::RandomKernels::Discrete, rng, m, low, high);
}

}}
#endif