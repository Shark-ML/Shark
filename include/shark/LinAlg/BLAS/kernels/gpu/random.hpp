/*!
 * 
 *
 * \brief       Generation of random variates on opencl devices
 *
 * \author      O. Krause
 * \date        2017
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
#ifndef REMORA_KERNELS_GPU_RANDOM_HPP
#define REMORA_KERNELS_GPU_RANDOM_HPP

#include <boost/compute/kernel.hpp>
#include <boost/compute/utility/program_cache.hpp>
#include <random>
#include <cstdint>

namespace remora{ namespace bindings{
	
inline boost::compute::kernel get_random_kernel32(std::string const& kernelname, boost::compute::context const& ctx){
        const char source[] =
		//rng4_32 is based on the implementation of the threefry algorithm by:
		// Copyright 2010-2012, D. E. Shaw Research.
		// All rights reserved.

		// Redistribution and use in source and binary forms, with or without
		// modification, are permitted provided that the following conditions are
		// met:

		// * Redistributions of source code must retain the above copyright
		//   notice, this list of conditions, and the following disclaimer.

		// * Redistributions in binary form must reproduce the above copyright
		//   notice, this list of conditions, and the following disclaimer in the
		//   documentation and/or other materials provided with the distribution.

		// * Neither the name of D. E. Shaw Research nor the names of its
		//   contributors may be used to endorse or promote products derived from
		//   this software without specific prior written permission.

		// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
		// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
		// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
		// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
		// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
		// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
		// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
		// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
		// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
		// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
		// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
		"#define SKEIN_KS_PARITY_32 0x1BD11BDA\n"
		"enum r123_enum_threefry32x4 {\n"
		"    R_32x4_0_0=10, R_32x4_0_1=26,\n"
		"    R_32x4_1_0=11, R_32x4_1_1=21,\n"
		"    R_32x4_2_0=13, R_32x4_2_1=27,\n"
		"    R_32x4_3_0=23, R_32x4_3_1= 5,\n"
		"    R_32x4_4_0= 6, R_32x4_4_1=20,\n"
		"    R_32x4_5_0=17, R_32x4_5_1=11,\n"
		"    R_32x4_6_0=25, R_32x4_6_1=10,\n"
		"    R_32x4_7_0=18, R_32x4_7_1=20\n"
		"};\n"

		"static uint RotL_32(uint x, uint N){\n"
		"    return (x << (N & 31)) | (x >> ((32-N) & 31));\n"
		"}\n"

		"uint4 rng4_32(uint4 X, uint4 k){   \n"                                   
		"    uint8 ks;\n"
		"    ks.lo = k;\n"
		"    ks.s4 =  SKEIN_KS_PARITY_32;\n"
		"    ks.s4 ^= k.s0;\n"
		"    ks.s4 ^= k.s1;\n"
		"    ks.s4 ^= k.s2;\n" 
		"    ks.s4 ^= k.s3;\n"
		    
		"    X += ks.lo;\n"
		"    X.s0+= X.s1; X.s1= RotL_32(X.s1,R_32x4_0_0); X.s1^= X.s0;\n"
		"    X.s2+= X.s3; X.s3= RotL_32(X.s3,R_32x4_0_1); X.s3^= X.s2;\n"
		"    X.s0+= X.s3; X.s3= RotL_32(X.s3,R_32x4_1_0); X.s3^= X.s0;\n"
		"    X.s2+= X.s1; X.s1= RotL_32(X.s1,R_32x4_1_1); X.s1^= X.s2;\n"
		"    X.s0+= X.s1; X.s1= RotL_32(X.s1,R_32x4_2_0); X.s1^= X.s0;\n"
		"    X.s2+= X.s3; X.s3= RotL_32(X.s3,R_32x4_2_1); X.s3^= X.s2;\n"
		"    X.s0+= X.s3; X.s3= RotL_32(X.s3,R_32x4_3_0); X.s3^= X.s0;\n"
		"    X.s2+= X.s1; X.s1= RotL_32(X.s1,R_32x4_3_1); X.s1^= X.s2;\n"
		"    X += ks.s1234;\n"
		"    X.s3 += 1;\n"

		"    X.s0+= X.s1; X.s1= RotL_32(X.s1,R_32x4_4_0); X.s1^= X.s0;\n"
		"    X.s2+= X.s3; X.s3= RotL_32(X.s3,R_32x4_4_1); X.s3^= X.s2;\n"
		"    X.s0+= X.s3; X.s3= RotL_32(X.s3,R_32x4_5_0); X.s3^= X.s0;\n"
		"    X.s2+= X.s1; X.s1= RotL_32(X.s1,R_32x4_5_1); X.s1^= X.s2;\n"
		"    X.s0+= X.s1; X.s1= RotL_32(X.s1,R_32x4_6_0); X.s1^= X.s0;\n"
		"    X.s2+= X.s3; X.s3= RotL_32(X.s3,R_32x4_6_1); X.s3^= X.s2;\n"
		"    X.s0+= X.s3; X.s3= RotL_32(X.s3,R_32x4_7_0); X.s3^= X.s0;\n"
		"    X.s2+= X.s1; X.s1= RotL_32(X.s1,R_32x4_7_1); X.s1^= X.s2;\n"
		"    X += ks.s2340;\n"
		"    X.s3 += 2;\n"

		"    X.s0+= X.s1; X.s1= RotL_32(X.s1,R_32x4_0_0); X.s1^= X.s0;\n"
		"    X.s2+= X.s3; X.s3= RotL_32(X.s3,R_32x4_0_1); X.s3^= X.s2;\n"
		"    X.s0+= X.s3; X.s3= RotL_32(X.s3,R_32x4_1_0); X.s3^= X.s0;\n"
		"    X.s2+= X.s1; X.s1= RotL_32(X.s1,R_32x4_1_1); X.s1^= X.s2;\n"
		"    X.s0+= X.s1; X.s1= RotL_32(X.s1,R_32x4_2_0); X.s1^= X.s0;\n"
		"    X.s2+= X.s3; X.s3= RotL_32(X.s3,R_32x4_2_1); X.s3^= X.s2;\n"
		"    X.s0+= X.s3; X.s3= RotL_32(X.s3,R_32x4_3_0); X.s3^= X.s0;\n"
		"    X.s2+= X.s1; X.s1= RotL_32(X.s1,R_32x4_3_1); X.s1^= X.s2;\n"
		"    X += ks.s3401;\n"
		"    X.s3 += 3;\n"
		"    return X;\n"
		"}\n"
		//source for internal routines to generate random numbers
		"#define MAX_RANDOM_32 0xffffffff\n"
		"float4 uniform4_32(float low, float high, uint4 x){\n"
		"    float4 z = convert_float4(x) / MAX_RANDOM_32;\n"
		"    return nextafter(low + z * (high - low), low);\n"
		"}\n"
		"float4 normal4_32(float mean, float stddev, uint4 x){\n"
		"    float4 u = uniform4_32(0,1,x);\n"
		"    float2 r = sqrt(-2 * log(u.lo));\n"
		"    float4 phi;\n"
		"    float2 z;\n"
		"    phi.lo = sincos(2*(float)M_PI * u.hi,&z);\n"
		"    phi.hi = z;"
		"    return mean + stddev * r.s0011 * phi;\n"
		"}\n"
		"uint4 discrete4_32(uint4 key, uint4 ctr, uint N, uint4 inc) {\n"
		"    uint max_valid = (MAX_RANDOM_32 / N) * N;\n"
		"    int num_valid = 0;\n"
		"    uint res[4];"
		"    while(num_valid < 4){\n"
		"        uint4 x = rng4_32(ctr, key);\n"
		"        if(x.s0 < max_valid){\n"
		"            res[num_valid] = x.s0 % N;\n"
		"            ++num_valid;\n"
		"        }\n"
		"        if(num_valid < 4 && x.s1 < max_valid){\n"
		"            res[num_valid] = x.s1 % N;\n"
		"            ++num_valid;\n"
		"        }\n"
		"        if(num_valid < 4 && x.s2 < max_valid){\n"
		"            res[num_valid] = x.s2 % N;\n"
		"            ++num_valid;\n"
		"        }\n"
		"        if(num_valid < 4 && x.s3 < max_valid){\n"
		"            res[num_valid] = x.s3 % N;\n"
		"            ++num_valid;\n"
		"        }\n"
		"        ctr += inc;\n"
		"    }\n"
		"    return (uint4)(res[0],res[1],res[2],res[3]);\n"
		"}\n"
		//kernel routines
		"__kernel void generate_uniform32(__global float *res, uint4 key, float low, float high, ulong offset, ulong cols, ulong leading, ulong stride) {\n"
		"    uint4 ctr;\n"
		"    ctr.s0 = get_global_id(0) & 0x0000ffff;\n"
		"    ctr.s2 = get_global_id(0) >> 32;\n"
		"    ctr.s1 = get_global_id(1) & 0x0000ffff;\n"
		"    ctr.s3 = get_global_id(1) >> 32;\n"
		"    uint4 x = rng4_32(ctr, key);\n"
		"    float4 u = uniform4_32(low, high, x);\n"
		"    size_t col = 4 * get_global_id(1);\n"
		"    size_t pos = offset + stride * (get_global_id(0) * cols + col);\n"
		"    res[pos] = u.s0;\n"
		"    if(col < cols) res[pos + stride] = u.s1;\n"
		"    if(col + 1 < cols) res[pos +2 * stride] = u.s2;\n"
		"    if(col + 2 < cols) res[pos +3 * stride] = u.s3;\n"
		"}\n"
		"__kernel void generate_normal32(__global float *res, uint4 key, float mean, float stddev, ulong offset, ulong cols, ulong leading, ulong stride) {\n"
		"    uint4 ctr;\n"
		"    ctr.s0 = get_global_id(0) & 0x0000ffff;\n"
		"    ctr.s2 = get_global_id(0) >> 32;\n"
		"    ctr.s1 = get_global_id(1) & 0x0000ffff;\n"
		"    ctr.s3 = get_global_id(1) >> 32;\n"
		"    uint4 x = rng4_32(ctr, key);\n"
		"    float4 u = normal4_32(mean, stddev, x);\n"
		"    size_t col = 4 * get_global_id(1);\n"
		"    size_t pos = offset + stride * (get_global_id(0) * cols + col);\n"
		"    res[pos] = u.s0;\n"
		"    if(col < cols) res[pos + stride] = u.s1;\n"
		"    if(col + 1 < cols) res[pos +2 * stride] = u.s2;\n"
		"    if(col + 2 < cols) res[pos +3 * stride] = u.s3;\n"
		"}\n"
		"__kernel void generate_discrete_float_int32(__global float* res,  uint4 key, int low, int high, ulong offset, ulong cols, ulong leading, ulong stride) {\n"
		"    uint N = high - low + 1;\n"
		"    uint4 ctr;\n"
		"    ctr.s0 = get_global_id(0) & 0x0000ffff;\n"
		"    ctr.s2 = get_global_id(0) >> 32;\n"
		"    ctr.s1 = get_global_id(1) & 0x0000ffff;\n"
		"    ctr.s3 = get_global_id(1) >> 32;\n"
		"    uint4 inc = (uint4)(0,get_global_size(1) & 0x0000ffff, 0, get_global_size(1) >> 32);\n"
		"    uint4 z = discrete4_32(key,ctr,N,inc);\n" 
		"    size_t col = 4 * get_global_id(1);\n"
		"    size_t pos = offset + stride * (get_global_id(0) * cols + col);\n"
		"    int valid_required = min((size_t)4, cols - col);\n"
		"    res[pos] = low + convert_float(z.s0)\n;"
		"    if(1 < valid_required) res[pos+stride] = low + convert_float(z.s1)\n;"
		"    if(2 < valid_required) res[pos+2*stride] = low + convert_float(z.s2)\n;"
		"    if(3 < valid_required) res[pos+3*stride] = low + convert_float(z.s2)\n;"
		"}\n";
	auto program = boost::compute::program_cache::get_global_cache(ctx)->get_or_build("remora_random_program32", "", source, ctx);
	return program.create_kernel(kernelname+"32");
}
template<class Rng, class T>
void run_random_kernel(std::string const& kernelname, Rng& rng, boost::compute::command_queue& queue,
	boost::compute::buffer& buffer, uint64_t offset, uint64_t major, uint64_t minor, uint64_t leading_dimension, uint64_t stride, 
	T arg1, T arg2
){
	static_assert(sizeof(T) == 4, "only 32 bit random number types are currently supported");
	if(sizeof(T) == 4){
		//get kernel from program
		auto k = get_random_kernel32(kernelname,queue.get_context());
		//seed key from rng (this is the only possible entropy source for the rng!)
		std::uniform_int_distribution<uint32_t> dist(0,0xffffffff);
		static const std::size_t vectorSize = 4; 
		uint32_t key[vectorSize] = {dist(rng),dist(rng),dist(rng),dist(rng)};
		
		k.set_arg(0,buffer.get());
		k.set_arg(1, sizeof(key), key);
		k.set_arg(2, sizeof(arg1), &arg1);
		k.set_arg(3, sizeof(arg2), &arg2);
		k.set_arg(4, 8, &offset);
		k.set_arg(5, 8, &minor);
		k.set_arg(6, 8, &leading_dimension);
		k.set_arg(7, 8, &stride);
		
		std::size_t global_work_size[2] = {major,(minor + vectorSize-1) / vectorSize};
		queue.enqueue_nd_range_kernel(k, 2,nullptr, global_work_size, nullptr);
	}
}
template<class V, class Rng>
void generate_normal(
	vector_expression<V, gpu_tag>& v,
	Rng& rng,
	typename V::value_type mean,
	typename V::value_type variance
) {
	auto storage = v().raw_storage();
	run_random_kernel("generate_normal", rng, v().queue(), storage.buffer, storage.offset, 1,v().size(), 1, storage.stride, mean, std::sqrt(variance));
}

template<class M, class Rng>
void generate_normal(
	matrix_expression<M, gpu_tag>& m,
	Rng& rng,
	typename M::value_type mean,
	typename M::value_type variance
) {
	auto storage = m().raw_storage();
	std::size_t major = M::orientation::index_M(m().size1(), m().size2());
	std::size_t minor = M::orientation::index_m(m().size1(), m().size2());
	run_random_kernel("generate_normal", rng, m().queue(), storage.buffer, storage.offset, major, minor,storage.leading_dimension, 1, mean, std::sqrt(variance));
}

template<class V, class Rng>
void generate_uniform(
	vector_expression<V, gpu_tag>& v,
	Rng& rng,
	typename V::value_type low,
	typename V::value_type high
) {
	
	auto storage = v().raw_storage();
	run_random_kernel("generate_uniform", rng, v().queue(), storage.buffer, storage.offset, 1, v().size(), 1, storage.stride, low, high);
}

template<class M, class Rng>
void generate_uniform(
	matrix_expression<M, gpu_tag>& m,
	Rng& rng,
	typename M::value_type low,
	typename M::value_type high
) {
	auto storage = m().raw_storage();
	std::size_t major = M::orientation::index_M(m().size1(), m().size2());
	std::size_t minor = M::orientation::index_m(m().size1(), m().size2());
	run_random_kernel("generate_uniform", rng, m().queue(), storage.buffer, storage.offset, major, minor, storage.leading_dimension, 1,low, high);
}

template<class V, class Rng>
void generate_discrete(
	vector_expression<V, gpu_tag>& v,
	Rng& rng,
	int low,
	int high
) {
	auto storage = v().raw_storage();
	run_random_kernel("generate_discrete_float_int", rng, v().queue(), storage.buffer, storage.offset, 1, v().size(), 1, storage.stride, low, high);
}

template<class M, class Rng>
void generate_discrete(
	matrix_expression<M, gpu_tag>& m,
	Rng& rng,
	int low,
	int high
) {
	auto storage = m().raw_storage();
	std::size_t major = M::orientation::index_M(m().size1(), m().size2());
	std::size_t minor = M::orientation::index_m(m().size1(), m().size2());
	run_random_kernel("generate_discrete_float_int", rng, m().queue(), storage.buffer, storage.offset, major, minor,storage.leading_dimension, 1, low, high);
}

}}
#endif