#ifndef SHARK_CORE_IMAGE_CPU_REORDER_H
#define SHARK_CORE_IMAGE_CPU_REORDER_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
namespace shark{

namespace image{
template<class T>
void reorder_impl(
	blas::dense_vector_adaptor<T const, blas::continuous_dense_tag, blas::cpu_tag> inputs, 
	blas::dense_vector_adaptor<T, blas::continuous_dense_tag, blas::cpu_tag> outputs, 
	std::size_t sizes[4],
	std::size_t strides[4]
){
	std::size_t elem = 0;
	for(std::size_t i0 = 0; i0 != sizes[0]; ++i0){
		for(std::size_t i1 = 0; i1 != sizes[1]; ++i1){
			for(std::size_t i2 = 0; i2 != sizes[2]; ++i2){
				for(std::size_t i3 = 0; i3 != sizes[3]; ++i3, ++elem){
					std::size_t index = strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3;
					outputs(elem) = inputs(index);
				}
			}
		}
	}
}

}}

#endif