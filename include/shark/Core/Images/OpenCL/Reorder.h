#ifndef SHARK_CORE_IMAGE_GPU_REORDER_H
#define SHARK_CORE_IMAGE_GPU_REORDER_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Exception.h>
#include <shark/Core/Shape.h>
namespace shark{

namespace image{
template<class T>
void reorder_impl(
	blas::dense_vector_adaptor<T const, blas::continuous_dense_tag, blas::gpu_tag> inputs_unreg, 
	blas::dense_vector_adaptor<T, blas::continuous_dense_tag, blas::gpu_tag> outputs_unreg, 
	std::size_t size[4],
	std::size_t stride[4]
){
	SIZE_CHECK(inputs_unreg.size() == outputs_unreg.size());
	SIZE_CHECK(inputs_unreg.size() == size[0]*size[1]*size[2]*size[3]);

	blas::gpu::detail::meta_kernel k("shark_reorder");
	std::size_t size_index = k.add_arg<boost::compute::uint4_>("size");
	std::size_t stride_index = k.add_arg<boost::compute::uint4_>("stride");
	auto inputs = k.register_args(to_functor(inputs_unreg));
	auto outputs = k.register_args(to_functor(outputs_unreg));

	k << "const ulong id = get_global_id(0);\n";
	k << "const ulong i0 = id / size.s1;\n";
	k << "const ulong i1 = id % size.s1;\n";
	//obtain base index for input and output
	k << "const ulong startInput = i0 * stride.s0 + i1 * stride.s1;\n";
	k << "ulong startOutput = (i0 * size.s1 + i1) * size.s2 * size.s3;\n";
		
	k << "for(ulong i2 = get_local_id(1); i2 < size.s2; i2 += get_local_size(1)){;\n";
	k << "	for(ulong i3 = get_local_id(2); i3 < size.s3; i3 += get_local_size(2)){;\n";
	k << "		ulong indexIn = startInput + stride.s2 * i2 + stride.s3 * i3;\n";
	k << "		ulong indexOut = startOutput + size.s3 * i2 + i3;\n";
	k << "		" << outputs(k.expr<cl_ulong>("indexOut"))<<" = "<<inputs(k.expr<cl_ulong>("indexIn"))<<";\n";
	k << "	}\n";
	k << "}\n";
	
	//compile kernel
	boost::compute::kernel kernel = k.compile(outputs_unreg.queue().get_context());
	
	//enqueue kernel with kernel args
	kernel.set_arg(size_index, boost::compute::uint4_({unsigned(size[0]),unsigned(size[1]),unsigned(size[2]),unsigned(size[3])}));
	kernel.set_arg(stride_index, boost::compute::uint4_({unsigned(stride[0]),unsigned(stride[1]),unsigned(stride[2]),unsigned(stride[3])}));
	
	
	std::size_t local_work_size[3] = {1, 8, 4};
	std::size_t global_work_size[3] = {size[0] * size[1], local_work_size[1], local_work_size[2] };
	outputs_unreg.queue().enqueue_nd_range_kernel(kernel, 3, nullptr, global_work_size, local_work_size);
}

}}

#endif