#ifndef SHARK_CORE_IMAGES_OPENCL_POOLING_H
#define SHARK_CORE_IMAGES_OPENCL_POOLING_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Exception.h>
#include <shark/Core/Shape.h>
namespace shark{
namespace image{
template<class T>
void maxPooling(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> inputs_unreg, 
	Shape const& shape,
	Shape const& patchSize,
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> outputs_unreg
){
	
	std::size_t depth = shape[2];
	std::size_t outputHeight = shape[0]/patchSize[0];
	std::size_t outputWidth = shape[1]/patchSize[1];
	std::size_t outputPixels = outputWidth * outputHeight;
	SIZE_CHECK(inputs_unreg.size2() == shape[0] * shape[1] * depth);
	SIZE_CHECK(outputs_unreg.size2() == outputPixels * depth);
	SIZE_CHECK(outputs_unreg.size1() == inputs_unreg.size1());

	blas::gpu::detail::meta_kernel k("shark_max_pooling");
	std::size_t width_index = k.add_arg<std::size_t>("width");
	std::size_t height_index = k.add_arg<std::size_t>("height");
	std::size_t depth_index = k.add_arg<std::size_t>("depth");
	std::size_t sizeH_index = k.add_arg<std::size_t>("sizeH");
	std::size_t sizeW_index = k.add_arg<std::size_t>("sizeW");
	std::size_t numImages_index = k.add_arg<std::size_t>("numImages");
	auto inputs = k.register_args(to_functor(inputs_unreg));
	auto outputs = k.register_args(to_functor(outputs_unreg));

	k << "const ulong outputWidth = width / sizeW;\n";
	k << "const ulong outputHeight = height / sizeH;\n";
	k << "const ulong numOutputs = outputWidth * outputHeight;\n";
	k << "const ulong id = get_global_id(0);\n";
	k << "if(id >= numImages * numOutputs) return;\n"; //bounds checking for groups
	
	k << "const ulong im = id / numOutputs;\n";//extract image id
	k << "const ulong p = id % numOutputs;\n";//extract patch id
	
	//get start and end-coordinates of the patch
	k << "const ulong starti = (p / outputWidth) * sizeH;\n";
	k << "const ulong startj = (p % outputWidth) * sizeW;\n";
	k << "const ulong endi = starti + sizeH;\n";
	k << "const ulong endj = startj + sizeW;\n";
	k << "for(ulong c = get_local_id(1); c < depth; c += get_local_size(1)){\n";
	k << "	ulong index = (starti * width + startj) * depth +c;\n";
	auto im = k.expr<cl_ulong>("im");
	auto index = k.expr<cl_ulong>("index");
		//traverse the patch on the input image and compute maximum
	k << "	" << k.decl<T>("val") <<" = "<< inputs(im, index) << ";\n";
	k << "	for(ulong i = starti; i != endi; ++i){\n";
	k << "		for(ulong j = startj; j != endj; ++j){\n";
	k << "			index =  (i * width + j) * depth + c;\n";
	k << "			val = max(val,"<<inputs(im, index)<<");\n";
	k << "		}\n";
	k << "	}\n";
	k << "	" << outputs(im, k.expr<cl_ulong>("(p * depth + c)"))<<" = val;\n";
	k << "}\n";
	
	//compile kernel
	boost::compute::kernel kernel = k.compile(outputs_unreg.queue().get_context());
	
	//enqueue kernel with kernel args
	kernel.set_arg(height_index, shape[0]);
	kernel.set_arg(width_index, shape[1]);
	kernel.set_arg(depth_index, shape[2]);
	kernel.set_arg(sizeH_index, patchSize[0]);
	kernel.set_arg(sizeW_index, patchSize[1]);
	kernel.set_arg(numImages_index, inputs_unreg.size1());
	
	
	std::size_t local_work_size[2] = {8, 4};
	//round global work size up to next multiple of local work size
	std::size_t global_work_size[2] = {
		((inputs_unreg.size1() * outputPixels + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0] , 
		((depth + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1] 
	};
	outputs_unreg.queue().enqueue_nd_range_kernel(kernel, 2, nullptr, global_work_size, local_work_size);
}

template<class T>
void maxPoolingDerivative(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> inputs_unreg, 
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> coefficients_unreg, 
	Shape const& shape,
	Shape const& patchSize,
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> derivatives_unreg
){
	derivatives_unreg.clear();
	std::size_t depth = shape[2];
	std::size_t outputHeight = shape[0]/patchSize[0];
	std::size_t outputWidth = shape[1]/patchSize[1];
	std::size_t outputPixels = outputWidth * outputHeight;
	SIZE_CHECK(derivatives_unreg.size2() == shape[0] * shape[1] * depth);
	SIZE_CHECK(inputs_unreg.size2() == shape[0] * shape[1] * depth);
	SIZE_CHECK(coefficients_unreg.size2() == outputPixels * depth);
	SIZE_CHECK(derivatives_unreg.size1() == inputs_unreg.size1());
	SIZE_CHECK(derivatives_unreg.size1() == coefficients_unreg.size1());

	blas::gpu::detail::meta_kernel k("shark_max_pooling_derivative");
	std::size_t width_index = k.add_arg<std::size_t>("width");
	std::size_t height_index = k.add_arg<std::size_t>("height");
	std::size_t depth_index = k.add_arg<std::size_t>("depth");
	std::size_t sizeH_index = k.add_arg<std::size_t>("sizeH");
	std::size_t sizeW_index = k.add_arg<std::size_t>("sizeW");
	std::size_t numImages_index = k.add_arg<std::size_t>("numImages");
	auto inputs = k.register_args(to_functor(inputs_unreg));
	auto coefficients = k.register_args(to_functor(coefficients_unreg));
	auto derivatives = k.register_args(to_functor(derivatives_unreg));

	k << "const ulong outputWidth = width / sizeW;\n";
	k << "const ulong outputHeight = height / sizeH;\n";
	k << "const ulong numOutputs = outputWidth * outputHeight;\n";
	k << "const ulong id = get_global_id(0);\n";
	k << "if(id >= numImages * numOutputs) return;\n"; //bounds checking for groups
	
	k << "const ulong im = id / numOutputs;\n";//extract image id
	k << "const ulong p = id % numOutputs;\n";//extract patch id
	
	//get start and end-coordinates of the patch
	k << "const ulong starti = (p / outputWidth) * sizeH;\n";
	k << "const ulong startj = (p % outputWidth) * sizeW;\n";
	k << "const ulong endi = starti + sizeH;\n";
	k << "const ulong endj = startj + sizeW;\n";
	k << "for(ulong c = get_local_id(1); c < depth; c += get_local_size(1)){\n";
	k << "	ulong index = (starti * width + startj) * depth +c;\n";
	auto im = k.expr<cl_ulong>("im");
	auto index = k.expr<cl_ulong>("index");
		//traverse the patch on the input image and compute maximum
	k << "	" << k.decl<T>("maxVal")<<" = " <<inputs(im, index) << ";\n";
	k << "	ulong maxIndex = index;\n";
	k << "	for(ulong i = starti; i != endi; ++i){\n";
	k << "		for(ulong j = startj; j != endj; ++j){\n";
	k << "			index =  (i * width + j) * depth + c;\n";
	k << "			if("<<inputs(im, index)<<" > maxVal){\n";
	k << "				maxVal = "<<inputs(im, index)<<";\n";
	k << "				maxIndex = index;\n";
	k << "			}\n";
	k << "		}\n";
	k << "	}\n";
	k << "	" << derivatives(im, k.expr<cl_ulong>("maxIndex")) << " = " << coefficients(im,k.expr<cl_ulong>("(p * depth +c)")) << ";\n";
	k << "}\n";
	
	//compile kernel
	boost::compute::kernel kernel = k.compile(derivatives_unreg.queue().get_context());
	
	//enqueue kernel with kernel args
	kernel.set_arg(height_index, shape[0]);
	kernel.set_arg(width_index, shape[1]);
	kernel.set_arg(depth_index, shape[2]);
	kernel.set_arg(sizeH_index, patchSize[0]);
	kernel.set_arg(sizeW_index, patchSize[1]);
	kernel.set_arg(numImages_index, inputs_unreg.size1());
	
	
	std::size_t local_work_size[2] = {8, 4};
	//round global work size up to next multiple of local work size
	std::size_t global_work_size[2] = {
		((inputs_unreg.size1() * outputPixels + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0] , 
		((depth + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1]
	};
	derivatives_unreg.queue().enqueue_nd_range_kernel(kernel, 2, nullptr, global_work_size, local_work_size);
}


}}

#endif