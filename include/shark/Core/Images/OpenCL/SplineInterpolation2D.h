#ifndef SHARK_CORE_IMAGES_GPU_SPLINE_INTERPOLATION_2D_H
#define SHARK_CORE_IMAGES_GPU_SPLINE_INTERPOLATION_2D_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
namespace shark{
namespace image{
template<class T>
void splineInterpolation2D(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> images_unreg, 
	Shape const& shape,
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> points_unreg, std::size_t pointsPerImage, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> values_unreg
){
	values_unreg.clear();
	
	//generate kernel source
	blas::gpu::detail::meta_kernel k("shark_spline_interpolation_2D");
	std::size_t width_index = k.add_arg<std::size_t>("width");
	std::size_t height_index = k.add_arg<std::size_t>("height");
	std::size_t depth_index = k.add_arg<std::size_t>("depth");
	std::size_t stride_index = k.add_arg<std::size_t>("stride");
	auto images = k.register_args(to_functor(images_unreg));
	auto points = k.register_args(to_functor(points_unreg));
	auto values = k.register_args(to_functor(values_unreg));
	
	k << "const ulong im = get_global_id(0);\n";
	k << "const ulong p = get_global_id(1);\n";
	k << "const ulong pointIdx = p + im * stride;\n";
	
	//integer pixel positions (rounded down)
	auto pointx = points(k.expr<cl_ulong>("pointIdx"),1);
	auto pointy = points(k.expr<cl_ulong>("pointIdx"),0);
	k << "int basex = (int) ("<<pointx<<" * width);\n";
	k << "int basey = (int) ("<<pointy<<" * height);\n";
	//the sets of indices accessed. this also incldues clamping border condition
	k << "int px[4] = {clamp(basex - 1, 0, (int)width - 1), clamp(basex, 0, (int)width - 1), clamp(basex + 1, 0, (int)width - 1), clamp(basex + 2, 0, (int)width - 1)};\n";
	k << "int py[4] = {clamp(basey - 1, 0, (int)height - 1), clamp(basey, 0, (int)height - 1), clamp(basey + 1, 0, (int)height - 1), clamp(basey + 2, 0, (int)height - 1)};\n";
	
	//compute the interpolation constants for the x-coordinate
	k << k.decl<T>("t") << "=" << pointx << " * width - basex;\n";
	k << k.decl<T>("t2") << " = t * t;\n";
	k << k.decl<T>("t3") << " = t2 * t;\n";
	k <<  k.decl<T>("x")<<"[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};\n";
	
	//compute the interpolation constants for the y-coordinate
	k << "t = " << pointy << " * height - basey;\n";
	k << "t2 = t * t;\n";
	k << "t3 = t2 * t;\n";
	k << k.decl<T>("y")<<"[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};\n";
	
	//perform actual interpolation
	k << "for(int k = 0;k < 4; ++k){\n";
	k << "	for(int l = 0;l < 4; ++l){\n";
	k << "		const ulong imIdx = (px[l] + width * py[k]) * depth;\n";
	k << "		const ulong valIdx = p * depth;\n";
	k << "		for(int c = 0;c < depth; ++c){\n";
	k << "			"<<values(k.expr<cl_ulong>("im"), k.expr<cl_ulong>("valIdx + c") )
					<<" += (x[l]*y[k]/36) * "<<images(k.expr<cl_ulong>("im"), k.expr<cl_ulong>("imIdx + c" ))<<";\n";
	k << "		}\n";
	k << "	}\n";
	k << "}\n";

	//compile kernel
	boost::compute::kernel kernel = k.compile(values_unreg.queue().get_context());
	
	//enqueue kernel with kernel args
	std::size_t stride = (pointsPerImage == points_unreg.size1())? 0: pointsPerImage;
	kernel.set_arg(height_index, shape[0]);
	kernel.set_arg(width_index, shape[1]);
	kernel.set_arg(depth_index, shape[2]);
	kernel.set_arg(stride_index, stride);
	
	std::size_t global_work_size[2] = {images_unreg.size1(), pointsPerImage};
	values_unreg.queue().enqueue_nd_range_kernel(kernel, 2, nullptr, global_work_size, nullptr);
}

template<class T>
void splineInterpolation2DDerivative(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> imageDerivatives_unreg, 
	Shape const& shape,
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> points_unreg, std::size_t pointsPerImage, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::gpu_tag> results_unreg
){
	
	results_unreg.clear();
	// implementationd etail: because we can not assume the points to lie ona  grid, we can not parallelize computation of the derivatives
	// in the number of points. reason is that several points share the same pixels in the image and we do not know which and how many.
	// instead we will parallelize over the image pixel lookup, so the parallelisation is numImages x (4*4)
	
	//generate kernel source
	blas::gpu::detail::meta_kernel k("shark_spline_interpolation_2D_derivative");
	std::size_t width_index = k.add_arg<std::size_t>("width");
	std::size_t height_index = k.add_arg<std::size_t>("height");
	std::size_t depth_index = k.add_arg<std::size_t>("depth");
	std::size_t stride_index = k.add_arg<std::size_t>("stride");
	std::size_t numPoints_index = k.add_arg<std::size_t>("numPoints");
	auto imageDerivatives = k.register_args(to_functor(imageDerivatives_unreg));
	auto points = k.register_args(to_functor(points_unreg));
	auto results = k.register_args(to_functor(results_unreg));
	
	k << "const ulong im = get_group_id(0);\n";
	k << "const ulong c = get_local_id(1);\n";
	//integer pixel positions (rounded down)
	k << "for(int p = 0;p < numPoints; ++p){\n";
	k << "	const ulong pointIdx = p + im * stride;\n";
	auto pointx = points(k.expr<cl_ulong>("pointIdx"),1);
	auto pointy = points(k.expr<cl_ulong>("pointIdx"),0);
	k << "	int basex = (int) ("<<pointx<<" * width);\n";
	k << "	int basey = (int) ("<<pointy<<" * height);\n";
	//the sets of indices accessed. this also incldues clamping border condition
	k << "	int px[4] = {clamp(basex - 1, 0, (int)width - 1), clamp(basex, 0, (int)width - 1), clamp(basex + 1, 0, (int)width - 1), clamp(basex + 2, 0, (int)width - 1)};\n";
	k << "	int py[4] = {clamp(basey - 1, 0, (int)height - 1), clamp(basey, 0, (int)height - 1), clamp(basey + 1, 0, (int)height - 1), clamp(basey + 2, 0, (int)height - 1)};\n";
	
	//compute the interpolation constants for the x-coordinate
	k << k.decl<T>("t") << "=" << pointx << " * width - basex;\n";
	k << k.decl<T>("t2") << " = t * t;\n";
	k << k.decl<T>("t3") << " = t2 * t;\n";
	k <<  k.decl<T>("x")<<"[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};\n";
	
	//compute the interpolation constants for the y-coordinate
	k << "	t = " << pointy << " * height - basey;\n";
	k << "	t2 = t * t;\n";
	k << "	t3 = t2 * t;\n";
	k << k.decl<T>("y")<<"[4]={-t3+3*t2-3*t+1, 3*t3-6*t2+4, -3*t3+3*t2+3*t+1,  t3};\n";
	
	//compute derivative
	k << "	for(int k = 0;k < 4; ++k){\n";
	k << "		for(int l = 0;l < 4; ++l){\n";
	k << "			const ulong imIdx = (px[l] + width * py[k]) * depth;\n";
	k << "			const ulong valIdx = p * depth;\n";
	k << "			"<<results(k.expr<cl_ulong>("im"), k.expr<cl_ulong>("imIdx + c" ))
					<<" += (x[l]*y[k]/36) * "<<imageDerivatives(k.expr<cl_ulong>("im"), k.expr<cl_ulong>("valIdx + c" ))<<";\n";
	k << "		}\n";
	k << "	}\n";
	k << "}\n";
	//compile kernel
	boost::compute::kernel kernel = k.compile(results_unreg.queue().get_context());
	
	//enqueue kernel with kernel args
	std::size_t stride = (pointsPerImage == points_unreg.size1())? 0: pointsPerImage;
	kernel.set_arg(height_index, shape[0]);
	kernel.set_arg(width_index, shape[1]);
	kernel.set_arg(depth_index, shape[2]);
	kernel.set_arg(stride_index, stride);
	kernel.set_arg(numPoints_index, pointsPerImage);
	
	std::size_t global_work_size[2] = {imageDerivatives_unreg.size1() * 1, shape[2]};
	std::size_t local_work_size[2] = {1,shape[2]};
	results_unreg.queue().enqueue_nd_range_kernel(kernel, 2, nullptr, global_work_size, local_work_size);
}

}}

#endif