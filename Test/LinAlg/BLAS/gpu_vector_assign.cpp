#define BOOST_TEST_MODULE LinAlg_BLAS_GPU_VectorAssign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/gpu/vector.hpp>
#include <shark/LinAlg/BLAS/gpu/copy.hpp>

using namespace shark;

template<class V1, class V2>
void checkVectorEqual(V1 const& v1_gpu, V2 const& v2_gpu){
	BOOST_REQUIRE_EQUAL(v1_gpu.size(),v2_gpu.size());
	
	blas::vector<unsigned int> v1 = copy_to_cpu(v1_gpu);
	blas::vector<unsigned int> v2 = copy_to_cpu(v2_gpu);
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_EQUAL(v1(i),v2(i));
	}
}

BOOST_AUTO_TEST_SUITE (LinAlg_BLAS_gpu_vector_assign)

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Vector_Assign_Dense ){
	std::cout<<"testing dense-dense assignment"<<std::endl;
	blas::vector<unsigned int> source_cpu(1000);
	blas::vector<unsigned int> target_cpu(1000);
	blas::vector<unsigned int> result_add_cpu(1000);
	blas::vector<unsigned int> result_add_scalar_cpu(1000);
	unsigned int scalar = 10;
	for(std::size_t i = 0; i != 1000; ++i){
		source_cpu(i) = 2*i+1;
		target_cpu(i) = 3*i+2;
		result_add_cpu(i) = source_cpu(i) + target_cpu(i);
		result_add_scalar_cpu(i) = source_cpu(i) + scalar;
	}
	blas::gpu::vector<unsigned int> source = blas::gpu::copy_to_gpu(source_cpu);
	blas::gpu::vector<unsigned int> result_add = blas::gpu::copy_to_gpu(result_add_cpu);
	blas::gpu::vector<unsigned int> result_add_scalar = blas::gpu::copy_to_gpu(result_add_scalar_cpu);
	{
		std::cout<<"testing direct assignment"<<std::endl;
		blas::gpu::vector<unsigned int> target = blas::gpu::copy_to_gpu(target_cpu);
		blas::kernels::assign(target,source);
		checkVectorEqual(target,source);
	}
	{
		std::cout<<"testing functor assignment"<<std::endl;
		blas::gpu::vector<unsigned int> target = blas::gpu::copy_to_gpu(target_cpu);
		std::cout<<"testing dense-dense"<<std::endl;
		blas::kernels::assign<blas::device_traits<blas::gpu_tag>::add<unsigned int> >(target,source);
		checkVectorEqual(target,result_add);
	}
	{
		std::cout<<"testing functor scalar assignment"<<std::endl;
		blas::gpu::vector<unsigned int> target = blas::gpu::copy_to_gpu(target_cpu);
		std::cout<<"testing dense-dense"<<std::endl;
		blas::kernels::assign<blas::device_traits<blas::gpu_tag>::add<unsigned int> >(target,scalar);
		checkVectorEqual(target,result_add_scalar);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
