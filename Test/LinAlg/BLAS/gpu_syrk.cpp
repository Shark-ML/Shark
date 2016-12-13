#define BOOST_TEST_MODULE BLAS_GPU_Syrk
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/mpl/list.hpp>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION

#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/io.hpp>
#include <shark/LinAlg/BLAS/gpu/matrix.hpp>
#include <shark/LinAlg/BLAS/gpu/copy.hpp>
#include <shark/LinAlg/BLAS/kernels/syrk.hpp>

using namespace shark;
using namespace blas;

template<class M, class Result>
void checkSyrk(M const& arg_gpu, Result const& result_gpu,double init, double alpha, bool upper){
	BOOST_REQUIRE_EQUAL(arg_gpu.size1(), result_gpu.size1());
	BOOST_REQUIRE_EQUAL(result_gpu.size1(), result_gpu.size2());
	
	matrix<float> arg = copy_to_cpu(arg_gpu);
	matrix<float> result = copy_to_cpu(result_gpu);
	
	if(upper){
		for(std::size_t i = 0; i != result.size1(); ++i) {
			for(std::size_t j = 0; j != result.size2(); ++j) {
				if(j < i){
					BOOST_CHECK_CLOSE(result(i,j),init, 1.e-5);
				}else{
					double test_result = alpha*inner_prod(row(arg,i),row(arg,j))+init;
					BOOST_CHECK_CLOSE(result(i,j), test_result, 1.e-5);
				}
			}
		}
	}else{
		for(std::size_t i = 0; i != result.size1(); ++i) {
			for(std::size_t j = 0; j != result.size2(); ++j) {
				if(j > i){
					BOOST_CHECK_CLOSE(result(i,j),init, 1.e-5);
				}else{
					double test_result = alpha*inner_prod(row(arg,i),row(arg,j))+init;
					BOOST_CHECK_CLOSE(result(i,j), test_result, 1.e-5);
				}
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE (BLAS_Gpu_Syrk)

typedef boost::mpl::list<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(syrk_test, Orientation,result_orientations) {
	std::size_t dims = 936;//chosen as not to be a multiple of the block size
	std::size_t K = 1039;

	//rhs
	matrix<float, row_major> arg_cpu(dims, K, 1.0);
	for(std::size_t i = 0; i != dims; ++i) {
		for(std::size_t j = 0; j != K; ++j) {
			arg_cpu(i, j) = (1.0/ dims) * i + 0.2/K * j + 1;
		}
	}
	
	gpu::matrix<float,row_major> argrm = gpu::copy_to_gpu(arg_cpu);
	gpu::matrix<float,column_major> argcm = gpu::copy_to_gpu(arg_cpu);

	std::cout << "\nchecking syrk V+=AA^T" << std::endl;
	{
		std::cout<<"row major A, lower V"<<std::endl;
		gpu::matrix<float,Orientation> result(dims,dims,3.0);
		kernels::syrk<false>(argrm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,false);
	}
	{
		std::cout<<"row major A, upper V"<<std::endl;
		gpu::matrix<float,Orientation> result(dims,dims,3.0);
		kernels::syrk<true>(argrm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,true);
	}
	{
		std::cout<<"column major A, lower V"<<std::endl;
		gpu::matrix<float,Orientation> result(dims,dims,3.0);
		kernels::syrk<false>(argcm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,false);
	}
	{
		std::cout<<"column major A, upper V"<<std::endl;
		gpu::matrix<float,Orientation> result(dims,dims,3.0);
		kernels::syrk<true>(argcm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,true);
	}
}

BOOST_AUTO_TEST_SUITE_END()
