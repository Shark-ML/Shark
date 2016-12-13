#define BOOST_TEST_MODULE BLAS_Syrk
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/mpl/list.hpp>

#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/kernels/syrk.hpp>

using namespace shark;
using namespace blas;


template<class M1, class Result>
void checkSyrk(M1 const& arg, Result const& result,double init, double alpha, bool upper){
	BOOST_REQUIRE_EQUAL(arg.size1(), result.size1());
	BOOST_REQUIRE_EQUAL(result.size1(), result.size2());
	
	if(upper){
		for(std::size_t i = 0; i != result.size1(); ++i) {
			for(std::size_t j = 0; j != result.size2(); ++j) {
				if(j < i){
					BOOST_CHECK_CLOSE(result(i,j),init, 1.e-10);
				}else{
					double test_result = alpha*inner_prod(row(arg,i),row(arg,j))+init;
					BOOST_CHECK_CLOSE(result(i,j), test_result, 1.e-10);
				}
			}
		}
	}else{
		for(std::size_t i = 0; i != result.size1(); ++i) {
			for(std::size_t j = 0; j != result.size2(); ++j) {
				if(j > i){
					BOOST_CHECK_CLOSE(result(i,j),init, 1.e-10);
				}else{
					double test_result = alpha*inner_prod(row(arg,i),row(arg,j))+init;
					BOOST_CHECK_CLOSE(result(i,j), test_result, 1.e-10);
				}
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE(BLAS_SYRK)



typedef boost::mpl::list<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(syrk_test, Orientation,result_orientations) {
	std::size_t dims = 936;//chosen as not to be a multiple of the block size
	std::size_t K = 1024;

	//rhs
	matrix<double, row_major> argrm(dims, K, 1.0);
	matrix<double, column_major> argcm(dims, K, 1.0);
	for(std::size_t i = 0; i != dims; ++i) {
		for(std::size_t j = 0; j != K; ++j) {
			argrm(i, j) = argcm(i, j) = (1.0/ dims) * i + 0.2/K * j + 1;
		}
	}

	std::cout << "\nchecking syrk V+=AA^T" << std::endl;
	{
		std::cout<<"row major A, lower V"<<std::endl;
		matrix<double,Orientation> result(dims, dims, 3.0);
		kernels::syrk<false>(argrm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,false);
	}
	{
		std::cout<<"row major A, upper V"<<std::endl;
		matrix<double,Orientation> result(dims, dims, 3.0);
		kernels::syrk<true>(argrm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,true);
	}
	{
		std::cout<<"column major A, lower V"<<std::endl;
		matrix<double,Orientation> result(dims, dims, 3.0);
		kernels::syrk<false>(argcm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,false);
	}
	{
		std::cout<<"column major A, upper V"<<std::endl;
		matrix<double,Orientation> result(dims, dims, 3.0);
		kernels::syrk<true>(argcm,result, 2.0);
		checkSyrk(argrm,result, 3.0, 2.0,true);
	}
	
}

BOOST_AUTO_TEST_SUITE_END()
