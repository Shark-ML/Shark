#define BOOST_TEST_MODULE BLAS_gpu_prod
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/gpu/vector.hpp>
#include <shark/LinAlg/BLAS/gpu/matrix.hpp>
#include <shark/LinAlg/BLAS/gpu/copy.hpp>

using namespace shark;
using namespace blas;

template<class M, class V, class Result>
void checkMatrixVectorMultiply(M const& arg1_gpu, V const& arg2_gpu, Result const& result_gpu, float factor, float init = 0){
	BOOST_REQUIRE_EQUAL(arg1_gpu.size1(), result_gpu.size());
	BOOST_REQUIRE_EQUAL(arg2_gpu.size(), arg1_gpu.size2());
	
	matrix<float> arg1 = copy_to_cpu(arg1_gpu);
	vector<float> arg2 = copy_to_cpu(arg2_gpu);
	vector<float> result = copy_to_cpu(result_gpu); 
	
	for(std::size_t i = 0; i != arg1.size1(); ++i){
		float test_result = init;
		for(std::size_t k = 0; k != arg1.size2(); ++k){
			test_result += factor * arg1(i,k)*arg2(k);
		}
		BOOST_CHECK_CLOSE(result(i), test_result,1.e-10);
	}
}

BOOST_AUTO_TEST_SUITE (BLAS_gpu_prod)

BOOST_AUTO_TEST_CASE( BLAS_gpu_prod_vector_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	matrix<float,row_major> arg1rm_cpu(rows,columns);
	matrix<float,column_major> arg1cm_cpu(rows,columns);
	matrix<float,row_major> arg1rmt_cpu(columns,rows);
	matrix<float,column_major> arg1cmt_cpu(columns,rows);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg1rm_cpu(i,j) = arg1cm_cpu(i,j) = i*columns+0.2*j;
			arg1rmt_cpu(j,i) = arg1cmt_cpu(j,i) = i*columns+0.2*j;
		}
	}
	vector<float> arg2_cpu(columns);
	for(std::size_t j = 0; j != columns; ++j){
		arg2_cpu(j)  = 1.5*j+2;
	}
	
	gpu::matrix<float,row_major> arg1rm = gpu::copy_to_gpu(arg1rm_cpu);
	gpu::matrix<float,column_major> arg1cm = gpu::copy_to_gpu(arg1cm_cpu);
	gpu::matrix<float,row_major> arg1rmt = gpu::copy_to_gpu(arg1rmt_cpu);
	gpu::matrix<float,column_major> arg1cmt = gpu::copy_to_gpu(arg1cmt_cpu);
	gpu::vector<float> arg2 = gpu::copy_to_gpu(arg2_cpu);
	std::cout<<"\nchecking dense matrix-vector plusassign multiply"<<std::endl;
	//test first expressions of the form A += alpha*B*C 
	{
		std::cout<<"row major Ax"<<std::endl;
		gpu::vector<float> result(rows,1.5);
		noalias(result) += -2*prod(arg1rm,arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		gpu::vector<float> result(rows,1.5);
		noalias(result) += -2*prod(arg1cm,arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		gpu::vector<float> result(rows,1.5);
		noalias(result) += -2*prod(arg2,arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		gpu::vector<float> result(rows,1.5);
		noalias(result) += -2*prod(arg2,arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking dense matrix-vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		gpu::vector<float> result(rows,1.5);
		noalias(result) = -2*prod(arg1rm,arg2);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		gpu::vector<float> result(rows,1.5);
		noalias(result) = -2*prod(arg1cm,arg2);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		gpu::vector<float> result(rows,1.5);
		noalias(result) = -2*prod(arg2,arg1rmt);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		gpu::vector<float> result(rows,1.5);
		noalias(result) = -2*prod(arg2,arg1cmt);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}

//we test using the textbook definition.
template<class Arg1, class Arg2, class Result>
void checkMatrixMatrixMultiply(Arg1 const& arg1_gpu, Arg2 const& arg2_gpu, Result const& result_gpu, float factor, float init = 0){
	BOOST_REQUIRE_EQUAL(arg1_gpu.size1(), result_gpu.size1());
	BOOST_REQUIRE_EQUAL(arg2_gpu.size2(), result_gpu.size2());
	
	matrix<float> arg1 = copy_to_cpu(arg1_gpu);
	matrix<float> arg2 = copy_to_cpu(arg2_gpu);
	matrix<float> result = copy_to_cpu(result_gpu); 
	
	for(std::size_t i = 0; i != arg1.size1(); ++i){
		for(std::size_t j = 0; j != arg2.size2(); ++j){
			float test_result = init;
			for(std::size_t k = 0; k != arg1.size2(); ++k){
				 test_result += factor * arg1(i,k)*arg2(k,j);
			}
			BOOST_CHECK_CLOSE(result(i,j), test_result,1.e-10);
		}
	}
}
typedef boost::mpl::list<row_major,column_major> result_orientations;
BOOST_AUTO_TEST_CASE_TEMPLATE(BLAS_prod_gpu_matrix_dense_dense, Orientation,result_orientations) {
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	matrix<float,row_major> arg1rm_cpu(rows,middle);
	matrix<float,column_major> arg1cm_cpu(rows,middle);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != middle; ++j){
			arg1rm_cpu(i,j) = arg1cm_cpu(i,j) = i*middle+0.2*j;
		}
	}
	matrix<float,row_major> arg2rm_cpu(middle,columns);
	matrix<float,column_major> arg2cm_cpu(middle,columns);
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg2rm_cpu(i,j) = arg2cm_cpu(i,j) = i*columns+1.5*j;
		}
	}
	
	gpu::matrix<float,row_major> arg1rm = gpu::copy_to_gpu(arg1rm_cpu);
	gpu::matrix<float,column_major> arg1cm = gpu::copy_to_gpu(arg1cm_cpu);
	gpu::matrix<float,row_major> arg2rm = gpu::copy_to_gpu(arg2rm_cpu);
	gpu::matrix<float,column_major> arg2cm = gpu::copy_to_gpu(arg2cm_cpu);
	
	std::cout<<"\nchecking dense-dense matrix-matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		gpu::matrix<float,row_major> resultrm(rows,columns,1.5);
		noalias(resultrm) += -2.0 * prod(arg1rm,arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		gpu::matrix<float,column_major> resultcm(rows,columns,1.5);
		noalias(resultcm) += -2.0 * prod(arg1rm,arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		gpu::matrix<float,row_major> resultrm(rows,columns,1.5);
		noalias(resultrm) += -2.0 * prod(arg1rm,arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		gpu::matrix<float,column_major> resultcm(rows,columns,1.5);
		noalias(resultcm) += -2.0 * prod(arg1rm,arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		gpu::matrix<float,row_major> resultrm(rows,columns,1.5);
		noalias(resultrm) += -2.0 * prod(arg1cm,arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		gpu::matrix<float,column_major> resultcm(rows,columns,1.5);
		noalias(resultcm) += -2.0 * prod(arg1cm,arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		gpu::matrix<float,row_major> resultrm(rows,columns,1.5);
		noalias(resultrm) += -2.0 * prod(arg1cm,arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		gpu::matrix<float,column_major> resultcm(rows,columns,1.5);
		noalias(resultcm) += -2.0 * prod(arg1cm,arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking dense-dense matrix-matrix assign multiply"<<std::endl;
	//testexpressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		gpu::matrix<float,row_major> resultrm(rows,columns,1.5);
		noalias(resultrm) = -2.0 * prod(arg1rm,arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		gpu::matrix<float,column_major> resultcm(rows,columns,1.5);
		noalias(resultcm) = -2.0 * prod(arg1rm,arg2rm);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		gpu::matrix<float,row_major> resultrm(rows,columns,1.5);
		noalias(resultrm) = -2.0 * prod(arg1rm,arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		gpu::matrix<float,column_major> resultcm(rows,columns,1.5);
		noalias(resultcm) = -2.0 * prod(arg1rm,arg2cm);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		gpu::matrix<float,row_major> resultrm(rows,columns,1.5);
		noalias(resultrm) = -2.0 * prod(arg1cm,arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		gpu::matrix<float,column_major> resultcm(rows,columns,1.5);
		noalias(resultcm) = -2.0 * prod(arg1cm,arg2rm);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		gpu::matrix<float,row_major> resultrm(rows,columns,1.5);
		noalias(resultrm) = -2.0 * prod(arg1cm,arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		gpu::matrix<float,column_major> resultcm(rows,columns,1.5);
		noalias(resultcm) = -2.0 * prod(arg1cm,arg2cm);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}

BOOST_AUTO_TEST_SUITE_END()
