#define BOOST_TEST_MODULE BLAS_GPU_Matrix_Expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/gpu/matrix.hpp>
#include <shark/LinAlg/BLAS/gpu/vector.hpp>
#include <shark/LinAlg/BLAS/gpu/copy.hpp>

using namespace shark;
using namespace blas;


template<class Operation, class Result>
void checkDenseExpressionEquality(
	Operation op_gpu, Result const& result
){
	BOOST_REQUIRE_EQUAL(op_gpu.size1(), result.size1());
	BOOST_REQUIRE_EQUAL(op_gpu.size2(), result.size2());
	
	//check that matrix assignment using op() works(implicit test)
	blas::matrix<float> op = copy_to_cpu(op_gpu);
	for(std::size_t i = 0; i != op.size1(); ++i){
		for(std::size_t j = 0; j != op.size2(); ++j){
			BOOST_CHECK_CLOSE(result(i,j), op(i,j),1.e-8);
		}
	}
	blas::gpu::vector<float> op_row_gpu(op_gpu.size2());
	blas::gpu::vector<float> op_col_gpu(op_gpu.size1());
	blas::vector<float> op_row(op_gpu.size2());
	blas::vector<float> op_col(op_gpu.size2());
	//check that row iterator work
	for(std::size_t i = 0; i != op.size1(); ++i){
		
		boost::compute::copy(op_gpu.row_begin(i), op_gpu.row_end(i), op_row_gpu.begin());
		boost::compute::copy(op_row_gpu.begin(), op_row_gpu.end(), op_row.begin());
		
		for(std::size_t j = 0; j != op.size2(); ++j){
			BOOST_CHECK_CLOSE(result(i,j), op_row(j),1.e-8);
		}
	}
	
	//check that column iterator work
	for(std::size_t j = 0; j != op.size2(); ++j){
		boost::compute::copy(op_gpu.column_begin(j), op_gpu.column_end(j), op_col_gpu.begin());
		boost::compute::copy(op_col_gpu.begin(), op_col_gpu.end(), op_col.begin());
		for(std::size_t i = 0; i != op.size1(); ++i){
			BOOST_CHECK_CLOSE(result(i,j), op_col(i),1.e-8);
		}
	}
}

std::size_t Dimension1 = 50;
std::size_t Dimension2 = 100;


/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_SUITE (BLAS_GPU_matrix_expression)

BOOST_AUTO_TEST_CASE( BLAS_matrix_Unary_Minus )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0+j;
			result(i,j)= -x_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(-x,result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Scalar_Multiply )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0+j;
			result(i,j)= 5.0* x_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(5.0*x,result);
	checkDenseExpressionEquality(x*5.0,result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Scalar_Div )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0+j;
			result(i,j)= x_cpu(i,j)/5.0;
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(x/5.0f,result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Abs )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::abs(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(abs(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Sqr )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= x_cpu(i,j) * x_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sqr(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Sqrt )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::sqrt(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sqrt(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Exp )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::exp(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(exp(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Log )
{

	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::log(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(log(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Tanh )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::tanh(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(tanh(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Sigmoid )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= 1.0/(1.0+std::exp(-x_cpu(i,j)));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sigmoid(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_SoftPlus )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= shark::softPlus(x_cpu(i,j));
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(softPlus(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Pow )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			result(i,j)= std::pow(x_cpu(i,j),3.2);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(pow(x,3.2),result);
}

/////////////////////////////////////////////////////
///////BINARY OPERATIONS//////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( BLAS_matrix_Binary_Plus)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			y_cpu(i,j) = i+j+Dimension1;
			result(i,j)= x_cpu(i,j)+y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x+y,result);
}
BOOST_AUTO_TEST_CASE( BLAS_matrix_Binary_Minus)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			y_cpu(i,j) = i+j+Dimension1;
			result(i,j)= x_cpu(i,j)-y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x-y,result);
}

BOOST_AUTO_TEST_CASE( BLAS_matrix_Binary_Multiply)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			y_cpu(i,j) = i+j+Dimension1;
			result(i,j)= x_cpu(i,j)*y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x*y,result);
	checkDenseExpressionEquality(element_prod(x,y),result);
}

BOOST_AUTO_TEST_CASE( BLAS_matrix_Binary_Div)
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i-3.0-j;
			y_cpu(i,j) = i+j+1;
			result(i,j)= x_cpu(i,j)+y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x/y,result);
	checkDenseExpressionEquality(element_div(x,y),result);
}

BOOST_AUTO_TEST_CASE( BLAS_matrix_Safe_Div )
{
	matrix<float> x_cpu(Dimension1, Dimension2); 
	matrix<float> y_cpu(Dimension1, Dimension2); 
	matrix<float> result(Dimension1, Dimension2);
	for (size_t i = 0; i < Dimension1; i++){
		for (size_t j = 0; j < Dimension2; j++){
			x_cpu(i,j) = i+j;
			y_cpu(i,j) = (i+j)%3;
			result(i,j)= ((i+j) % 3 == 0)? 2.0: x_cpu(i,j)/y_cpu(i,j);
		}
	}
	gpu::matrix<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::matrix<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(safe_div(x,y,2.0),result);
}

BOOST_AUTO_TEST_SUITE_END()
