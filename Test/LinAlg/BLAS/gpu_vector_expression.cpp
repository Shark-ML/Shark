#define BOOST_TEST_MODULE BLAS_GPU_vector_expression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/gpu/vector.hpp>
#include <shark/LinAlg/BLAS/gpu/copy.hpp>

using namespace shark;
using namespace blas;


template<class Operation, class Result>
void checkDenseExpressionEquality(
	Operation op_gpu, Result const& result
){
	BOOST_REQUIRE_EQUAL(op_gpu.size(), result.size());
	
	blas::vector<float> op = copy_to_cpu(op_gpu);
	for(std::size_t i = 0; i != op.size(); ++i){
		BOOST_CHECK_CLOSE(result(i), op(i),1.e-8);
	}
}

const std::size_t Dimensions = 1000;

/////////////////////////////////////////////////////////////
//////UNARY TRANSFORMATIONS///////
////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_SUITE (LinAlg_BLAS_vector_expression)

BOOST_AUTO_TEST_CASE( BLAS_Vector_Unary_Minus )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i-3.0;
		result(i)= -x_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(-x,result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Scalar_Multiply )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i-3.0;
		result(i)= 5.0*x_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(5.0*x,result);
	checkDenseExpressionEquality(x*5.0,result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Scalar_Div )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i-3.0;
		result(i)= x_cpu(i)/5.0;
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(x/5.0f,result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Abs )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = -i+3.0;
		result(i)= std::abs(x_cpu(i));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(abs(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Sqr )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = -i+3.0;
		result(i)= x_cpu(i)*x_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sqr(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Sqrt )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		result(i)= sqrt(x_cpu(i));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sqrt(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Exp )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		result(i)=std::exp(x_cpu(i));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(exp(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Log )
{

	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i+1;
		result(i)=std::log(x_cpu(i));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(log(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Tanh )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = i;
		result(i)=std::tanh(x_cpu(i));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(tanh(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Sigmoid )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = i;
		result(i) = 1.0/(1.0+std::exp(-x_cpu(i)));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(sigmoid(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_SoftPlus )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		result(i) = shark::softPlus(x_cpu(i));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(softPlus(x),result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Pow )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i+1.0;
		result(i)= std::pow(x_cpu(i),3.2);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	checkDenseExpressionEquality(pow(x,3.2),result);
}

/////////////////////////////////////////////////////
///////BINARY OPERATIONS//////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( BLAS_Vector_Binary_Plus)
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		y_cpu(i) = i+Dimensions;
		result(i) = x_cpu(i)+y_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::vector<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x+y,result);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Binary_Minus)
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		y_cpu(i) = -3.0*i+Dimensions;
		result(i) = x_cpu(i)-y_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::vector<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x-y,result);
}

BOOST_AUTO_TEST_CASE( BLAS_Vector_Binary_Multiply)
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		y_cpu(i) = -3.0*i+Dimensions;
		result(i) = x_cpu(i)*y_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::vector<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x*y,result);
	checkDenseExpressionEquality(element_prod(x,y),result);
}

BOOST_AUTO_TEST_CASE( BLAS_Vector_Binary_Div)
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = 3.0*i+3.0;
		y_cpu(i) = i+1;
		result(i) = x_cpu(i)/y_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::vector<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(x/y,result);
	checkDenseExpressionEquality(element_div(x,y),result);
}

BOOST_AUTO_TEST_CASE( BLAS_Vector_Safe_Div )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	vector<float> result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x_cpu(i) = i;
		y_cpu(i) = i % 3;
		result(i) = (i % 3 == 0)? 2.0: x_cpu(i)/y_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::vector<float> y = gpu::copy_to_gpu(y_cpu);
	checkDenseExpressionEquality(safe_div(x,y,2.0),result);
}

/////////////////////////////////////////////////////
///////////Vector Reductions///////////
/////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( BLAS_Vector_Max )
{
	vector<float> x_cpu(Dimensions); 
	float result = 1;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = exp(-(i-5.0)*(i-5.0));//max at i = 5
		result = std::max(result,x_cpu(i));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(max(x),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_Min )
{
	vector<float> x_cpu(Dimensions); 
	float result = -1;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = -std::exp(-(i-5.0)*(i-5.0));//min at i = 5
		result = std::min(result,x_cpu(i));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(min(x),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( BLAS_Vector_Arg_Max )
{
	vector<float> x_cpu(Dimensions); 
	unsigned int result = 5;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = exp(-(i-5.0)*(i-5.0));//max at i = 5
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_EQUAL(arg_max(x),result);
}

BOOST_AUTO_TEST_CASE( BLAS_Vector_Arg_Min )
{
	vector<float> x_cpu(Dimensions); 
	unsigned int result = 5;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = -exp(-(i-5.0)*(i-5.0));//min at i = 5
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_EQUAL(arg_min(x),result);
}

BOOST_AUTO_TEST_CASE( BLAS_Vector_Sum )
{
	vector<float> x_cpu(Dimensions); 
	float result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = 2*i-5;
		result +=x_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(sum(x),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( BLAS_Vector_norm_1 )
{
	vector<float> x_cpu(Dimensions); 
	float result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = 2*i-5;
		result +=std::abs(x_cpu(i));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(norm_1(x),result,1.e-10);
}

BOOST_AUTO_TEST_CASE( BLAS_Vector_norm_sqr )
{
	vector<float> x_cpu(Dimensions); 
	float result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = 2*i-5;
		result +=x_cpu(i)*x_cpu(i);
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(norm_sqr(x),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_norm_2 )
{
	vector<float> x_cpu(Dimensions); 
	float result = 0;
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = 2*i-5;
		result +=x_cpu(i)*x_cpu(i);
	}
	result = std::sqrt(result);
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_CLOSE(norm_2(x),result,1.e-10);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_norm_inf )
{
	vector<float> x_cpu(Dimensions); 
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = exp(-(i-5.0)*(i-5.0));
	}
	x_cpu(8)=-2;
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_EQUAL(norm_inf(x),2.0);
}
BOOST_AUTO_TEST_CASE( BLAS_Vector_index_norm_inf )
{
	vector<float> x_cpu(Dimensions); 
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = exp(-(i-5.0)*(i-5.0));
	}
	x_cpu(8)=-2;
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	BOOST_CHECK_EQUAL(index_norm_inf(x),8);
}

BOOST_AUTO_TEST_CASE( BLAS_Vector_inner_prod )
{
	vector<float> x_cpu(Dimensions); 
	vector<float> y_cpu(Dimensions); 
	
	for (size_t i = 0; i < Dimensions; i++){
		x_cpu(i) = exp(-(i-5.0)*(i-5.0));
		y_cpu(i) = exp((i-5.0)*(i-5.0));
	}
	gpu::vector<float> x = gpu::copy_to_gpu(x_cpu);
	gpu::vector<float> y = gpu::copy_to_gpu(y_cpu);
	BOOST_CHECK_CLOSE(inner_prod(x,y),(float)Dimensions,1.e-5);
}


BOOST_AUTO_TEST_SUITE_END()
