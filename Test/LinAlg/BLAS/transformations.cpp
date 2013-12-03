#define BOOST_TEST_MODULE LinAlg_VectorTransformations
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/Base.h>

using namespace shark;
using namespace std;

const size_t Dimensions=3;

double target[Dimensions] = {3, 4, 15};

BOOST_AUTO_TEST_CASE( LinAlg_Exp )
{
	RealVector x(Dimensions); 
	RealMatrix xm(Dimensions,1);
	RealVector result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		xm(i,0) = i;
		result(i)=std::exp(x(i));
	}
	RealVector y = exp(x);
	RealMatrix ym = exp(xm);
	//check result
	for (size_t i = 0; i < Dimensions; i++)
	{
		BOOST_CHECK_SMALL(y(i) - result(i),1.e-10);
		BOOST_CHECK_SMALL(ym(i,0) - result(i),1.e-10);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_Log )
{

	RealVector x(Dimensions); 
	RealMatrix xm(Dimensions,1);
	RealVector result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i+1;
		xm(i,0) = i+1;
		result(i)=std::log(x(i));
	}
	RealVector y = log(x);
	RealMatrix ym = log(xm);
	//check result
	for (size_t i = 0; i < Dimensions; i++)
	{
		BOOST_CHECK_SMALL(y(i) - result(i),1.e-10);
		BOOST_CHECK_SMALL(ym(i,0) - result(i),1.e-10);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_Tanh )
{
	RealVector x(Dimensions); 
	RealMatrix xm(Dimensions,1);
	RealVector result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		xm(i,0) = i;
		result(i)=std::tanh(x(i));
	}
	RealVector y = tanh(x);
	RealMatrix ym = tanh(xm);
	//check result
	for (size_t i = 0; i < Dimensions; i++)
	{
		BOOST_CHECK_SMALL(y(i) - result(i),1.e-10);
		BOOST_CHECK_SMALL(ym(i,0) - result(i),1.e-10);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_Sigmoid )
{
	RealVector x(Dimensions); 
	RealMatrix xm(Dimensions,1);
	RealVector result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		xm(i,0) = i;
		result(i) = sigmoid(x(i));
	}
	RealVector y = sigmoid(x);
	RealMatrix ym = sigmoid(xm);
	//check result
	for (size_t i = 0; i < Dimensions; i++)
	{
		BOOST_CHECK_SMALL(y(i) - result(i),1.e-10);
		BOOST_CHECK_SMALL(ym(i,0) - result(i),1.e-10);
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_Safe_Div )
{
	RealVector x(Dimensions); 
	RealVector y(Dimensions); 
	RealMatrix xm(Dimensions,1);
	RealMatrix ym(Dimensions,1);
	RealVector result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		y(i) = i % 3;
		xm(i,0) = i;
		ym(i,0) = i % 3;
		result(i) = (i % 3 == 0)? 2.0: x(i)/y(i);
	}
	RealVector z = safe_div(x,y,2.0);
	RealMatrix zm = safe_div(xm,ym,2.0);
	//check result
	for (size_t i = 0; i < Dimensions; i++)
	{
		BOOST_CHECK_SMALL(z(i) - result(i),1.e-10);
		BOOST_CHECK_SMALL(zm(i,0) - result(i),1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_SoftPlus )
{
	RealVector x(Dimensions); 
	RealMatrix xm(Dimensions,1);
	RealVector result(Dimensions);
	
	for (size_t i = 0; i < Dimensions; i++)
	{
		x(i) = i;
		xm(i,0) = i;
		result(i) = softPlus(x(i));
	}
	RealVector y = softPlus(x);
	RealMatrix ym = softPlus(xm);
	//check result
	for (size_t i = 0; i < Dimensions; i++)
	{
		BOOST_CHECK_SMALL(y(i) - result(i),1.e-10);
		BOOST_CHECK_SMALL(ym(i,0) - result(i),1.e-10);
	}
}

//when dense works, this tests, whether sparse input is also correct
BOOST_AUTO_TEST_CASE( LinAlg_UnaryTransformation_Sparse_Vector ){
    //we first check, that transformations which allow sparseness are actually sparse. 
    CompressedRealVector compressed(100);
    compressed(10) = 2;
    compressed(50) = 3;
    {
		typedef blas::vector_unary<CompressedRealVector, blas::scalar_sqr<double> > SqrExpression;
		SqrExpression sqrexpr = sqr(compressed);
	
		SqrExpression::const_iterator iter = sqrexpr.begin();
		BOOST_CHECK_EQUAL(iter.index(),10);
		BOOST_CHECK_SMALL(*iter-4, 1.e-15);
		++iter;
		BOOST_CHECK_EQUAL(iter.index(),50);
		BOOST_CHECK_SMALL(*iter-9, 1.e-15);
		++iter;
		BOOST_CHECK(iter==sqrexpr.end());
	}
	
	//won't fix. This is stupid.
	//~ {
		//~ //now we check, that transformations which don't allow, are dense.
		//~ typedef blas::vector_unary<CompressedRealVector, blas::scalar_exp<double> > ExpExpression;
		//~ ExpExpression expexpr = exp(compressed);
	
		//~ ExpExpression::const_iterator iter = expexpr.begin();
		//~ std::size_t dist = std::distance(expexpr.begin(),expexpr.end());
		//~ BOOST_REQUIRE_EQUAL(dist,100);
		//~ for(std::size_t i = 0; i != 10; ++i){
			//~ BOOST_CHECK_EQUAL(iter.index(),i);
			//~ BOOST_CHECK_EQUAL(*iter, 1.0);
			//~ ++iter;
		//~ }
		//~ BOOST_CHECK_EQUAL(iter.index(),10);
		//~ BOOST_CHECK_SMALL(*iter-std::exp(2.0), 1.e-15);
		//~ ++iter;
		//~ for(std::size_t i = 11; i != 50; ++i){
			//~ BOOST_CHECK_EQUAL(iter.index(),i);
			//~ BOOST_CHECK_EQUAL(*iter, 1.0);
			//~ ++iter;
		//~ }
		//~ BOOST_CHECK_EQUAL(iter.index(),50);
		//~ BOOST_CHECK_SMALL(*iter-std::exp(3.0), 1.e-15);
		//~ ++iter;
		//~ for(std::size_t i = 51; i != 100; ++i){
			//~ BOOST_CHECK_EQUAL(iter.index(),i);
			//~ BOOST_CHECK_EQUAL(*iter, 1.0);
			//~ ++iter;
		//~ }
		//~ BOOST_CHECK(iter==expexpr.end());
	//~ }
}

BOOST_AUTO_TEST_CASE( LinAlg_UnaryTransformation_Sparse_Matrix ){
	//we first check, that transformations which allow sparseness are actually sparse. 
	CompressedIntMatrix compressed(10,9);
	compressed(5,5) = 2;
	compressed(8,2) = 3;
	compressed(8,5) = 5;
	{
		typedef blas::matrix_unary<CompressedIntMatrix, blas::scalar_sqr<int> > SqrExpression;
		SqrExpression sqrexpr = sqr(compressed);
	    
		BOOST_CHECK_EQUAL(sqrexpr.size1(),10);
		BOOST_CHECK_EQUAL(sqrexpr.size2(),9);
		for(std::size_t i = 0; i != 10; ++i){
			int elements = std::distance(sqrexpr.row_begin(i),sqrexpr.row_end(i));
			if(i == 5){
				BOOST_CHECK_EQUAL(elements,1);
				BOOST_CHECK_EQUAL(*sqrexpr.row_begin(i),4);
			}else if(i == 8){
				BOOST_CHECK_EQUAL(elements,2);
				BOOST_CHECK_EQUAL(*sqrexpr.row_begin(i),9);
				BOOST_CHECK_EQUAL(*(++sqrexpr.row_begin(i)),25);
			}else{
				BOOST_CHECK_EQUAL(elements,0);
			}
		}
	}
	//won't fix. This is stupid.
	//~ {
		//~ //now we check, that transformations which don't allow, are dense.
		//~ typedef blas::matrix_unary<CompressedRealMatrix, blas::scalar_exp<double> > SqrExpression;
		//~ ExpExpression expexpr = exp(compressed);
	
		//~ std::size_t i = 0;
		//~ for(ExpExpression::const_iterator1 iter = expexpr.begin1(); iter != expexpr.end1(); ++iter,++i){
			//~ std::size_t j = 0;
			//~ for(ExpExpression::const_iterator2 iter2 = iter.begin(); iter2 != iter.end(); ++iter2,++j){
				//~ BOOST_CHECK_EQUAL(iter2.index1(),i);
				//~ BOOST_CHECK_EQUAL(iter2.index2(),j);
				//~ if(i == 5 && j ==5){
					//~ BOOST_CHECK_SMALL(*iter2-std::exp(2.0), 1.e-15);
				//~ } else if(i == 8 && j == 2){
					//~ BOOST_CHECK_SMALL(*iter2-std::exp(3.0), 1.e-15);
				//~ } else
					//~ BOOST_CHECK_SMALL(*iter2-1.0, 1.e-15);
			//~ }
		//~ }
		
		//~ std::size_t j = 0;
		//~ for(ExpExpression::const_iterator2 iter2 = expexpr.begin2(); iter2 != expexpr.end2(); ++iter2,++j){
			//~ std::size_t i = 0;
			//~ for(ExpExpression::const_iterator1 iter = iter2.begin(); iter != iter2.end(); ++iter,++i){
				//~ BOOST_CHECK_EQUAL(iter.index1(),i);
				//~ BOOST_CHECK_EQUAL(iter.index2(),j);
				//~ if(i == 5 && j ==5){
					//~ BOOST_CHECK_SMALL(*iter-std::exp(2.0), 1.e-15);
				//~ } else if(i == 8 && j == 2){
				//~ BOOST_CHECK_SMALL(*iter-std::exp(3.0), 1.e-15);
				//~ } else
					//~ BOOST_CHECK_SMALL(*iter-1.0, 1.e-15);
			//~ }
		//~ }
		
	//~ }
}

