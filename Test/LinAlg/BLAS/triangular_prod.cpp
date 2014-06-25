#define BOOST_TEST_MODULE LinAlg_axpy_prod
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;
using namespace blas;

template<class M, class V, class Result>
void checkMatrixVectorMultiply(M const& arg1, V const& arg2, Result const& result){
	BOOST_REQUIRE_EQUAL(arg1.size1(), result.size());
	BOOST_REQUIRE_EQUAL(arg2.size(), arg1.size2());
	
	for(std::size_t i = 0; i != arg1.size1(); ++i){
		double test_result = 0.0;
		for(std::size_t k = 0; k != arg1.size2(); ++k){
			test_result += arg1(i,k)*arg2(k);
		}
		BOOST_CHECK_CLOSE(result(i), test_result,1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_axpy_prod_matrix_vector_dense ){
	std::size_t dims = 231;//chosen as not to be a multiple of the block size
	//initialize the arguments in both row and column major, lower and upper, unit and non-unit diagonal
	//we add one on the remaining elements to ensure, that triangular_prod does not tuch these elements
	matrix<double,row_major> arg1lowerrm(dims,dims,1.0);
	matrix<double,column_major> arg1lowercm(dims,dims,1.0);
	matrix<double,row_major> arg1upperrm(dims,dims,1.0);
	matrix<double,column_major> arg1uppercm(dims,dims,1.0);
	
	//inputs to compare to with the standard prod
	matrix<double,row_major> arg1lowertest(dims,dims,0.0);
	matrix<double,row_major> arg1uppertest(dims,dims,0.0);
	for(std::size_t i = 0; i != dims; ++i){
		for(std::size_t j = 0; j <=i; ++j){
			arg1lowerrm(i,j) = arg1lowercm(i,j) = i*dims+0.2*j+1;
			arg1lowertest(i,j) = i*dims+0.2*j+1;
			arg1upperrm(j,i) = arg1uppercm(j,i) = i*dims+0.2*j+1;
			arg1uppertest(j,i) = i*dims+0.2*j+1;
		}
	}
	vector<double> arg2(dims);
	for(std::size_t j = 0; j != dims; ++j){
		arg2(j)  = 1.5*j+2;
	}

	std::cout<<"\nchecking matrix-vector prod non-unit"<<std::endl;
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<Lower>(arg1lowerrm,result);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<Lower>(arg1lowercm,result);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<Upper>(arg1upperrm,result);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<Upper>(arg1uppercm,result);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result);
	}
	
	diag(arg1lowertest) = blas::repeat(1.0,dims);
	diag(arg1uppertest) = blas::repeat(1.0,dims);
	std::cout<<"\nchecking matrix-vector prod unit"<<std::endl;
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<UnitLower>(arg1lowerrm,result);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<UnitLower>(arg1lowercm,result);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<UnitUpper>(arg1upperrm,result);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<UnitUpper>(arg1uppercm,result);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result);
	}
}
