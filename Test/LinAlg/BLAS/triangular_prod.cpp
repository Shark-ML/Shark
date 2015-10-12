#define BOOST_TEST_MODULE LinAlg_triangular_prod
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;
using namespace blas;

template<class M, class V, class Result>
void checkMatrixVectorMultiply(M const& arg1, V const& arg2, Result const& result){
	BOOST_REQUIRE_EQUAL(arg1.size1(), result.size());
	BOOST_REQUIRE_EQUAL(arg2.size(), arg1.size2());

	for(std::size_t i = 0; i != arg1.size1(); ++i) {
		double test_result = inner_prod(row(arg1,i),arg2);
		BOOST_CHECK_CLOSE(result(i), test_result, 1.e-10);
	}
}

template<class M1, class M2, class Result>
void checkMatrixMatrixMultiply(M1 const& arg1, M2 const& arg2, Result const& result) {
	BOOST_REQUIRE_EQUAL(arg1.size1(), arg1.size2());
	BOOST_REQUIRE_EQUAL(arg1.size2(), arg2.size1());
	BOOST_REQUIRE_EQUAL(arg1.size2(), result.size1());
	BOOST_REQUIRE_EQUAL(arg2.size2(), result.size2());
	
	for(std::size_t i = 0; i != arg2.size1(); ++i) {
		for(std::size_t j = 0; j != arg2.size2(); ++j) {
			double test_result = inner_prod(row(arg1,i),column(arg2,j));
			BOOST_CHECK_CLOSE(result(i,j), test_result, 1.e-10);
		}
	}
}

BOOST_AUTO_TEST_SUITE(LinAlg_BLAS_triangular_prod)

BOOST_AUTO_TEST_CASE(LinAlg_triangular_prod_matrix_vector) {
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
		triangular_prod<lower>(arg1lowerrm,result);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<lower>(arg1lowercm,result);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<upper>(arg1upperrm,result);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<upper>(arg1uppercm,result);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result);
	}
	
	diag(arg1lowertest) = blas::repeat(1.0,dims);
	diag(arg1uppertest) = blas::repeat(1.0,dims);
	std::cout<<"\nchecking matrix-vector prod unit"<<std::endl;
	{
		std::cout<<"row major lower Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<unit_lower>(arg1lowerrm,result);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result);
	}
	{
		std::cout<<"column major lower Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<unit_lower>(arg1lowercm,result);
		checkMatrixVectorMultiply(arg1lowertest,arg2,result);
	}
	{
		std::cout<<"row major upper Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<unit_upper>(arg1upperrm,result);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result);
	}
	{
		std::cout<<"column major upper Ax"<<std::endl;
		vector<double> result = arg2;
		triangular_prod<unit_upper>(arg1uppercm,result);
		checkMatrixVectorMultiply(arg1uppertest,arg2,result);
	}
}


BOOST_AUTO_TEST_CASE(LinAlg_triangular_prod_matrix_matrix) {
	std::size_t dims = 231;//chosen as not to be a multiple of the block size
	std::size_t N = 30;
	//initialize the arguments in both row and column major, lower and upper, unit and non-unit diagonal
	//we add one on the remaining elements to ensure, that triangular_prod does not tuch these elements
	matrix<double, row_major> arg1lowerrm(dims, dims, 1.0);
	matrix<double, column_major> arg1lowercm(dims, dims, 1.0);
	matrix<double, row_major> arg1upperrm(dims, dims, 1.0);
	matrix<double, column_major> arg1uppercm(dims, dims, 1.0);

	//inputs to compare to with the standard prod
	matrix<double, row_major> arg1lowertest(dims, dims, 0.0);
	matrix<double, row_major> arg1uppertest(dims, dims, 0.0);
	for(std::size_t i = 0; i != dims; ++i) {
		for(std::size_t j = 0; j <= i; ++j) {
			arg1lowerrm(i, j) = arg1lowercm(i, j) = i * dims + 0.2 * j + 1;
			arg1lowertest(i, j) = i * dims + 0.2 * j + 1;
			arg1upperrm(j, i) = arg1uppercm(j, i) = i * dims + 0.2 * j + 1;
			arg1uppertest(j, i) = i * dims + 0.2 * j + 1;
		}
	}
	matrix<double> arg2(dims,N);
	for(std::size_t i = 0; i != dims; ++i) {
		for(std::size_t j = 0; j != N; ++j) {
			arg2(i,j)  = 1.5 * j + 2+i;
		}
	}

	std::cout << "\nchecking matrix-matrix prod non-unit" << std::endl;
	{
		std::cout << "row major lower AX" << std::endl;
		matrix<double> result = arg2;
		triangular_prod<lower>(arg1lowerrm, result);
		checkMatrixMatrixMultiply(arg1lowertest, arg2, result);
	}
	{
		std::cout << "row major upper AX" << std::endl;
		matrix<double> result = arg2;
		triangular_prod<upper>(arg1upperrm, result);
		checkMatrixMatrixMultiply(arg1uppertest, arg2, result);
	}
	{
		std::cout << "column major lower AX" << std::endl;
		matrix<double> result = arg2;
		triangular_prod<lower>(arg1lowercm, result);
		checkMatrixMatrixMultiply(arg1lowertest, arg2, result);
	}
	{
		std::cout << "column major upper AX" << std::endl;
		matrix<double> result = arg2;
		triangular_prod<upper>(arg1uppercm, result);
		checkMatrixMatrixMultiply(arg1uppertest, arg2, result);
	}

	diag(arg1lowertest) = blas::repeat(1.0, dims);
	diag(arg1uppertest) = blas::repeat(1.0, dims);
	std::cout << "\nchecking matrix-matrix prod unit" << std::endl;
	{
		std::cout << "row major lower AX" << std::endl;
		matrix<double> result = arg2;
		triangular_prod<unit_lower>(arg1lowerrm, result);
		checkMatrixMatrixMultiply(arg1lowertest, arg2, result);
	}
	{
		std::cout << "row major upper AX" << std::endl;
		matrix<double> result = arg2;
		triangular_prod<unit_upper>(arg1upperrm, result);
		checkMatrixMatrixMultiply(arg1uppertest, arg2, result);
	}
	{
		std::cout << "column major lower AX" << std::endl;
		matrix<double> result = arg2;
		triangular_prod<unit_lower>(arg1lowercm, result);
		checkMatrixMatrixMultiply(arg1lowertest, arg2, result);
	}
	{
		std::cout << "column major upper AX" << std::endl;
		matrix<double> result = arg2;
		triangular_prod<unit_upper>(arg1uppercm, result);
		checkMatrixMatrixMultiply(arg1uppertest, arg2, result);
	}
}

BOOST_AUTO_TEST_SUITE_END()
