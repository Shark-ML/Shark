#define BOOST_TEST_MODULE LinAlg_axpy_prod
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;
using namespace blas;

template<class M, class V, class Result>
void checkMatrixVectorMultiply(M const& arg1, V const& arg2, Result const& result, double factor, double init = 0){
	BOOST_REQUIRE_EQUAL(arg1.size1(), result.size());
	BOOST_REQUIRE_EQUAL(arg2.size(), arg1.size2());
	
	for(std::size_t i = 0; i != arg1.size1(); ++i){
		double test_result = init;
		for(std::size_t k = 0; k != arg1.size2(); ++k){
			test_result += factor * arg1(i,k)*arg2(k);
		}
		BOOST_CHECK_CLOSE(result(i), test_result,1.e-10);
	}
}

BOOST_AUTO_TEST_SUITE (LinAlg_BLAS_axpy_prod)

BOOST_AUTO_TEST_CASE( LinAlg_axpy_prod_matrix_vector_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	matrix<double,row_major> arg1rm(rows,columns);
	matrix<double,column_major> arg1cm(rows,columns);
	matrix<double,row_major> arg1rmt(columns,rows);
	matrix<double,column_major> arg1cmt(columns,rows);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*columns+0.2*j;
			arg1rmt(j,i) = arg1cmt(j,i) = i*columns+0.2*j;
		}
	}
	vector<double> arg2(columns);
	for(std::size_t j = 0; j != columns; ++j){
		arg2(j)  = 1.5*j+2;
	}

	std::cout<<"\nchecking dense matrix vector plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1rm,arg2,result,false,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1cm,arg2,result,false,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1rmt,result,false,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1cmt,result,false,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking dense matrix vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1rm,arg2,result,true,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1cm,arg2,result,true,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1rmt,result,true,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1cmt,result,true,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_axpy_prod_matrix_vector_dense_sparse ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	matrix<double,row_major> arg1rm(rows,columns);
	matrix<double,column_major> arg1cm(rows,columns);
	matrix<double,row_major> arg1rmt(columns,rows);
	matrix<double,column_major> arg1cmt(columns,rows);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*columns+0.2*j;
			arg1rmt(j,i) = arg1cmt(j,i) = i*columns+0.2*j;
		}
	}
	compressed_vector<double> arg2(columns);
	for(std::size_t j = 1; j < columns; j+=3){
		arg2(j)  = 1.5*j+2;
	}

	std::cout<<"\nchecking dense-sparse matrix vector plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1rm,arg2,result,false,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1cm,arg2,result,false,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1rmt,result,false,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1cmt,result,false,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking dense-sparse matrix vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1rm,arg2,result,true,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1cm,arg2,result,true,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1rmt,result,true,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1cmt,result,true,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_axpy_prod_matrix_vector_sparse_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	compressed_matrix<double> arg1rm(rows,columns);
	compressed_matrix<double>  arg1cm_base(columns,rows);
	matrix_transpose<compressed_matrix<double> >  arg1cm = trans(arg1cm_base);
	compressed_matrix<double>  arg1rmt(columns,rows);
	compressed_matrix<double>  arg1cmt_base(rows,columns);
	matrix_transpose<compressed_matrix<double> >  arg1cmt = trans(arg1cmt_base);
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 1; j < 20; j+=(i+1)){
			arg1rm(i,j) = arg1cm(i,j) = 2*(20*i+1)+1;
			arg1rmt(j,i) = arg1cmt(j,i) = arg1rm(i,j);
		}
	}

	vector<double> arg2(columns);
	for(std::size_t j = 0; j != columns; ++j){
		arg2(j)  = 1.5*j+2;
	}

	std::cout<<"\nchecking sparse-dense matrix vector plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1rm,arg2,result,false,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1cm,arg2,result,false,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1rmt,result,false,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1cmt,result,false,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking sparse-dense matrix vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1rm,arg2,result,true,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1cm,arg2,result,true,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1rmt,result,true,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1cmt,result,true,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_axpy_prod_matrix_vector_sparse_sparse ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	//initialize the arguments in both row and column major as well as transposed
	compressed_matrix<double> arg1rm(rows,columns);
	compressed_matrix<double>  arg1cm_base(columns,rows);
	matrix_transpose<compressed_matrix<double> >  arg1cm = trans(arg1cm_base);
	compressed_matrix<double>  arg1rmt(columns,rows);
	compressed_matrix<double>  arg1cmt_base(rows,columns);
	matrix_transpose<compressed_matrix<double> >  arg1cmt = trans(arg1cmt_base);
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 1; j < 20; j+=(i+1)){
			arg1rm(i,j) = arg1cm(i,j) = 2*(20*i+1)+1;
			arg1rmt(j,i) = arg1cmt(j,i) = arg1rm(i,j);
		}
	}

	compressed_vector<double> arg2(columns);
	for(std::size_t j = 1; j < columns; j+=3){
		arg2(j)  = 1.5*j+2;
	}

	std::cout<<"\nchecking sparse-sparse matrix vector plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1rm,arg2,result,false,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1cm,arg2,result,false,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1rmt,result,false,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,1.5);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1cmt,result,false,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,1.5);
	}
	std::cout<<"\nchecking sparse-sparse matrix vector assign multiply"<<std::endl;
	//test expressions of the form A=B*C
	{
		std::cout<<"row major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1rm,arg2,result,true,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major Ax"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg1cm,arg2,result,true,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"row major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1rmt,result,true,-2.0);
		checkMatrixVectorMultiply(arg1rm,arg2,result,-2.0,0);
	}
	{
		std::cout<<"column major xA"<<std::endl;
		vector<double> result(rows,1.5);
		axpy_prod(arg2,arg1cmt,result,true,-2.0);
		checkMatrixVectorMultiply(arg1cm,arg2,result,-2.0,0);
	}
}

//we test using the textbook definition.
template<class Arg1, class Arg2, class Result>
void checkMatrixMatrixMultiply(Arg1 const& arg1, Arg2 const& arg2, Result const& result, double factor, double init = 0){
	BOOST_REQUIRE_EQUAL(arg1.size1(), result.size1());
	BOOST_REQUIRE_EQUAL(arg2.size2(), result.size2());
	
	for(std::size_t i = 0; i != arg1.size1(); ++i){
		for(std::size_t j = 0; j != arg2.size2(); ++j){
			double test_result = init;
			for(std::size_t k = 0; k != arg1.size2(); ++k){
				 test_result += factor * arg1(i,k)*arg2(k,j);
			}
			BOOST_CHECK_CLOSE(result(i,j), test_result,1.e-10);
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_axpy_prod_matrix_matrix_dense_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	matrix<double,row_major> arg1rm(rows,middle);
	matrix<double,column_major> arg1cm(rows,middle);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != middle; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*middle+0.2*j;
		}
	}
	matrix<double,row_major> arg2rm(middle,columns);
	matrix<double,column_major> arg2cm(middle,columns);
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg2rm(i,j) = arg2cm(i,j) = i*columns+1.5*j;
		}
	}
	std::cout<<"\nchecking dense-dense matrix matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking dense-dense matrix matrix assign multiply"<<std::endl;
	//testexpressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}

//second argument sparse
BOOST_AUTO_TEST_CASE( LinAlg_axpy_prod_matrix_matrix_dense_sparse ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	matrix<double,row_major> arg1rm(rows,middle);
	matrix<double,column_major> arg1cm(rows,middle);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j != middle; ++j){
			arg1rm(i,j) = arg1cm(i,j) = i*middle+0.2*j;
		}
	}
	compressed_matrix<double> arg2rm(middle,columns);
	compressed_matrix<double>  arg2cm_base(columns,middle);
	matrix_transpose<compressed_matrix<double> >  arg2cm = trans(arg2cm_base);
	
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 1; j < columns; j+=(i+1)){
			arg2rm(i,j) = arg2cm(i,j) = 2*(20*i+1)+1;
		}
	}
	std::cout<<"\nchecking dense-sparse matrix matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking dense-sparse matrix matrix assign multiply"<<std::endl;
	//testexpressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}

//first argument sparse
BOOST_AUTO_TEST_CASE( LinAlg_axpy_prod_matrix_matrix_sparse_dense ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	compressed_matrix<double> arg1rm(rows,middle);
	compressed_matrix<double>  arg1cm_base(middle,rows);
	matrix_transpose<compressed_matrix<double> >  arg1cm = trans(arg1cm_base);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 1; j < middle; j+=(i+1)){
			arg1rm(i,j) = arg1cm(i,j) = 2*(20*i+1)+1;
		}
	}
	matrix<double,row_major> arg2rm(middle,columns);
	matrix<double,column_major> arg2cm(middle,columns);
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 0; j != columns; ++j){
			arg2rm(i,j) = arg2cm(i,j) = i*columns+1.5*j;
		}
	}
	
	std::cout<<"\nchecking sparse-dense matrix matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking sparse-dense matrix matrix assign multiply"<<std::endl;
	//testexpressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_axpy_prod_matrix_matrix_sparse_sparse ){
	std::size_t rows = 50;
	std::size_t columns = 80;
	std::size_t middle = 33;
	//initialize the arguments in both row and column major
	compressed_matrix<double> arg1rm(rows,middle);
	compressed_matrix<double>  arg1cm_base(middle,rows);
	matrix_transpose<compressed_matrix<double> >  arg1cm = trans(arg1cm_base);
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 1; j < middle; j+=(i+1)){
			arg1rm(i,j) = arg1cm(i,j) = 2*(20*i+1)+1;
		}
	}
	compressed_matrix<double> arg2rm(middle,columns);
	compressed_matrix<double>  arg2cm_base(columns,middle);
	matrix_transpose<compressed_matrix<double> >  arg2cm = trans(arg2cm_base);
	
	for(std::size_t i = 0; i != middle; ++i){
		for(std::size_t j = 1; j < columns; j+=(i+1)){
			arg2rm(i,j) = arg2cm(i,j) = 2*(20*i+1)+1;
		}
	}
	
	std::cout<<"\nchecking sparse-sparse matrix matrix plusassign multiply"<<std::endl;
	//test first expressions of the form A+=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,1.5);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,1.5);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultrm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,1.5);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultcm,false,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,1.5);
	}
	
	std::cout<<"\nchecking sparse-dense matrix matrix assign multiply"<<std::endl;
	//testexpressions of the form A=B*C
	{
		std::cout<<"rrr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rrc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2rm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"rcr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"rcc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1rm,arg2cm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1rm,arg2cm,resultcm,-2.0,0);
	}
	
	{
		std::cout<<"crr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultrm,-2.0,0);
	}
	{
		std::cout<<"crc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2rm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2rm,resultcm,-2.0,0);
	}
	{
		std::cout<<"ccr"<<std::endl;
		matrix<double,row_major> resultrm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultrm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultrm,-2.0,0);
	}
	{
		std::cout<<"ccc"<<std::endl;
		matrix<double,column_major> resultcm(rows,columns,1.5);
		axpy_prod(arg1cm,arg2cm,resultcm,true,-2.0);
		checkMatrixMatrixMultiply(arg1cm,arg2cm,resultcm,-2.0,0);
	}
}
BOOST_AUTO_TEST_SUITE_END()
