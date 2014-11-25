#define BOOST_TEST_MODULE LinAlg_MatrixTriangular
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/triangular_matrix.hpp>
#include <shark/LinAlg/BLAS/matrix.hpp>

using namespace shark;
using namespace blas;

template<class TriangularMatrix>
void testTriangularEqualLower(TriangularMatrix& matrix,blas::matrix<int> const& result,std::size_t rows){
	BOOST_REQUIRE_EQUAL(matrix.size1(),rows);
	BOOST_REQUIRE_EQUAL(matrix.size1(),rows);
	
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j !=rows; ++j){
			BOOST_CHECK_EQUAL(matrix(i,j),result(i,j));
		}
	}
	
	for(std::size_t i = 0; i != rows;++i){
		typename TriangularMatrix::row_iterator pos=matrix.row_begin(i);
		BOOST_REQUIRE(pos < matrix.row_end(i));
		std::size_t posIndex = 0;
		while(pos != matrix.row_end(i)){
			BOOST_REQUIRE(posIndex < rows);
			BOOST_CHECK_EQUAL(pos.index(),posIndex);
			BOOST_CHECK_EQUAL(*pos,result(i,posIndex));
			//test operator [n]
			for(int j=0;j!= (int)i;++j){
				BOOST_CHECK_EQUAL(pos[j-posIndex],result(i,j));
			}
			++pos;
			++posIndex;
		}
		BOOST_CHECK(posIndex == i+1);
	}
	
	for(std::size_t i = 0; i!=rows;++i){
		typename TriangularMatrix::column_iterator pos=matrix.column_begin(i);
		BOOST_REQUIRE(pos < matrix.column_end(i));
		std::size_t posIndex = i;
		while(pos != matrix.column_end(i)){
			BOOST_REQUIRE(posIndex < rows);
			BOOST_CHECK_EQUAL(pos.index(),posIndex);
			BOOST_CHECK_EQUAL(*pos,result(posIndex,i));
			//test operator [n]
			for(int j=i;j!= (int)rows;++j){
				BOOST_CHECK_EQUAL(pos[j-(int)posIndex],result(j,i));
			}
			++pos;
			++posIndex;
		}
		BOOST_CHECK(posIndex == rows);
	}
	
	TriangularMatrix const& cmatrix = matrix;
	
	for(std::size_t i = 0; i != rows;++i){
		typename TriangularMatrix::const_row_iterator pos=cmatrix.row_begin(i);
		BOOST_REQUIRE(pos < cmatrix.row_end(i));
		std::size_t posIndex = 0;
		while(pos != cmatrix.row_end(i)){
			BOOST_REQUIRE(posIndex < rows);
			BOOST_CHECK_EQUAL(pos.index(),posIndex);
			BOOST_CHECK_EQUAL(*pos,result(i,posIndex));
			//test operator [n]
			for(int j=0;j!= (int)i;++j){
				BOOST_CHECK_EQUAL(pos[j-posIndex],result(i,j));
			}
			++pos;
			++posIndex;
		}
		BOOST_CHECK(posIndex == i+1);
	}
	
	for(std::size_t i = 0; i!=rows;++i){
		typename TriangularMatrix::const_column_iterator pos=cmatrix.column_begin(i);
		BOOST_REQUIRE(pos < cmatrix.column_end(i));
		std::size_t posIndex = i;
		while(pos != cmatrix.column_end(i)){
			BOOST_REQUIRE(posIndex < rows);
			BOOST_CHECK_EQUAL(pos.index(),posIndex);
			BOOST_CHECK_EQUAL(*pos,result(posIndex,i));
			//test operator [n]
			for(int j=i;j!= (int)rows;++j){
				BOOST_CHECK_EQUAL(pos[j-(int)posIndex],result(j,i));
			}
			++pos;
			++posIndex;
		}
		BOOST_CHECK(posIndex == rows);
	}
}

template<class TriangularMatrix>
void testTriangularEqualUpper(TriangularMatrix& matrix,blas::matrix<int> const& result,std::size_t rows){
	BOOST_REQUIRE_EQUAL(matrix.size1(),rows);
	BOOST_REQUIRE_EQUAL(matrix.size1(),rows);
	
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t j = 0; j !=rows; ++j){
			BOOST_CHECK_EQUAL(matrix(i,j),result(i,j));
		}
	}
	
	for(std::size_t i = 0; i != rows;++i){
		typename TriangularMatrix::row_iterator pos=matrix.row_begin(i);
		BOOST_REQUIRE(pos < matrix.row_end(i));
		std::size_t posIndex = i;
		while(pos != matrix.row_end(i)){
			BOOST_REQUIRE(posIndex < rows);
			BOOST_CHECK_EQUAL(pos.index(),posIndex);
			BOOST_CHECK_EQUAL(*pos,result(i,posIndex));
			//test operator [n]
			for(int j=i;j!= (int)rows;++j){
				BOOST_CHECK_EQUAL(pos[j-posIndex],result(i,j));
			}
			++pos;
			++posIndex;
		}
		BOOST_CHECK(posIndex == rows);
	}
	
	for(std::size_t i = 0; i!=rows;++i){
		typename TriangularMatrix::column_iterator pos=matrix.column_begin(i);
		BOOST_REQUIRE(pos < matrix.column_end(i));
		std::size_t posIndex = 0;
		while(pos != matrix.column_end(i)){
			BOOST_REQUIRE(posIndex < rows);
			BOOST_CHECK_EQUAL(pos.index(),posIndex);
			BOOST_CHECK_EQUAL(*pos,result(posIndex,i));
			//test operator [n]
			for(int j=0;j!= (int)i;++j){
				BOOST_CHECK_EQUAL(pos[j-(int)posIndex],result(j,i));
			}
			++pos;
			++posIndex;
		}
		BOOST_CHECK(posIndex == i+1);
	}
	
	TriangularMatrix const& cmatrix = matrix;
	
	for(std::size_t i = 0; i != rows;++i){
		typename TriangularMatrix::const_row_iterator pos=cmatrix.row_begin(i);
		BOOST_REQUIRE(pos < cmatrix.row_end(i));
		std::size_t posIndex = i;
		while(pos != cmatrix.row_end(i)){
			BOOST_REQUIRE(posIndex < rows);
			BOOST_CHECK_EQUAL(pos.index(),posIndex);
			BOOST_CHECK_EQUAL(*pos,result(i,posIndex));
			//test operator [n]
			for(int j=i;j!= (int)rows;++j){
				BOOST_CHECK_EQUAL(pos[j-posIndex],result(i,j));
			}
			++pos;
			++posIndex;
		}
		BOOST_CHECK(posIndex == rows);
	}
	
	for(std::size_t i = 0; i!=rows;++i){
		typename TriangularMatrix::const_column_iterator pos=cmatrix.column_begin(i);
		BOOST_REQUIRE(pos < cmatrix.column_end(i));
		std::size_t posIndex = 0;
		while(pos != cmatrix.column_end(i)){
			BOOST_REQUIRE(posIndex < rows);
			BOOST_CHECK_EQUAL(pos.index(),posIndex);
			BOOST_CHECK_EQUAL(*pos,result(posIndex,i));
			//test operator [n]
			for(int j=0;j!= (int)i;++j){
				BOOST_CHECK_EQUAL(pos[j-(int)posIndex],result(j,i));
			}
			++pos;
			++posIndex;
		}
		BOOST_CHECK(posIndex == i+1);
	}
}

struct TriangularMatrixFixture
{
	std::size_t rows;
	std::size_t elements;
	triangular_matrix<int,row_major,lower> matrix1;
	triangular_matrix<int,column_major,lower> matrix2;
	triangular_matrix<int,row_major,upper> matrix3;
	triangular_matrix<int,column_major,upper> matrix4;
	
	matrix<int> result1;
	matrix<int> result2;
	matrix<int> result3;
	matrix<int> result4;
	
	TriangularMatrixFixture():rows(5),elements(15)
	,matrix1(rows),matrix2(rows),matrix3(rows),matrix4(rows){
		int result1array[]={
			1,0,0,0,0,
			2,3,0,0,0,
			4,5,6,0,0,
			7,8,9,10,0,
			11,12,13,14,15
		};
		
		int result2array[]={
			1,0,0,0,0,
			2,6,0,0,0,
			3,7,10,0,0,
			4,8,11,13,0,
			5,9,12,14,15
		};
		
		result1 = adapt_matrix(rows,rows,result1array);
		result2 = adapt_matrix(rows,rows,result2array);
		result3 = trans(result2);
		result4 = trans(result1);
		
		for(std::size_t elem=0;elem != elements;++elem){
			matrix1.storage()[elem]=elem+1;
			matrix2.storage()[elem]=elem+1;
			matrix3.storage()[elem]=elem+1;
			matrix4.storage()[elem]=elem+1;
		}
	}
};

BOOST_FIXTURE_TEST_SUITE (LinAlg_BLAS_triangular_matrix,TriangularMatrixFixture)


//Check that reading entries of the matrix works and it is the same structure
//as demanded by the BLAS Standard
BOOST_AUTO_TEST_CASE( triangular_matrix_structure){
	BOOST_REQUIRE_EQUAL(matrix1.nnz(),elements);
	BOOST_REQUIRE_EQUAL(matrix2.nnz(),elements);
	BOOST_REQUIRE_EQUAL(matrix3.nnz(),elements);
	BOOST_REQUIRE_EQUAL(matrix4.nnz(),elements);

	//test that matrices are the same
	
	std::cout<<"test lower/row_major"<<std::endl;
	testTriangularEqualLower(matrix1,result1,rows);
	std::cout<<"test lower/column_major"<<std::endl;
	testTriangularEqualLower(matrix2,result2,rows);
	std::cout<<"test upper/row_major"<<std::endl;
	testTriangularEqualUpper(matrix3,result3,rows);
	std::cout<<"test upper/column_major"<<std::endl;
	testTriangularEqualUpper(matrix4,result4,rows);
}

//check that copy-constructor works
BOOST_AUTO_TEST_CASE( triangular_copy_assign){
	triangular_matrix<int,row_major,lower> matrix11(matrix1);
	triangular_matrix<int,row_major,lower> matrix11exp(trans(matrix4));
	triangular_matrix<int,row_major,lower> matrix12(matrix2);
	
	triangular_matrix<int,column_major,lower> matrix21(matrix1);
	triangular_matrix<int,column_major,lower> matrix22(matrix2);
	triangular_matrix<int,column_major,lower> matrix22exp(trans(matrix3));
	
	triangular_matrix<int,row_major,upper> matrix33(matrix3);
	triangular_matrix<int,row_major,upper> matrix33exp(trans(matrix2));
	triangular_matrix<int,row_major,upper> matrix34(matrix4);
	
	triangular_matrix<int,column_major,upper> matrix43(matrix3);
	triangular_matrix<int,column_major,upper> matrix44(matrix4);
	triangular_matrix<int,column_major,upper> matrix44exp(trans(matrix1));
	
	std::cout<<"test copying"<<std::endl;
	std::cout<<"test 11"<<std::endl;
	testTriangularEqualLower(matrix11,result1,rows);
	std::cout<<"test 11 expression"<<std::endl;
	testTriangularEqualLower(matrix11exp,result1,rows);
	std::cout<<"test 12"<<std::endl;
	testTriangularEqualLower(matrix12,result2,rows);
	std::cout<<"test 21"<<std::endl;
	testTriangularEqualLower(matrix21,result1,rows);
	std::cout<<"test 22"<<std::endl;
	testTriangularEqualLower(matrix22,result2,rows);
	std::cout<<"test 22 expression"<<std::endl;
	testTriangularEqualLower(matrix22exp,result2,rows);
	
	std::cout<<"test 33"<<std::endl;
	testTriangularEqualUpper(matrix33,result3,rows);
	std::cout<<"test 33 expression"<<std::endl;
	testTriangularEqualUpper(matrix33exp,result3,rows);
	std::cout<<"test 34"<<std::endl;
	testTriangularEqualUpper(matrix34,result4,rows);
	std::cout<<"test 43"<<std::endl;
	testTriangularEqualUpper(matrix43,result3,rows);
	std::cout<<"test 44"<<std::endl;
	testTriangularEqualUpper(matrix44,result4,rows);
	std::cout<<"test 44 expression"<<std::endl;
	testTriangularEqualUpper(matrix44exp,result4,rows);
}

//check that assignment operator works
BOOST_AUTO_TEST_CASE( triangular_op_assignment){
	triangular_matrix<int,row_major,lower> matrix11;
	matrix11=matrix1;
	triangular_matrix<int,row_major,lower> matrix11exp;
	matrix11exp=trans(matrix4);
	triangular_matrix<int,row_major,lower> matrix12;
	matrix12=matrix2;

	
	std::cout<<"test op="<<std::endl;
	std::cout<<"test 11"<<std::endl;
	testTriangularEqualLower(matrix11,result1,rows);
	std::cout<<"test 11 expression"<<std::endl;
	testTriangularEqualLower(matrix11exp,result1,rows);
	std::cout<<"test 12"<<std::endl;
	testTriangularEqualLower(matrix12,result2,rows);
}

BOOST_AUTO_TEST_SUITE_END()
