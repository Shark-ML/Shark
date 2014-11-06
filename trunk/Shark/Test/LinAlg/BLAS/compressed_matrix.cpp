#define BOOST_TEST_MODULE LinAlg_MatrixSparse
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;
using namespace blas;

struct Element{
	std::size_t row;
	std::size_t column;
	int value;
	Element(){}
	Element(std::size_t r, std::size_t col, double v):row(r),column(col),value(v){}
};

//checks the internal memory structure of the matrix and ensures that it stores the same elements as 
//are given in the vector.
void checkCompressedMatrixStructure(std::vector<Element> const& elements, compressed_matrix<int> const& matrix){
	std::size_t rows = matrix.size1();
	
	//check storage invariants
	//BOOST_REQUIRE_EQUAL(matrix. outer_indices_end()[rows-1], matrix.outer_indices()[rows]);
	BOOST_REQUIRE_EQUAL( matrix.nnz(), elements.size());
	BOOST_REQUIRE( matrix.nnz_capacity() >= elements.size());
	BOOST_REQUIRE_EQUAL(matrix.outer_indices()[0], 0);
	
	std::size_t elem = 0;
	for(std::size_t i = 0; i != rows; ++i){
		//find row end in the array
		std::size_t elem_end = elem;
		while(elem_end != elements.size() && elements[elem_end].row == i)++elem_end;
		std::size_t rowSize = elem_end-elem;
		
		//check row invariants
		BOOST_REQUIRE(matrix.row_capacity(i) >= rowSize );
		BOOST_REQUIRE(matrix.outer_indices()[i+1] >= matrix.outer_indices()[i] );//values are increasing
		BOOST_REQUIRE(matrix.outer_indices_end()[i] >= matrix.outer_indices()[i] );//end is bigger than start
		BOOST_REQUIRE(matrix.outer_indices_end()[i] <= matrix.outer_indices()[i+1] );//end of previous is smaller equal than start of next
		BOOST_REQUIRE_EQUAL(matrix.inner_nnz(i), rowSize );
		BOOST_REQUIRE_EQUAL(matrix.outer_indices_end()[i] - matrix.outer_indices()[i], rowSize);
		BOOST_REQUIRE_EQUAL(matrix.outer_indices()[i+1] - matrix.outer_indices()[i], matrix.row_capacity(i));
		BOOST_REQUIRE(matrix.inner_nnz(i) <= matrix.row_capacity(i));
		BOOST_REQUIRE_EQUAL(matrix.row_end(i) - matrix.row_begin(i),rowSize);
		BOOST_REQUIRE_EQUAL(matrix.row_begin(i).row(), i);
		BOOST_REQUIRE_EQUAL(matrix.row_end(i).row(), i);
		
		
		//check row elements
		std::size_t rowIndex = matrix.outer_indices()[i];
		for(compressed_matrix<int>::const_row_iterator pos = matrix.row_begin(i); pos != matrix.row_end(i); 
			++pos,++elem,++rowIndex
		){
			//check array
			BOOST_CHECK_EQUAL(matrix.inner_indices()[rowIndex],elements[elem].column);
			BOOST_CHECK_EQUAL(matrix.values()[rowIndex],elements[elem].value);
			//check iterator
			BOOST_CHECK_EQUAL(pos.index(),elements[elem].column);
			BOOST_CHECK_EQUAL(*pos,elements[elem].value);
		}
	}
	
	
}

void checkRowSizes(std::vector<std::size_t> const& rowSizes, compressed_matrix<int> const& matrix){
	std::size_t rowStart = 0;
	for(std::size_t i = 0; i != rowSizes.size(); ++i){
		BOOST_CHECK_EQUAL(matrix.outer_indices()[i], rowStart);
		BOOST_CHECK(matrix.row_capacity(i) >= rowSizes[i]);
		BOOST_CHECK(matrix.outer_indices_end()[i] >= rowStart);
		rowStart += rowSizes[i];
	}
	//BOOST_CHECK_EQUAL(matrix.outer_indices()[rowSizes.size()], matrix.outer_indices_end()[rowSizes.size()-1]);
	BOOST_CHECK(matrix.nnz_capacity() >= rowStart);
}


//tests whether reserve calls are correct
BOOST_AUTO_TEST_SUITE (LinAlg_BLAS_compressed_matrix)

BOOST_AUTO_TEST_CASE( LinAlg_sparse_matrix_reserve_row){
	std::size_t rows = 11;//Should be prime :)
	std::size_t columns = 30;
	std::size_t base = 8;
	compressed_matrix<int> matrix(rows,columns);
	std::vector<std::size_t> rowSizes(rows,0);
	
	std::size_t i = 1;
	for(std::size_t j = 0; j != columns; ++j){
		i = (i*base)% rows;
		rowSizes[i]=j;
		matrix.reserve_row(i,j);
		checkRowSizes(rowSizes,matrix);
	}
}

//this test tests push_back behavior of set_element and operator()
BOOST_AUTO_TEST_CASE( LinAlg_sparse_matrix_insert_element_end){
	std::size_t rows = 10;
	std::size_t columns = 20;
	
	compressed_matrix<int> matrix_set(rows,columns);
	compressed_matrix<int> matrix_operator(rows,columns);
	std::vector<Element> elements;
	for(std::size_t i = 0; i != rows; ++i){
		compressed_matrix<int>::row_iterator row_iter = matrix_set.row_begin(i);
		for(std::size_t j = 0; j != columns; ++j,++row_iter){
			BOOST_REQUIRE_EQUAL(row_iter.row(),i);
			row_iter = matrix_set.set_element(row_iter,j,i+j);
			matrix_operator(i,j) = i+j;
			elements.push_back(Element(i,j,i+j));
			checkCompressedMatrixStructure(elements,matrix_set);
			checkCompressedMatrixStructure(elements,matrix_operator);
		}
	}
}

//we still insert row by row, but now with different gaps between indices.
BOOST_AUTO_TEST_CASE( LinAlg_sparse_matrix_insert_element_some_elements ){
	std::size_t rows = 10;
	std::size_t columns = 23;//must be prime
	std::size_t colElements = 3;
	std::vector<std::size_t> cols(colElements);
	
	std::vector<Element> elements;
	compressed_matrix<int> matrix_set(rows,columns);
	compressed_matrix<int> matrix_operator(rows,columns);
	std::size_t base = 8;
	std::size_t l = 1;
	for(std::size_t i = 0; i != rows; ++i){
		for(std::size_t k = 0; k != colElements; ++k){
			l = (l*base)%columns;//field theory gives us a nice "random like" order
			cols[k] = l;
		}
		
		std::sort(cols.begin(),cols.end());
		compressed_matrix<int>::row_iterator row_iter = matrix_set.row_begin(i);
		for(std::size_t elem = 0; elem != colElements; ++elem,++row_iter){
			std::size_t j = cols[elem];
			row_iter = matrix_set.set_element(row_iter,j,i+j);
			matrix_operator(i,j) = i+j;
			elements.push_back(Element(i,j,i+j));
			checkCompressedMatrixStructure(elements,matrix_set);
			checkCompressedMatrixStructure(elements,matrix_operator);
		}
	}
}



BOOST_AUTO_TEST_SUITE_END()
