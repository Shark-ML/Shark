#define BOOST_TEST_MODULE LinAlg_VectorSparse
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;
using namespace blas;



//this test tests push_back behavior of set_element and operator()
BOOST_AUTO_TEST_SUITE (LinAlg_BLAS_compressed_vector)

BOOST_AUTO_TEST_CASE( LinAlg_sparse_vector_insert_element_end){
	std::size_t dimensions = 20;
	
	compressed_vector<std::size_t> vector_set(dimensions);
	compressed_vector<std::size_t> vector_operator(dimensions);
	compressed_vector<std::size_t>::iterator iter = vector_set.begin();
	for(std::size_t i = 0; i != dimensions; ++i){
		//check set_element
		iter = vector_set.set_element(iter,i,3*i);
		BOOST_REQUIRE_EQUAL(iter.index(), i);
		BOOST_REQUIRE_EQUAL(iter-vector_set.begin(), i);
		BOOST_REQUIRE_EQUAL(*iter, 3*i);
		BOOST_REQUIRE_EQUAL(vector_set.nnz(),i+1);
		for(std::size_t k = 0; k <=i; ++k){
			BOOST_CHECK_EQUAL(vector_set.values()[k], 3*k);
			BOOST_CHECK_EQUAL(vector_set.indices()[k], k);
		}
		++iter;
		BOOST_REQUIRE_EQUAL(iter-vector_set.begin(), i+1);
	}
	
	for(std::size_t i = 0; i != dimensions; ++i){
		//check operator()
		vector_operator(i) = 3*i;
		BOOST_REQUIRE_EQUAL(vector_operator(i), 3*i);
		BOOST_REQUIRE_EQUAL(vector_operator.nnz(),i+1);
		for(std::size_t k = 0; k <=i; ++k){
			BOOST_CHECK_EQUAL(vector_operator.values()[k], 3*k);
			BOOST_CHECK_EQUAL(vector_operator(k), 3*k);
			BOOST_CHECK_EQUAL(vector_operator.indices()[k], k);
		}
	}
}

//~ //we still insert row by row, but now with different gaps between indices.
//~ BOOST_AUTO_TEST_CASE( LinAlg_sparse_matrix_insert_element_some_elements ){
	//~ std::size_t rows = 10;
	//~ std::size_t columns = 23;//must be prime
	//~ std::size_t colElements = 3;
	//~ std::vector<std::size_t> cols(colElements);
	
	//~ std::vector<Element> elements;
	//~ compressed_matrix<int> matrix_set(rows,columns);
	//~ compressed_matrix<int> matrix_operator(rows,columns);
	//~ std::size_t base = 8;
	//~ std::size_t l = 1;
	//~ for(std::size_t i = 0; i != rows; ++i){
		//~ for(std::size_t k = 0; k != colElements; ++k){
			//~ l = (l*base)%columns;//field theory gives us a nice "random like" order
			//~ cols[k] = l;
		//~ }
		
		//~ std::sort(cols.begin(),cols.end());
		//~ compressed_matrix<int>::row_iterator row_iter = matrix_set.row_begin(i);
		//~ for(std::size_t elem = 0; elem != colElements; ++elem,++row_iter){
			//~ std::size_t j = cols[elem];
			//~ row_iter = matrix_set.set_element(row_iter,j,i+j);
			//~ matrix_operator(i,j) = i+j;
			//~ elements.push_back(Element(i,j,i+j));
			//~ checkCompressedMatrixStructure(elements,matrix_set);
			//~ checkCompressedMatrixStructure(elements,matrix_operator);
		//~ }
	//~ }
//~ }



BOOST_AUTO_TEST_SUITE_END()
